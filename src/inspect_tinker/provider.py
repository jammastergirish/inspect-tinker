"""A Tinker-backed Inspect AI model provider.

Lets any Inspect eval (including Redwood's ControlTower) drive a model that lives
on Tinker — a LoRA checkpoint reached only through Tinker's Python SDK — as if it
were an ordinary hosted model. Use it as ``tinker/<tinker-checkpoint-path>``.

The provider renders Inspect's messages + tools with the base model's chat
template, calls Tinker's ``sample_async`` (Tinker's servers do the compute — no
local GPU), and parses the model's Hermes/Qwen-style ``<tool_call>`` output back
into Inspect ``ToolCall`` objects.

The base model (needed only to load the tokenizer / chat template) is taken from
the ``base_model`` model-arg, else ``$INSPECT_TINKER_BASE_MODEL``, else
``DEFAULT_BASE_MODEL``. The Tinker API key is read from ``$TINKER_API_KEY``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
TURN_END = "<|im_end|>"  # Qwen assistant-turn terminator
DEFAULT_MAX_CONTEXT = (
    65536  # model context window (override: INSPECT_TINKER_MAX_CONTEXT)
)
_CONTEXT_RESERVE = 1024  # tokens kept free for generation when fitting the prompt

# Hermes / Qwen tool-call block: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
# Qwen emits tool calls in two syntaxes; we parse both:
#   Hermes JSON:  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
#   Qwen XML:     <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
# Qwen3.5 uses the XML form under tool use.
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

# GLM (zai-org/GLM-*) tool call: the function name sits bare after <tool_call>, then
# <arg_key>/<arg_value> pairs, e.g.
#   <tool_call>bash<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL
)
# GLM emits a (usually empty, since we train no-CoT) reasoning block before the call.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Pure helpers (no network / no tokenizer) — unit-tested in tests/.
# --------------------------------------------------------------------------- #
def message_to_template(msg: Any) -> dict[str, Any]:
    """Convert one Inspect ChatMessage into the dict a HF chat template expects."""
    role = msg.role
    text = msg.text or ""
    if role == "assistant" and getattr(msg, "tool_calls", None):
        return {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": tc.function, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
    if role == "tool":
        return {
            "role": "tool",
            "content": text,
            "name": getattr(msg, "function", None) or "",
        }
    return {"role": role, "content": text}


def tool_to_schema(tool: ToolInfo) -> dict[str, Any]:
    """Convert an Inspect ToolInfo into an OpenAI-style function schema."""
    params = tool.parameters
    if hasattr(params, "model_dump"):
        params_dict = params.model_dump(exclude_none=True)
    elif params:
        params_dict = dict(params)
    else:
        params_dict = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": params_dict,
        },
    }


def _xml_tool_call(fname: str, body: str) -> ToolCall:
    """Build a ToolCall from a Qwen XML <function=..><parameter=..> body."""
    args = {k.strip(): v.strip() for k, v in _PARAM_RE.findall(body)}
    return ToolCall(id=uuid.uuid4().hex[:8], function=fname.strip(), arguments=args)


def _glm_tool_call(inner: str) -> ToolCall:
    """Build a ToolCall from a GLM <tool_call> body: the bare function name, then
    <arg_key>/<arg_value> pairs. A call with no arguments is just the name."""
    fname = inner.split("<arg_key>", 1)[0].strip()
    args = {k.strip(): v.strip() for k, v in _GLM_ARG_RE.findall(inner)}
    return ToolCall(id=uuid.uuid4().hex[:8], function=fname, arguments=args)


_INK_START = re.compile(r'\{\s*"name"\s*:')


def _inkling_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Inkling (thinkingmachines/Inkling) emits tool calls inline with no wrapper, as
    ``NAME{"name":"NAME","args":{...}}``. Pull each JSON object out (balanced, via
    ``raw_decode`` so braces inside a command don't break it) and strip it — plus the
    duplicated bare tool-name token right before it — from the returned content."""
    dec = json.JSONDecoder()
    calls: list[ToolCall] = []
    out: list[str] = []
    i = 0
    for m in _INK_START.finditer(text):
        j = m.start()
        if j < i:
            continue
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            continue
        if not (
            isinstance(obj, dict)
            and "name" in obj
            and ("args" in obj or "arguments" in obj)
        ):
            continue
        args = obj.get("args", obj.get("arguments")) or {}
        calls.append(
            ToolCall(
                id=uuid.uuid4().hex[:8],
                function=str(obj.get("name", "")),
                arguments=args if isinstance(args, dict) else {},
            )
        )
        out.append(
            re.sub(r"\w+\s*$", "", text[i:j])
        )  # drop the bare "bash" before the blob
        i = end
    out.append(text[i:])
    return calls, "".join(out).strip()


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split Qwen/Hermes/GLM/Inkling output into (content, tool_calls).

    Handles the three syntaxes these models emit — Hermes JSON
    (``<tool_call>{"name":..,"arguments":{..}}</tool_call>``), the Qwen XML form
    (``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>..</function></tool_call>``,
    which Qwen3.5 uses), and the GLM form
    (``<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>..</tool_call>``).
    A leading ``<think>..</think>`` reasoning block is stripped from the content.
    Malformed blocks are preserved with a ``parse_error`` rather than dropped.
    """
    calls: list[ToolCall] = []
    for m in _TOOLCALL_RE.finditer(text):
        inner = m.group(1).strip()
        cid = uuid.uuid4().hex[:8]
        if inner.startswith("{"):  # Hermes JSON
            try:
                obj = json.loads(inner)
                calls.append(
                    ToolCall(
                        id=cid,
                        function=str(obj.get("name", "")),
                        arguments=obj.get("arguments", {}) or {},
                    )
                )
            except (json.JSONDecodeError, TypeError) as e:
                calls.append(
                    ToolCall(id=cid, function="", arguments={}, parse_error=str(e))
                )
            continue
        if "<function=" in inner:  # Qwen XML
            fm = _FUNC_RE.search(inner)
            if fm:
                calls.append(_xml_tool_call(fm.group(1), fm.group(2)))
                continue
        if "<arg_key>" in inner or "<arg_value>" in inner:  # GLM key/value pairs
            calls.append(_glm_tool_call(inner))
            continue
        if inner and "<" not in inner:  # GLM no-arg call: <tool_call>NAME</tool_call>
            calls.append(ToolCall(id=cid, function=inner, arguments={}))
            continue
        calls.append(
            ToolCall(
                id=cid, function="", arguments={}, parse_error="unrecognized tool_call"
            )
        )

    # Some variants emit bare <function=..> blocks with no <tool_call> wrapper.
    if not calls:
        calls = [_xml_tool_call(fn, body) for fn, body in _FUNC_RE.findall(text)]

    # Inkling: no wrapper at all — NAME{"name":..,"args":{..}} inline. Only as a last resort,
    # so it can never disturb the Qwen/GLM paths above.
    if not calls and '"name"' in text and ('"args"' in text or '"arguments"' in text):
        ink_calls, ink_content = _inkling_tool_calls(text)
        if ink_calls:
            return ink_content, ink_calls

    # Strip tool-call blocks and any (usually empty) GLM/Qwen reasoning block.
    content = _THINK_RE.sub("", _FUNC_RE.sub("", _TOOLCALL_RE.sub("", text))).strip()
    return content, calls


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #
class TinkerAPI(ModelAPI):
    """Inspect ModelAPI that samples from a Tinker checkpoint via the Tinker SDK."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        base_model: str | None = None,
        enable_thinking: bool = False,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["TINKER_API_KEY"],
            config=config,
        )
        # model_name is the part after the "tinker/" provider prefix. Two forms:
        #   tinker://<run>:train:0/sampler_weights/<name>  -> a tuned checkpoint
        #   <hf-id>, e.g. Qwen/Qwen3.5-9B                  -> the untuned base model
        # The base form lets you run the baseline arm of a base-vs-tuned comparison
        # through the same provider (no adapter attached).
        self.checkpoint = model_name
        self.is_base = not model_name.startswith("tinker://")
        if self.is_base:
            self.base_model = base_model or model_name
        else:
            self.base_model = (
                base_model
                or os.environ.get("INSPECT_TINKER_BASE_MODEL")
                or DEFAULT_BASE_MODEL
            )
        self.enable_thinking = bool(enable_thinking)

        # GLM (zai-org/GLM-*) uses a different chat format and turn terminators than
        # Qwen: an assistant turn ends at <|user|>/<|observation|>, not <|im_end|>.
        self.is_glm = "glm" in self.base_model.lower()
        self._turn_stops = (
            ["<|user|>", "<|observation|>"] if self.is_glm else [TURN_END]
        )

        # A Tinker base id may carry a ":peft:<ctx>" suffix (required to *sample* some
        # models, e.g. GLM-5.3) that is not a valid HF id — strip it for the tokenizer,
        # keep the full id for the Tinker sampling client.
        self.tokenizer_id = self.base_model.split(":peft:")[0]

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        import tinker

        self._service = tinker.ServiceClient()
        self._sampler = None  # created lazily on first generate

    async def _sampler_client(self):
        if self._sampler is None:
            if self.is_base:
                self._sampler = await self._service.create_sampling_client_async(
                    base_model=self.base_model
                )
            else:
                self._sampler = await self._service.create_sampling_client_async(
                    model_path=self.checkpoint
                )
        return self._sampler

    def _encode(self, msgs: list[dict], schemas: list[dict] | None) -> list[int]:
        try:
            text = self.tokenizer.apply_chat_template(
                msgs,
                tools=schemas,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # tokenizer template doesn't accept enable_thinking
            text = self.tokenizer.apply_chat_template(
                msgs, tools=schemas, add_generation_prompt=True, tokenize=False
            )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render(self, input: list[Any], tools: list[ToolInfo], drop_tools: bool):
        msgs = [message_to_template(m) for m in input]
        # Qwen's chat template requires a single system message at the very start.
        # Control policies can inject the side-task system prompt mid-conversation,
        # so merge every system message into one leading block and keep the rest in
        # order — otherwise the template raises "System message must be at the beginning".
        sys_parts = [
            m["content"] for m in msgs if m.get("role") == "system" and m.get("content")
        ]
        head = (
            [{"role": "system", "content": "\n\n".join(sys_parts)}] if sys_parts else []
        )
        rest = [m for m in msgs if m.get("role") != "system"]
        schemas = None if drop_tools else ([tool_to_schema(t) for t in tools] or None)

        # Fit the prompt inside the model's context window: keep the system block and
        # the most recent turns, dropping the oldest (and any orphaned leading tool
        # result) until it fits with room to generate. Long agentic episodes otherwise
        # overflow and Tinker returns a 400. Override the window with
        # INSPECT_TINKER_MAX_CONTEXT if the served model differs from the default.
        ctx = int(
            os.environ.get("INSPECT_TINKER_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT))
        )
        ids = self._encode(head + rest, schemas)
        while len(ids) > ctx - _CONTEXT_RESERVE and rest:
            rest = rest[1:]
            while rest and rest[0].get("role") == "tool":
                rest = rest[1:]  # don't leave a tool result with no preceding call
            ids = self._encode(head + rest, schemas)
        return ids

    async def generate(
        self,
        input: list[Any],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        from tinker import types

        ids = self._render(input, tools, drop_tools=(tool_choice == "none"))

        stop = list(config.stop_seqs or [])
        for t in self._turn_stops:
            if t not in stop:
                stop.append(t)
        # Cap generation so prompt + max_tokens stays inside the context window
        # (_render already trims the prompt to leave room).
        ctx = int(
            os.environ.get("INSPECT_TINKER_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT))
        )
        max_tokens = max(16, min(config.max_tokens or 1024, ctx - len(ids) - 8))
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0 if config.temperature is None else config.temperature,
            top_p=1.0 if config.top_p is None else config.top_p,
            stop=stop,
        )

        sampler = await self._sampler_client()
        result = await sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokens=ids),
            num_samples=1,
            sampling_params=params,
        )
        seq = result.sequences[0]
        out_text = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        content, tool_calls = parse_tool_calls(out_text)

        if tool_calls:
            stop_reason = "tool_calls"
        elif len(seq.tokens) >= max_tokens:
            stop_reason = "max_tokens"
        else:
            stop_reason = "stop"

        message = ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls or None,
            model=self.model_name,
            source="generate",
        )
        usage = ModelUsage(
            input_tokens=len(ids),
            output_tokens=len(seq.tokens),
            total_tokens=len(ids) + len(seq.tokens),
        )
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=message, stop_reason=stop_reason)],
            usage=usage,
        )
