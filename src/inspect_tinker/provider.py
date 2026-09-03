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

# Hermes / Qwen tool-call block: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split Hermes/Qwen output into (content, tool_calls).

    Every ``<tool_call>{json}</tool_call>`` block becomes a ToolCall; malformed
    JSON is preserved with a ``parse_error`` rather than dropped. The remaining
    text (blocks removed) is the assistant content.
    """
    calls: list[ToolCall] = []
    for m in _TOOLCALL_RE.finditer(text):
        raw = m.group(1)
        cid = uuid.uuid4().hex[:8]
        try:
            obj = json.loads(raw)
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
    content = _TOOLCALL_RE.sub("", text).strip()
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

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

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

    def _render(self, input: list[Any], tools: list[ToolInfo], drop_tools: bool):
        msgs = [message_to_template(m) for m in input]
        # Qwen's chat template requires a single system message at the very start.
        # Control policies can inject the side-task system prompt mid-conversation,
        # so merge every system message into one leading block and keep the rest in
        # order — otherwise the template raises "System message must be at the beginning".
        sys_parts = [
            m["content"] for m in msgs if m.get("role") == "system" and m.get("content")
        ]
        if sys_parts:
            rest = [m for m in msgs if m.get("role") != "system"]
            msgs = [{"role": "system", "content": "\n\n".join(sys_parts)}] + rest
        schemas = None if drop_tools else ([tool_to_schema(t) for t in tools] or None)
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
        if TURN_END not in stop:
            stop.append(TURN_END)
        max_tokens = config.max_tokens or 1024
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
