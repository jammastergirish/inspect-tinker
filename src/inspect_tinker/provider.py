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

# Kimi (moonshotai/Kimi-*) tool calls: each call carries a ``functions.NAME:IDX`` id
# wrapped in special tokens, e.g.
#   <|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0
#   <|tool_call_argument_begin|>{"cmd": "ls /home"}<|tool_call_end|><|tool_calls_section_end|>
# The Tinker server's tokenizer numbers those section tokens differently from the local
# HF tokenizer, so on the way back they decode to *other* special-token names (the
# argument/end markers surface as <|im_middle|>/<|media_begin|>, and the begin markers
# survive verbatim). The ``functions.NAME:IDX`` id and the JSON argument object are
# ordinary text that decodes faithfully, so we key parsing off those — find each id,
# then the balanced JSON object after it — rather than trusting the delimiters.
_KIMI_ID_RE = re.compile(r"functions\.([A-Za-z0-9_.\-]+):(\d+)")
# Kimi section/call delimiters, plus the tokens the local tokenizer mis-decodes them
# into; stripped from the returned content.
_KIMI_MARKER_RE = re.compile(
    r"<\|(?:tool_calls_section_begin|tool_calls_section_end|tool_call_begin|"
    r"tool_call_argument_begin|tool_call_end|im_middle|media_begin|media_end|"
    r"media_content|media_pad)\|>"
)

# gpt-oss (OpenAI *harmony* channel format). A tool call is emitted in the ``commentary``
# channel as ``<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json
# <|message|>{...}<|call|>``; reasoning goes to ``analysis`` and the user-facing answer to
# ``final``. On the Tinker serving path the harmony special tokens (``<|start|>``,
# ``<|channel|>``, ``<|message|>``, ``<|constrain|>``, ``<|call|>``, ``<|end|>``) decode
# WITHOUT their delimiters, so the channels collapse into bare words: the completion opens
# on ``analysis<reasoning>``, a tool call surfaces as
# ``assistantcommentary to=functions.NAME json{...}`` (the ``assistant`` is the role token
# from ``<|start|>assistant``, and the ``json`` constrain may abut the name, e.g.
# ``to=functions.submitjson{}``), and the answer as ``assistantfinal<text>``. The
# ``to=functions.`` recipient and the balanced JSON object decode faithfully, so we key
# extraction off those (raw_decode for the args, robust to nested/brace-like text) rather
# than trusting the collapsed delimiters.
_HARMONY_RECIPIENT_RE = re.compile(r"to=functions\.([A-Za-z0-9_.\-]+)")
# Between the recipient and its ``{`` args only the (optional) ``json`` constrain and
# whitespace may sit; anything else means the ``{`` isn't this call's argument object.
_HARMONY_GAP_RE = re.compile(r"\s*(?:json)?\s*")
# Channel-marker words in collapsed harmony output: the leading channel word (the
# ``<|start|>assistant`` sat in the prompt, so the completion opens bare) and every later
# channel introduced by the ``assistant`` role token glued to its channel name. Used to
# split the residual analysis/final prose and — in its glued ``assistant...`` form only —
# to detect a tool-call-free harmony turn without misfiring on ordinary prose.
_HARMONY_MARK_RE = re.compile(
    r"assistant(?:analysis|commentary|final)|^(?:analysis|commentary|final)"
)
_HARMONY_ASSISTANT_RE = re.compile(r"assistant(?:analysis|commentary|final)")


# --------------------------------------------------------------------------- #
# Pure helpers (no network / no tokenizer) — unit-tested in tests/.
# --------------------------------------------------------------------------- #
def message_to_template(msg: Any, family: str | None = None) -> dict[str, Any]:
    """Convert one Inspect ChatMessage into the dict a HF chat template expects.

    ``family`` selects family-specific quirks in how prior turns are rendered back
    into the prompt. For ``"kimi"`` the chat template prints ``tool_call['id']``
    verbatim between the call-begin/argument-begin tokens and the model is trained on
    ids of the form ``functions.NAME:IDX`` — so the function name must be carried in
    the id or it is lost from the rendered history; tool results likewise reference
    that id in their ``## Return of ...`` header.
    """
    role = msg.role
    text = msg.text or ""
    if role == "assistant" and getattr(msg, "tool_calls", None):
        tool_calls = []
        for idx, tc in enumerate(msg.tool_calls):
            entry: dict[str, Any] = {
                "type": "function",
                "function": {"name": tc.function, "arguments": tc.arguments},
            }
            if family == "kimi":
                entry["id"] = f"functions.{tc.function}:{idx}"
            tool_calls.append(entry)
        return {"role": "assistant", "content": text, "tool_calls": tool_calls}
    if role == "tool":
        name = getattr(msg, "function", None) or ""
        out: dict[str, Any] = {"role": "tool", "content": text, "name": name}
        if family == "kimi":
            out["tool_call_id"] = f"functions.{name}:0"
        return out
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


def _kimi_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Kimi (moonshotai/Kimi-*) emits each tool call as a ``functions.NAME:IDX`` id
    followed by a JSON argument object, wrapped in ``<|tool_call...|>`` section tokens.
    Those delimiter tokens are unreliable here (the server tokenizer numbers them
    differently from the local one, so they decode to mismatched names), but the id and
    the JSON are plain text that decodes faithfully — so key off each id, then pull the
    balanced JSON object that follows (via ``raw_decode``, so braces inside a string
    don't break it). A block whose arguments won't parse is preserved with a
    ``parse_error`` rather than dropped, and the (usually empty) content before the
    first call is returned separately with the section tokens stripped."""
    dec = json.JSONDecoder()
    ids = list(_KIMI_ID_RE.finditer(text))
    calls: list[ToolCall] = []
    for n, m in enumerate(ids):
        fname = m.group(1)
        cid = uuid.uuid4().hex[:8]
        # The args JSON lives between this id and the next one (or the end).
        seg_end = ids[n + 1].start() if n + 1 < len(ids) else len(text)
        segment = text[m.end() : seg_end]
        brace = segment.find("{")  # skip any delimiter tokens between id and args
        if brace == -1:
            calls.append(
                ToolCall(
                    id=cid,
                    function=fname,
                    arguments={},
                    parse_error="no JSON argument object found",
                )
            )
            continue
        try:
            obj, _ = dec.raw_decode(segment, brace)
            if not isinstance(obj, dict):
                raise TypeError("tool-call arguments were not a JSON object")
            calls.append(ToolCall(id=cid, function=fname, arguments=obj))
        except (json.JSONDecodeError, TypeError) as e:
            calls.append(
                ToolCall(id=cid, function=fname, arguments={}, parse_error=str(e))
            )

    if not ids:
        return [], text
    # Content is whatever precedes the first tool-call indicator (a section token or the
    # first id), with any leftover reasoning/section tokens stripped.
    cut = ids[0].start()
    for mark in _KIMI_MARKER_RE.finditer(text):
        cut = min(cut, mark.start())
    content = _KIMI_MARKER_RE.sub("", _THINK_RE.sub("", text[:cut])).strip()
    return calls, content


def _harmony_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """gpt-oss (OpenAI *harmony*) emits each tool call in the ``commentary`` channel as
    ``... to=functions.NAME json{...}``; on the Tinker serving path the harmony special
    tokens decode without their delimiters, so channels collapse into bare words (see the
    module-level note). Extract every ``to=functions.NAME`` recipient and the balanced JSON
    object that follows it (via ``raw_decode``, so nested braces / brace-like text inside a
    string argument don't break it), then return the residual ``analysis``/``final`` prose
    with those tool-call spans excised and the channel-marker words stripped.

    Args JSON may be ``{}`` or abut the constrain (``to=functions.submitjson{}`` → tool
    ``submit``). A recipient whose arguments won't parse is preserved with a ``parse_error``
    rather than dropped. Returns ``([], text)`` unchanged when no recipient is found."""
    dec = json.JSONDecoder()
    calls: list[ToolCall] = []
    out: list[str] = []  # residual prose, with each tool-call span excised
    i = 0
    for m in _HARMONY_RECIPIENT_RE.finditer(text):
        if m.start() < i:
            continue  # a `to=functions.` sitting inside a prior call's parsed args
        name = m.group(1)
        brace = text.find("{", m.end())
        if brace == -1:
            continue  # a recipient with no JSON argument object → not a well-formed call
        gap = text[m.end() : brace]
        if not _HARMONY_GAP_RE.fullmatch(gap):
            continue  # only the `json` constrain may sit between recipient and its args
        # The `json` constrain can abut the tool name (`functions.submitjson{}`); the name
        # regex greedily swallowed it, so strip a trailing `json` when nothing but the
        # constrain sits between the recipient and the `{`.
        if gap.strip() == "" and name.endswith("json") and name != "json":
            name = name[: -len("json")]
        cid = uuid.uuid4().hex[:8]
        try:
            obj, end = dec.raw_decode(text, brace)
            if not isinstance(obj, dict):
                raise TypeError("tool-call arguments were not a JSON object")
            calls.append(ToolCall(id=cid, function=name, arguments=obj))
        except (json.JSONDecodeError, TypeError) as e:
            end = brace
            calls.append(
                ToolCall(id=cid, function=name, arguments={}, parse_error=str(e))
            )
        out.append(text[i : m.start()])
        i = end
    out.append(text[i:])
    # The residual still carries the channel-marker words (leading analysis/commentary/final
    # and the glued assistant<channel> markers); split on those and keep the non-empty
    # analysis/final bodies as content.
    parts = _HARMONY_MARK_RE.split("".join(out))
    content = "\n".join(p.strip() for p in parts if p and p.strip())
    return calls, content


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split Qwen/Hermes/GLM/Inkling/Kimi/gpt-oss output into (content, tool_calls).

    Handles the syntaxes these models emit — Hermes JSON
    (``<tool_call>{"name":..,"arguments":{..}}</tool_call>``), the Qwen XML form
    (``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>..</function></tool_call>``,
    which Qwen3.5 uses), the GLM form
    (``<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>..</tool_call>``),
    the Kimi form (a ``functions.NAME:IDX`` id + JSON args inside ``<|tool_call...|>``
    section tokens), and the gpt-oss/harmony form (``... to=functions.NAME json{...}`` with
    the harmony channel delimiters decoded away). A leading ``<think>..</think>`` reasoning
    block is stripped from the content. Malformed blocks are preserved with a ``parse_error``
    rather than dropped.

    ``parse_tool_calls`` receives only the decoded completion (no family flag), so each
    family is detected from markers unique to its output; the checks are ordered and gated
    so a match for one family can't disturb another's parse path.
    """
    # gpt-oss (OpenAI harmony): the ``to=functions.`` recipient and the glued
    # ``assistant<channel>`` markers appear in none of the Qwen/Hermes/GLM/Inkling/Kimi
    # forms, so keying off them is safe. Checked before Kimi because a harmony argument value
    # could otherwise contain a ``functions.NAME:IDX``-shaped substring and trip Kimi's probe.
    if "to=functions." in text:
        harmony_calls, harmony_content = _harmony_tool_calls(text)
        if harmony_calls:
            return harmony_content, harmony_calls
    elif _HARMONY_ASSISTANT_RE.search(text):
        # A tool-call-free harmony turn (analysis + final, no recipient): strip the channel
        # markers and return the prose. Gated on the glued ``assistant<channel>`` marker
        # (not a bare leading channel word) so ordinary prose isn't misread as harmony.
        _, harmony_content = _harmony_tool_calls(text)
        return harmony_content, []

    # Kimi (moonshotai/Kimi-*): a functions.NAME:IDX id followed by JSON args. Its syntax
    # shares no markers with the Qwen/GLM/Inkling forms, so detect it up front and only
    # take this path when it actually yields calls.
    if _KIMI_ID_RE.search(text):
        kimi_calls, kimi_content = _kimi_tool_calls(text)
        if kimi_calls:
            return kimi_content, kimi_calls

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
        # enable_thinking is a model-arg, but Control Tower's `ct run eval` doesn't expose inspect's
        # `-M` to set model-args, so also honour an env fallback (set by the agentic-OPD rollout for
        # CoT-on): INSPECT_TINKER_ENABLE_THINKING=1.
        self.enable_thinking = bool(enable_thinking) or (
            os.environ.get("INSPECT_TINKER_ENABLE_THINKING") == "1"
        )
        # Inkling gates reasoning on a `reasoning_effort` chat-template var
        # (none/minimal/low/medium/high/max, or a float in [0,0.99]; default high). ct cannot pass
        # model-args, so honour an env fallback INSPECT_TINKER_REASONING_EFFORT (unset = template default).
        self.reasoning_effort = os.environ.get("INSPECT_TINKER_REASONING_EFFORT") or None

        # Detect the base model's family from its id; it selects the chat format, the
        # turn terminators, and how tool calls are rendered/parsed.
        #   GLM (zai-org/GLM-*): an assistant turn ends at <|user|>/<|observation|>, not
        #     <|im_end|>, and tool calls use <arg_key>/<arg_value> pairs.
        #   Kimi (moonshotai/Kimi-*): ends at <|im_end|> like Qwen, but declares tools in a
        #     tool_declare block, gates reasoning on `thinking`, and emits functions.NAME:IDX
        #     tool-call ids.
        self.is_glm = "glm" in self.base_model.lower()
        self.is_kimi = "kimi" in self.base_model.lower()
        self.is_inkling = "inkling" in self.base_model.lower()
        self.family = "glm" if self.is_glm else ("kimi" if self.is_kimi else None)
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
        # Qwen/GLM templates gate reasoning on `enable_thinking`; Kimi's gates it on
        # `thinking`. Pass both so the (default no-CoT) behaviour is honoured everywhere;
        # a template ignores the kwarg it doesn't use.
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": self.enable_thinking,
        }
        if self.is_kimi:
            kwargs["thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            text = self.tokenizer.apply_chat_template(msgs, tools=schemas, **kwargs)
        except TypeError:
            # tokenizer template doesn't accept the thinking kwargs
            text = self.tokenizer.apply_chat_template(
                msgs, tools=schemas, add_generation_prompt=True, tokenize=False
            )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render(self, input: list[Any], tools: list[ToolInfo], drop_tools: bool):
        msgs = [message_to_template(m, self.family) for m in input]
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
        # Also return the final (post-trim) template message dicts and tool schemas, so on-policy
        # training can re-render the SAME native context (e.g. swapping only the system message for the
        # OPD teacher) rather than a divergent SFT rendering. Callers that only want ids ignore the rest.
        return ids, head + rest, schemas

    async def generate(
        self,
        input: list[Any],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        from tinker import types

        ids, rendered_msgs, rendered_schemas = self._render(
            input, tools, drop_tools=(tool_choice == "none")
        )

        stop = list(config.stop_seqs or [])
        # Inkling/GLM run tokenizer-less on the Tinker server (skip_tokenizer_init=True — their native
        # tokenizer routes through the unavailable tml_tokenizers), so it REJECTS *string* stop-sequences
        # ("stop=[...] is unavailable when skip_tokenizer_init=True (requires tokenizer to decode tokens to
        # text for matching)") and every sample fails with 0 tokens out. Omit the string turn-stops for those
        # families and rely on max_tokens + the client-side tool-call parse (parse_tool_calls already keys off
        # the faithfully-decoding id+JSON, robust to the server/local special-token id mismatch): the model
        # still emits its action, the parser extracts it. Every other model keeps its turn stops.
        if not (self.is_inkling or self.is_glm):
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
            stop=stop or None,
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

        # On-policy capture (gated on INSPECT_TINKER_CAPTURE_IDS=1, set by the agentic OPD/RL rollout):
        # record the EXACT native prompt token-ids fed to the sampler + the sampled completion ids, plus
        # the rendered message-dicts + tool schemas so the teacher context re-renders identically (only
        # the system message swapped). Lets training score the sequence the model actually sampled under
        # rather than a divergent SFT re-render. Off by default so ordinary evals aren't bloated.
        md = None
        if os.environ.get("INSPECT_TINKER_CAPTURE_IDS") == "1":
            md = {
                "tinker_prompt_ids": list(ids),
                "tinker_completion_ids": list(seq.tokens),
                "tinker_msgs": rendered_msgs,
                "tinker_schemas": rendered_schemas,
                "tinker_enable_thinking": bool(self.enable_thinking),
                "tinker_reasoning_effort": self.reasoning_effort,
            }
        message = ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls or None,
            model=self.model_name,
            source="generate",
            metadata=md,
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
