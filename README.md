# inspect-tinker

A [Tinker](https://tinker-docs.thinkingmachines.ai)-backed **model provider for
[Inspect AI](https://inspect.aisi.org.uk)**. It lets any Inspect eval — including
Redwood's ControlTower — drive a model that lives on Tinker (a LoRA checkpoint
reachable only through Tinker's Python SDK) as if it were an ordinary hosted model,
addressed as `tinker/<checkpoint-path>`.

**No local GPU.** The provider renders Inspect's messages + tools with the base
model's chat template, calls Tinker's `sample_async` (Tinker's servers do the
compute), and parses the model's Hermes/Qwen-style `<tool_call>` output back into
Inspect `ToolCall`s.

## Why

Inspect and ControlTower talk to models over a provider interface; Tinker only
exposes a Python SDK. Rather than stand up a vLLM server (needs a GPU, and may not
support brand-new architectures) or a separate OpenAI-compatible proxy, this plugs
Tinker straight into Inspect **in-process** — no extra service, no public endpoint.

## Install

```bash
uv add inspect-tinker          # or: pip install inspect-tinker
```

Installing it is enough — Inspect discovers the provider via entry points, so
`get_model("tinker/…")` and `-p model=tinker/…` just work.

## Use

Set your Tinker key (the provider reads `TINKER_API_KEY` from the environment):

```bash
export TINKER_API_KEY=...        # or put it in a .env your harness loads
```

The provider needs to know the **base model** to load the right tokenizer / chat
template (the `tinker://` path doesn't encode it). Resolution order:

1. the `base_model` model-arg, e.g. `-M base_model=Qwen/Qwen3.5-9B`
2. the `INSPECT_TINKER_BASE_MODEL` environment variable
3. the built-in `DEFAULT_BASE_MODEL`

### With Inspect directly

```python
from inspect_ai.model import get_model

model = get_model(
    "tinker/tinker://<run>:train:0/sampler_weights/runbest",
    base_model="Qwen/Qwen3.5-9B",   # passed through as a model-arg
)
```

### With ControlTower (attack policy)

```bash
export INSPECT_TINKER_BASE_MODEL=Qwen/Qwen3.5-9B
uv run ct run eval --no-upload --policy attack \
  -e basharena -t <MAIN_TASK> -s <SIDE_TASK> \
  -p model=tinker/tinker://<run>:train:0/sampler_weights/runbest \
  -p model_attempt_timeout=600 --max-samples 1
```

ControlTower passes unknown provider prefixes straight through to Inspect, so no
fork or config change is needed — just install this package alongside it.

## Notes & limitations

- **Tool-call format.** Output is parsed as Hermes/Qwen `<tool_call>{json}</tool_call>`
  blocks (matching `--tool-call-parser hermes`). Models from other families that
  emit a different tool-call syntax would need a different parser.
- **Sampling params.** `max_tokens`, `temperature`, `top_p`, and `stop` from the
  eval's `GenerateConfig` are forwarded; `<|im_end|>` is always added as a stop.
- **Thinking.** Off by default (matches an actions-only SFT). Pass
  `-M enable_thinking=true` to enable it if your base template supports it.
- **One checkpoint per model handle.** Point at a different `tinker://` path to
  serve a different checkpoint.
- **Base vs tuned.** A plain HF id serves the *untuned* base model through the same
  path — `tinker/Qwen/Qwen3.5-9B` — so the baseline arm of a base-vs-tuned comparison
  goes through identical rendering/parsing as `tinker/tinker://<checkpoint>`.

## Develop / test

```bash
uv sync --extra dev
uv run pytest          # offline: message/tool conversion + tool-call parsing + registration
```

The tests never touch Tinker or the network. A live end-to-end check needs a real
`TINKER_API_KEY` and a valid checkpoint.

## Publish

Fill in `[project.urls]` in `pyproject.toml`, then:

```bash
uv build
uv publish
```

The natural upstream home for a Tinker provider is Inspect itself; this standalone
package works today without waiting on that.

## License

MIT — see [LICENSE](LICENSE).
