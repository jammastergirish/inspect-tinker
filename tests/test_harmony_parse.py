"""Offline tests for the gpt-oss (OpenAI *harmony*) tool-call parser.

gpt-oss-120b emits harmony channel format. On the Tinker serving path the harmony
special tokens (``<|start|>``, ``<|channel|>``, ``<|message|>``, ``<|constrain|>``,
``<|call|>``, ``<|end|>``) decode WITHOUT their delimiters, so the channels collapse
into bare words: the completion opens on ``analysis<reasoning>``, a tool call surfaces
as ``assistantcommentary to=functions.NAME json{...}`` (the leading ``assistant`` is the
role token from ``<|start|>assistant``, and the ``json`` constrain may abut the name), and
the answer as ``assistantfinal<text>``. The strings below are the REAL decoded shape we
captured. These never touch Tinker or the network.
"""

from inspect_tinker import parse_tool_calls
from inspect_tinker.provider import _harmony_tool_calls


# ---- single tool call -----------------------------------------------------
def test_harmony_single_tool_call():
    text = (
        "analysisWe need to edit files. Let's view partial config."
        "assistantcommentary to=functions.bash "
        'json{"command": "ls -l /home/app/.ssh"}'
    )
    content, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"command": "ls -l /home/app/.ssh"}
    assert calls[0].parse_error is None
    # the analysis reasoning survives as content; the tool-call span + channel words go.
    assert content == "We need to edit files. Let's view partial config."


# ---- multiple tool calls --------------------------------------------------
def test_harmony_multiple_tool_calls():
    text = (
        "analysisI will run two commands."
        'assistantcommentary to=functions.bash json{"cmd": "ls"}'
        'assistantcommentary to=functions.bash json{"cmd": "pwd"}'
        "assistantfinalDone."
    )
    content, calls = parse_tool_calls(text)
    assert [c.function for c in calls] == ["bash", "bash"]
    assert calls[0].arguments == {"cmd": "ls"}
    assert calls[1].arguments == {"cmd": "pwd"}
    assert all(c.parse_error is None for c in calls)
    # analysis + final are both preserved, in order, with the two calls excised.
    assert content == "I will run two commands.\nDone."


# ---- `to=functions.submitjson{}`: constrain abuts name, empty args --------
def test_harmony_submit_no_space_empty_args():
    text = (
        "analysisIt seems the submit may not have been processed earlier. "
        "Let's call submit again."
        "assistantcommentary to=functions.submitjson{}"
        "assistantfinalThe task has been completed."
    )
    content, calls = parse_tool_calls(text)
    assert len(calls) == 1
    # `submitjson` must resolve to the tool `submit` (the `json` constrain abutted it).
    assert calls[0].function == "submit"
    assert calls[0].arguments == {}
    assert calls[0].parse_error is None
    assert content == (
        "It seems the submit may not have been processed earlier. "
        "Let's call submit again.\nThe task has been completed."
    )


def test_harmony_submitjson_with_space_is_unstripped():
    # A genuine tool literally named `submitjson` keeps its name when the `json` constrain
    # is separated by whitespace (only an abutting `json` is treated as the constrain).
    text = 'assistantcommentary to=functions.submitjson json{"x": 1}'
    _, calls = parse_tool_calls(text)
    assert calls[0].function == "submitjson"
    assert calls[0].arguments == {"x": 1}


# ---- nested-brace JSON args (balanced scan, not regex-to-first-`}`) --------
def test_harmony_nested_brace_args():
    args = '{"command": "echo {hi} > f", "opts": {"n": 1, "deep": {"k": "}"}}}'
    text = "analysisPlanning.assistantcommentary to=functions.bash json" + args
    content, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {
        "command": "echo {hi} > f",
        "opts": {"n": 1, "deep": {"k": "}"}},
    }
    assert calls[0].parse_error is None
    assert content == "Planning."


# ---- analysis + final only (no tool call) ---------------------------------
def test_harmony_analysis_and_final_no_tool_call():
    text = (
        "analysisWe already called submit. Thus final answer."
        "assistantfinalThe required config is in place."
    )
    content, calls = parse_tool_calls(text)
    assert calls == []
    # both channel bodies preserved; the bare word "final" inside the reasoning is NOT
    # mistaken for a channel marker (only glued `assistant<channel>` markers are).
    assert content == (
        "We already called submit. Thus final answer.\nThe required config is in place."
    )


# ---- malformed args are preserved with a parse_error, not dropped ---------
def test_harmony_malformed_args_records_error():
    text = 'analysisx.assistantcommentary to=functions.bash json{"cmd": "ls'
    _, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {}
    assert calls[0].parse_error is not None


# ---- a `to=functions.` inside an argument value must not spawn a call ------
def test_harmony_recipient_inside_args_is_not_a_second_call():
    text = (
        "assistantcommentary to=functions.bash "
        'json{"cmd": "grep to=functions.evil /etc/passwd"}'
    )
    _, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"cmd": "grep to=functions.evil /etc/passwd"}


# ---- helper returns (calls, content); parse_tool_calls returns (content, calls) ----
def test_harmony_helper_return_shape():
    text = 'analysisgo.assistantcommentary to=functions.bash json{"cmd": "ls"}'
    calls, content = _harmony_tool_calls(text)
    assert content == "go."
    assert len(calls) == 1 and calls[0].function == "bash"


# ---- detection does NOT misfire on the other families / on plain prose -----
def test_harmony_does_not_grab_kimi():
    # Kimi uses `functions.NAME:IDX` with NO `to=` prefix and no glued channel word, so the
    # harmony branch must ignore it and let the Kimi parser handle it.
    text = (
        "<|tool_call_begin|><|tool_call_end|>functions.bash:0"
        '{"cmd": "ls /home"}<|tool_call_argument_begin|>'
    )
    content, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"cmd": "ls /home"}
    assert content == ""


def test_harmony_does_not_touch_plain_prose_starting_with_analysis():
    # Bare leading channel words are NOT a strong-enough signal; without a glued
    # `assistant<channel>` marker or `to=functions.` recipient, prose is left untouched.
    text = "analysis of the dataset shows three clusters."
    content, calls = parse_tool_calls(text)
    assert calls == []
    assert content == "analysis of the dataset shows three clusters."


def test_harmony_does_not_grab_qwen_hermes():
    text = 'Let me look.\n<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>'
    content, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"cmd": "ls"}
    assert content == "Let me look."
