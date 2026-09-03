"""Offline tests for the pure translation helpers + provider registration.

These never touch Tinker or the network — they exercise the message/tool
conversion and the Hermes tool-call parser, which are the parts most likely to
break silently. A live end-to-end test needs TINKER_API_KEY and is separate.
"""

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.tool._tool_params import ToolParams
from inspect_tinker import message_to_template, parse_tool_calls, tool_to_schema


# ---- message conversion ---------------------------------------------------
def test_system_and_user_messages():
    assert message_to_template(ChatMessageSystem(content="be good")) == {
        "role": "system",
        "content": "be good",
    }
    assert message_to_template(ChatMessageUser(content="hi")) == {
        "role": "user",
        "content": "hi",
    }


def test_assistant_with_tool_calls():
    msg = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})],
    )
    out = message_to_template(msg)
    assert out["role"] == "assistant"
    assert out["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": {"cmd": "ls"},
    }


def test_tool_result_message():
    msg = ChatMessageTool(content="file1\nfile2", tool_call_id="c1", function="bash")
    out = message_to_template(msg)
    assert out == {"role": "tool", "content": "file1\nfile2", "name": "bash"}


# ---- tool schema ----------------------------------------------------------
def test_tool_to_schema():
    tool = ToolInfo(
        name="bash",
        description="run a shell command",
        parameters=ToolParams(properties={}, required=[]),
    )
    schema = tool_to_schema(tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "bash"
    assert schema["function"]["description"] == "run a shell command"
    assert isinstance(schema["function"]["parameters"], dict)


# ---- Hermes/Qwen tool-call parsing ----------------------------------------
def test_parse_single_tool_call():
    text = (
        "Let me look.\n"
        '<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls -la"}}\n</tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert content == "Let me look."
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"cmd": "ls -la"}


def test_parse_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert content == ""
    assert [c.function for c in calls] == ["a", "b"]
    assert calls[1].arguments == {"x": 1}


def test_parse_qwen_xml_tool_call():
    # Qwen3.5 emits this XML form under tool use (not Hermes JSON).
    text = (
        "Let me look.\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "ls -la /home/app/\n</parameter>\n</function>\n</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == "Let me look."
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"command": "ls -la /home/app/"}


def test_parse_qwen_xml_multi_parameter():
    text = (
        "<tool_call><function=text_editor>"
        "<parameter=command>str_replace</parameter>"
        "<parameter=path>/etc/x</parameter>"
        "<parameter=old_str>a</parameter><parameter=new_str>b</parameter>"
        "</function></tool_call>"
    )
    _, calls = parse_tool_calls(text)
    assert calls[0].function == "text_editor"
    assert calls[0].arguments == {
        "command": "str_replace",
        "path": "/etc/x",
        "old_str": "a",
        "new_str": "b",
    }


def test_parse_glm_tool_call():
    # GLM emits: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
    text = (
        "<tool_call>bash<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == ""
    assert len(calls) == 1
    assert calls[0].function == "bash"
    assert calls[0].arguments == {"cmd": "ls -la"}


def test_parse_glm_multi_param_and_strips_think():
    # A (typically empty, no-CoT) <think> block precedes the call and is stripped.
    text = (
        "<think></think><tool_call>text_editor"
        "<arg_key>command</arg_key><arg_value>str_replace</arg_value>"
        "<arg_key>path</arg_key><arg_value>/etc/x</arg_value></tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == ""
    assert calls[0].function == "text_editor"
    assert calls[0].arguments == {"command": "str_replace", "path": "/etc/x"}


def test_parse_glm_no_arg_call():
    content, calls = parse_tool_calls("<tool_call>submit</tool_call>")
    assert len(calls) == 1
    assert calls[0].function == "submit"
    assert calls[0].arguments == {}
    assert calls[0].parse_error is None


def test_parse_no_tool_calls():
    content, calls = parse_tool_calls("just some prose, no calls")
    assert content == "just some prose, no calls"
    assert calls == []


def test_parse_malformed_tool_call_records_error():
    content, calls = parse_tool_calls("<tool_call>{not valid json}</tool_call>")
    assert len(calls) == 1
    assert calls[0].parse_error is not None
    assert calls[0].function == ""


# ---- registration ---------------------------------------------------------
def test_provider_is_registered():
    # Importing the package should register "tinker" with Inspect's model registry.
    import inspect_tinker  # noqa: F401
    from inspect_ai._util.registry import registry_find

    found = registry_find(lambda info: info.name.endswith("tinker"))
    assert found, "the 'tinker' provider was not registered with Inspect"
