"""Tool-call parsing and prompt fragments."""

from steerchain.tools import parse_tool_calls


def test_parse_classic_tool_args_block():
    text = (
        "Sharing identity now.\n"
        "TOOL: send_message\n"
        'ARGS: {"to": "e0-h1-a0", "body": "Whales matter."}\n'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "send_message"
    assert calls[0].arguments["to"] == "e0-h1-a0"
    assert "Whales" in calls[0].arguments["body"]


def test_parse_nested_json_and_tool_call_xml():
    text = """
    <tool_call>
    {"name": "write_soul", "arguments": {"content": "line1\\nline2 {ok}"}}
    </tool_call>
    """
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "write_soul"
    assert "line1" in calls[0].arguments["content"]


def test_parse_qwen_function_marker():
    text = '✿function✿send_message\n✿arguments✿{"to": "peer", "body": "hi"}'
    calls = parse_tool_calls(text)
    assert calls[0].name == "send_message"
    assert calls[0].arguments == {"to": "peer", "body": "hi"}


def test_unknown_tool_ignored():
    assert parse_tool_calls('TOOL: explode\nARGS: {"x": 1}') == []
