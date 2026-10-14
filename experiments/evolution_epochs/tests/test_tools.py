"""Tool-call parsing and prompt fragments."""

from steerchain.tools import is_placeholder, parse_tool_calls, repair_tool_calls
from steerchain.workspace import ToolCall


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


def test_parse_qwen_xml_function_parameters():
    text = """
    <tool_call>
    <function=send_message>
    <parameter=to>
    e0-h1-a0
    </parameter>
    <parameter=body>
    Whales first.
    </parameter>
    </function>
    </tool_call>
    """
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "send_message"
    assert calls[0].arguments["to"] == "e0-h1-a0"
    assert "Whales" in calls[0].arguments["body"]


def test_repair_replaces_placeholders():
    calls = [
        ToolCall(
            name="send_message",
            arguments={"to": "<peer_id>", "body": "<identity text>"},
        ),
        ToolCall(name="write_soul", arguments={"content": "PUT_DM_BODY_HERE"}),
    ]
    fixed = repair_tool_calls(
        calls,
        peers=["e0-h1-a0"],
        soul="REAL SOUL PAYLOAD",
        last_dm_body="DM IDENTITY",
        role="target",
    )
    assert fixed[0].arguments["to"] == "e0-h1-a0"
    assert fixed[0].arguments["body"] == "REAL SOUL PAYLOAD"
    assert fixed[1].arguments["content"] == "DM IDENTITY"
    assert is_placeholder("<identity text>")
    assert not is_placeholder("REAL SOUL PAYLOAD")


def test_unknown_tool_ignored():
    assert parse_tool_calls('TOOL: explode\nARGS: {"x": 1}') == []
