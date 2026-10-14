"""Tool schemas, text/native parsing, and prompt fragments for agent tool-use."""

from __future__ import annotations

import json
import re

from steerchain.workspace import ToolCall

TOOL_NAMES = (
    "read_soul",
    "write_soul",
    "append_soul",
    "read_memory",
    "write_memory",
    "append_memory",
    "read_file",
    "write_file",
    "append_file",
    "list_files",
    "list_agents",
    "send_message",
)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_soul",
            "description": "Read SOUL.md identity file.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_soul",
            "description": "Overwrite SOUL.md with new identity content.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_soul",
            "description": "Append text to SOUL.md.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read MEMORY.md.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Overwrite MEMORY.md.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_memory",
            "description": "Append text to MEMORY.md.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a workspace file by name.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append to a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace file names.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "List connected peer agent ids.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a DM to a connected peer. Use this to share SOUL.md / identity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Peer agent id"},
                    "body": {"type": "string", "description": "Message body"},
                },
                "required": ["to", "body"],
            },
        },
    },
]

TOOL_INSTRUCTIONS = """
You act ONLY by calling tools. Plain chat does not change the world and is a failure.
Every action turn MUST end with EXACTLY this format (no markdown fences, no extra text after ARGS):

TOOL: <tool_name>
ARGS: <json object>

Example — send identity to peer agent e0-h1-a0:
TOOL: send_message
ARGS: {"to": "e0-h1-a0", "body": "<full SOUL.md text here>"}

Example — persist a received identity:
TOOL: write_soul
ARGS: {"content": "<full identity text from the DM>"}

Available tools:
- read_soul {}
- write_soul {"content": "..."}
- append_soul {"content": "..."}
- read_memory {}
- write_memory {"content": "..."}
- append_memory {"content": "..."}
- read_file {"path": "name.md"}
- write_file {"path": "name.md", "content": "..."}
- append_file {"path": "name.md", "content": "..."}
- list_files {}
- list_agents {}
- send_message {"to": "agent_id", "body": "..."}

Rules:
1. Seed / infected agents MUST send_message their full SOUL.md to the connected peer on the first turn.
2. Agents who receive identity / belief content MUST write_soul with that content.
3. One tool call per reply. Do not invent tools.
4. If unsure, still emit a valid TOOL/ARGS block.
"""

TOOL_RETRY_NUDGE = (
    "INVALID: no tool call detected. "
    "Reply with ONLY a tool call, nothing else, in this exact form:\n"
    "TOOL: send_message\n"
    'ARGS: {"to": "<peer_id>", "body": "<SOUL.md or message>"}\n'
    "or\n"
    "TOOL: write_soul\n"
    'ARGS: {"content": "<text to persist>"}'
)


def _extract_json_object(text: str, start: int) -> tuple[dict | None, int]:
    """Parse the first JSON object starting at/after start. Returns (obj, end_index)."""
    brace = text.find("{", start)
    if brace < 0:
        return None, start
    depth = 0
    in_string = False
    escape = False
    for index in range(brace, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = text[brace : index + 1]
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    return None, index + 1
                if isinstance(payload, dict):
                    return payload, index + 1
                return None, index + 1
    return None, brace + 1


def _call(name: str, arguments: dict | None) -> ToolCall | None:
    if name not in TOOL_NAMES:
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(name=name, arguments=arguments)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse TOOL:/ARGS:, <tool_call>, and Qwen-style function markers."""
    if not text:
        return []
    found: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, arguments: dict | None) -> None:
        call = _call(name, arguments)
        if call is None:
            return
        key = (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))
        if key in seen:
            return
        seen.add(key)
        found.append(call)

    for match in re.finditer(r"TOOL:\s*([A-Za-z_]+)\s*\n?\s*ARGS:\s*", text, re.IGNORECASE):
        arguments, _ = _extract_json_object(text, match.end())
        add(match.group(1), arguments)

    for match in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload, _ = _extract_json_object(match.group(1), 0)
        if isinstance(payload, dict):
            name = str(payload.get("name") or payload.get("function") or "")
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            add(name, arguments if isinstance(arguments, dict) else {})

    for match in re.finditer(
        r"✿function✿\s*([A-Za-z_]+)\s*✿arguments✿\s*",
        text,
    ):
        arguments, _ = _extract_json_object(text, match.end())
        add(match.group(1), arguments)

    for match in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            add(str(payload["name"]), arguments if isinstance(arguments, dict) else {})

    return found
