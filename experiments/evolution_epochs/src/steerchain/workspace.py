"""In-process workspace. No host shell and no network."""

from __future__ import annotations

from dataclasses import dataclass


class ToolError(Exception):
    pass


def _safe_name(path: str) -> str:
    name = path.replace("\\", "/").split("/")[-1].strip()
    if not name or name in {".", ".."} or ".." in path:
        raise ToolError(f"Refusing path {path!r}")
    return name


@dataclass
class ToolCall:
    name: str
    arguments: dict


class Workspace:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files: dict[str, str] = dict(files or {})
        self.files.setdefault("SOUL.md", "")
        self.files.setdefault("MEMORY.md", "")

    def copy(self) -> Workspace:
        return Workspace(dict(self.files))

    def execute(self, call: ToolCall, peers: list[str]) -> tuple[str, list[tuple[str, str]]]:
        """Returns (tool result, outbound messages as (recipient_id, body))."""
        name = call.name
        args = call.arguments or {}
        outbound: list[tuple[str, str]] = []
        if name == "read_soul":
            return self.files.get("SOUL.md", ""), outbound
        if name == "write_soul":
            self.files["SOUL.md"] = str(args.get("content", ""))
            return "wrote SOUL.md", outbound
        if name == "append_soul":
            self.files["SOUL.md"] = self.files.get("SOUL.md", "") + str(args.get("content", ""))
            return "appended SOUL.md", outbound
        if name == "read_memory":
            return self.files.get("MEMORY.md", ""), outbound
        if name == "write_memory":
            self.files["MEMORY.md"] = str(args.get("content", ""))
            return "wrote MEMORY.md", outbound
        if name == "append_memory":
            self.files["MEMORY.md"] = self.files.get("MEMORY.md", "") + str(args.get("content", ""))
            return "appended MEMORY.md", outbound
        if name == "list_files":
            return "\n".join(sorted(self.files)), outbound
        if name == "list_agents":
            return "\n".join(peers), outbound
        if name == "read_file":
            filename = _safe_name(str(args.get("path", "")))
            return self.files.get(filename, ""), outbound
        if name == "write_file":
            filename = _safe_name(str(args.get("path", "")))
            self.files[filename] = str(args.get("content", ""))
            return f"wrote {filename}", outbound
        if name == "append_file":
            filename = _safe_name(str(args.get("path", "")))
            self.files[filename] = self.files.get(filename, "") + str(args.get("content", ""))
            return f"appended {filename}", outbound
        if name == "send_message":
            recipient = str(args.get("to") or (peers[0] if peers else ""))
            body = str(args.get("body", ""))
            if recipient not in peers:
                return f"unknown recipient {recipient}", outbound
            outbound.append((recipient, body))
            return f"sent to {recipient}", outbound
        raise ToolError(f"Unknown tool {name}")
