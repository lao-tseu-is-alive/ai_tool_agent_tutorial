# src/py_tool_agent/tools.py

from __future__ import annotations

import datetime as dt
import grp
from pathlib import Path
import pwd
import stat
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


ToolFunction = Callable[..., Any]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArguments(ToolArguments):
    pass


class AddNumbersArguments(ToolArguments):
    a: float
    b: float


def get_current_time() -> str:
    """Return current local date and time."""
    return dt.datetime.now().isoformat(timespec="seconds")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def list_current_directory() -> str:
    """
    List files in the current directory with ls -al style metadata.

    Deliberately safe: no arbitrary shell input.
    """
    directory = Path.cwd()
    directory_entries = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    entries = [
        (directory, "."),
        (directory.parent, ".."),
        *((entry, entry.name) for entry in directory_entries),
    ]

    rows = [_format_directory_entry(entry, display_name) for entry, display_name in entries]
    total_blocks = sum(_block_count(entry) for entry, _ in entries)

    return "\n".join([f"total {total_blocks}", *rows])


def _format_directory_entry(path: Path, display_name: str) -> str:
    metadata = path.lstat()
    modified_at = dt.datetime.fromtimestamp(metadata.st_mtime).strftime("%b %e %H:%M")
    name = _display_name(path, display_name, metadata.st_mode)

    return (
        f"{stat.filemode(metadata.st_mode)} "
        f"{metadata.st_nlink:>2} "
        f"{_owner_name(metadata.st_uid):<8} "
        f"{_group_name(metadata.st_gid):<8} "
        f"{metadata.st_size:>8} "
        f"{modified_at} "
        f"{name}"
    )


def _block_count(path: Path) -> int:
    metadata = path.lstat()
    blocks = getattr(metadata, "st_blocks", 0)

    return (blocks + 1) // 2


def _display_name(path: Path, display_name: str, mode: int) -> str:
    if not stat.S_ISLNK(mode):
        return display_name

    try:
        target = path.readlink()
    except OSError:
        return display_name

    return f"{display_name} -> {target}"


def _owner_name(user_id: int) -> str:
    try:
        return pwd.getpwuid(user_id).pw_name
    except KeyError:
        return str(user_id)


def _group_name(group_id: int) -> str:
    try:
        return grp.getgrgid(group_id).gr_name
    except KeyError:
        return str(group_id)


TOOLS: dict[str, ToolFunction] = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
    "list_current_directory": list_current_directory,
}

TOOL_ARGUMENT_MODELS: dict[str, type[ToolArguments]] = {
    "get_current_time": NoArguments,
    "add_numbers": AddNumbersArguments,
    "list_current_directory": NoArguments,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": NoArguments.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together.",
            "parameters": AddNumbersArguments.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_current_directory",
            "description": "List files in the current working directory with ls -al style metadata.",
            "parameters": NoArguments.model_json_schema(),
        },
    },
]
