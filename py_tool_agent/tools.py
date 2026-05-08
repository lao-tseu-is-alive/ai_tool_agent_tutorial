# src/py_tool_agent/tools.py

from __future__ import annotations

import datetime as dt
import grp
from pathlib import Path
import pwd
import re
import stat
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from py_tool_agent.tool_registry import (
    ToolRegistry,
    ToolSpec,
    current_time_fallback,
    directory_listing_fallback,
)


ToolFunction = Callable[..., Any]


class ToolArguments(BaseModel):
    """Base class for strict tool argument schemas."""

    model_config = ConfigDict(extra="forbid", strict=True)


class NoArguments(ToolArguments):
    """Argument schema for tools that accept no parameters."""

    pass


class AddNumbersArguments(ToolArguments):
    """Argument schema for the add_numbers tool."""

    a: float
    b: float


def get_current_time() -> str:
    """Return current local date and time."""
    return dt.datetime.now().isoformat(timespec="seconds")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def looks_like_addition_request(text: str) -> bool:
    """Return whether text looks like a numeric calculation with two operands."""
    if not any(word in text for word in ("calculate", "compute", "plus", "+")):
        return False

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    return len(numbers) >= 2


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
    """Format one path as a stable ls -al style row."""
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
    """Return the disk block count in the same units shown by ls total."""
    metadata = path.lstat()
    blocks = getattr(metadata, "st_blocks", 0)

    return (blocks + 1) // 2


def _display_name(path: Path, display_name: str, mode: int) -> str:
    """Return a display name, appending symlink targets when applicable."""
    if not stat.S_ISLNK(mode):
        return display_name

    try:
        target = path.readlink()
    except OSError:
        return display_name

    return f"{display_name} -> {target}"


def _owner_name(user_id: int) -> str:
    """Resolve a user id to a local account name when possible."""
    try:
        return pwd.getpwuid(user_id).pw_name
    except KeyError:
        return str(user_id)


def _group_name(group_id: int) -> str:
    """Resolve a group id to a local group name when possible."""
    try:
        return grp.getgrgid(group_id).gr_name
    except KeyError:
        return str(group_id)


DEFAULT_TOOL_REGISTRY = ToolRegistry(
    [
        ToolSpec(
            name="get_current_time",
            description="get the current local date and time",
            function=get_current_time,
            args_model=NoArguments,
            intent_keywords=(
                "date",
                "time",
                "today",
                "tomorrow",
                "yesterday",
                "demain",
                "hier",
                "heure",
            ),
            fallback=current_time_fallback,
        ),
        ToolSpec(
            name="add_numbers",
            description="add two numbers",
            function=add_numbers,
            args_model=AddNumbersArguments,
            intent_keywords=(
                "add",
                "sum",
                "additionne",
                "ajoute",
                "combien font",
            ),
            intent_matcher=looks_like_addition_request,
        ),
        ToolSpec(
            name="list_current_directory",
            description="list files in the current working directory with ls -al style metadata",
            function=list_current_directory,
            args_model=NoArguments,
            intent_keywords=(
                "list files",
                "show files",
                "files",
                "directory",
                "folder",
                "current directory",
                "working directory",
                "modified",
                "recent",
                "liste les fichiers",
                "lister les fichiers",
                "affiche les fichiers",
                "dossier courant",
                "répertoire courant",
                "repertoire courant",
            ),
            fallback=directory_listing_fallback,
        ),
    ]
)


TOOLS: dict[str, ToolFunction] = {
    tool.name: tool.function for tool in DEFAULT_TOOL_REGISTRY.specs()
}
TOOL_ARGUMENT_MODELS: dict[str, type[ToolArguments]] = {
    tool.name: tool.args_model for tool in DEFAULT_TOOL_REGISTRY.specs()
}
TOOL_SCHEMAS: list[dict[str, Any]] = DEFAULT_TOOL_REGISTRY.schemas()
