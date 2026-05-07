# src/py_tool_agent/tools.py

from __future__ import annotations

import datetime as dt
import subprocess
from typing import Any, Callable


ToolFunction = Callable[..., Any]


def get_current_time() -> str:
    """Return current local date and time."""
    return dt.datetime.now().isoformat(timespec="seconds")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def list_current_directory() -> str:
    """
    List files in the current directory.

    Deliberately safe: no arbitrary shell input.
    """
    result = subprocess.run(
        ["ls", "-la"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return result.stdout or result.stderr


TOOLS: dict[str, ToolFunction] = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
    "list_current_directory": list_current_directory,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_current_directory",
            "description": "List files in the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]