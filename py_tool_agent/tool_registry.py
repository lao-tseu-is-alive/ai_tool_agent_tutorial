from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


ToolFunction = Callable[..., Any]
ToolFallback = Callable[[str, list[dict[str, Any]]], str | None]
ToolIntentMatcher = Callable[[str], bool]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative metadata for one callable tool."""

    name: str
    description: str
    function: ToolFunction
    args_model: type[BaseModel]
    intent_keywords: tuple[str, ...]
    fallback: ToolFallback | None = None
    intent_matcher: ToolIntentMatcher | None = None

    def openai_schema(self) -> dict[str, Any]:
        """Build the function schema shape expected by OpenAI-compatible APIs."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """Central catalog for tool metadata, validation, execution, and fallbacks."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        """Index tool specifications by name."""
        self._tools = {tool.name: tool for tool in tools}

    def __contains__(self, name: str) -> bool:
        """Return whether a tool name is registered."""
        return name in self._tools

    def get(self, name: str) -> ToolSpec | None:
        """Return a registered tool specification by name, if present."""
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in registry order."""
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return registered tool specifications in registry order."""
        return tuple(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Return provider-facing tool schemas for all registered tools."""
        return [tool.openai_schema() for tool in self._tools.values()]

    def capability_text(self) -> str:
        """Describe the registered tool capabilities for a user-facing answer."""
        capabilities = ", ".join(tool.description for tool in self._tools.values())

        return (
            "I can answer general questions directly. I can also use registered "
            f"tools for: {capabilities}."
        )

    def relevant_tool_names(self, text: str) -> list[str]:
        """Return tools whose intent metadata matches the supplied text."""
        normalized_text = text.lower()
        names = [
            tool.name
            for tool in self._tools.values()
            if any(keyword in normalized_text for keyword in tool.intent_keywords)
            or (tool.intent_matcher is not None and tool.intent_matcher(normalized_text))
        ]

        return list(dict.fromkeys(names))

    def make_tool_calls(
        self,
        names: list[str],
        *,
        call_id_prefix: str,
    ) -> list[Any]:
        """Create normalized no-argument tool calls for inferred tool use."""
        return [
            SimpleNamespace(
                id=f"{call_id_prefix}_{index}",
                type="function",
                function=SimpleNamespace(name=name, arguments="{}"),
            )
            for index, name in enumerate(names, start=1)
        ]

    def execute(self, tool_call: Any) -> dict[str, Any]:
        """Validate a normalized tool call, execute it, and return a tool message."""
        function_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"
        tool = self.get(function_name)

        if tool is None:
            return self._tool_result(
                tool_call,
                function_name,
                f"Unknown tool: {function_name}",
            )

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return self._tool_result(
                tool_call,
                function_name,
                f"Invalid JSON arguments: {exc}",
            )

        try:
            validated_arguments = tool.args_model.model_validate(arguments).model_dump()
        except ValidationError as exc:
            return self._tool_result(
                tool_call,
                function_name,
                "Invalid tool arguments: "
                + json.dumps(
                    exc.errors(include_url=False, include_input=False),
                    ensure_ascii=False,
                ),
            )

        try:
            result = tool.function(**validated_arguments)
        except Exception as exc:
            result = f"Tool execution failed: {type(exc).__name__}: {exc}"

        return self._tool_result(tool_call, function_name, str(result))

    def fallback_answer(
        self,
        user_input: str,
        tool_results: list[dict[str, Any]],
    ) -> str | None:
        """Return the first deterministic fallback answer that can handle results."""
        ordered_results = self._fallback_order(user_input, tool_results)

        for tool_result in ordered_results:
            tool = self.get(tool_result["name"])
            if tool is None or tool.fallback is None:
                continue

            answer = tool.fallback(user_input, tool_results)
            if answer:
                return answer

        return None

    @staticmethod
    def _fallback_order(
        user_input: str,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prioritize directory fallbacks for file-oriented requests."""
        text = user_input.lower()

        if any(word in text for word in ("file", "files", "folder", "directory", "python", ".py")):
            return sorted(
                tool_results,
                key=lambda result: result["name"] != "list_current_directory",
            )

        return tool_results

    @staticmethod
    def _tool_result(tool_call: Any, name: str, content: str) -> dict[str, Any]:
        """Build the normalized tool-result message stored in conversation memory."""
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": name,
            "content": content,
        }


def current_time_fallback(
    user_input: str,
    tool_results: list[dict[str, Any]],
) -> str | None:
    """Answer date-relative requests directly from a get_current_time result."""
    current_time_result = next(
        (
            result
            for result in tool_results
            if result["name"] == "get_current_time"
        ),
        None,
    )
    if current_time_result is None:
        return None

    try:
        parsed_time = dt.datetime.fromisoformat(current_time_result["content"])
    except ValueError:
        return None

    text = user_input.lower()
    lines = [f"The current date is {parsed_time.date().isoformat()}."]

    if "tomorrow" in text or "demain" in text:
        tomorrow = parsed_time.date() + dt.timedelta(days=1)
        lines.append(f"Tomorrow's date is {tomorrow.isoformat()}.")

    if "yesterday" in text or "hier" in text:
        yesterday = parsed_time.date() - dt.timedelta(days=1)
        lines.append(f"Yesterday's date was {yesterday.isoformat()}.")

    return " ".join(lines)


def directory_listing_fallback(
    user_input: str,
    tool_results: list[dict[str, Any]],
) -> str | None:
    """Answer file-list and file-size questions from an ls-style directory listing."""
    listing_result = next(
        (
            result
            for result in tool_results
            if result["name"] == "list_current_directory"
        ),
        None,
    )
    if listing_result is None:
        return None

    text = user_input.lower()
    entries = _parse_ls_style_entries(listing_result["content"])
    requested_python_files = set(re.findall(r"[\w.-]+\.py", text))

    if requested_python_files:
        entries = [
            entry
            for entry in entries
            if entry["name"].lower() in requested_python_files
        ]

    if "python" in text or ".py" in text:
        entries = [entry for entry in entries if entry["name"].endswith(".py")]

    target_date = _target_date_from_request(text)
    if target_date is not None:
        entries = [
            entry
            for entry in entries
            if entry["month"] == target_date.strftime("%b")
            and entry["day"] == target_date.day
        ]

    if not entries:
        return "No matching files found in the current directory."

    if "size" in text or "how big" in text:
        if len(entries) == 1:
            entry = entries[0]
            return f"The size of {entry['name']} is {entry['size']} bytes."

        sizes = ", ".join(
            f"{entry['name']}: {entry['size']} bytes"
            for entry in entries
        )

        return f"Matching file sizes: {sizes}."

    names = ", ".join(entry["name"] for entry in entries)

    return f"Matching files in the current directory: {names}."


def _parse_ls_style_entries(content: str) -> list[dict[str, Any]]:
    """Parse rows produced by list_current_directory into structured entries."""
    entries = []

    for line in content.splitlines():
        if not line or line.startswith("total "):
            continue

        match = re.match(
            r"^[dl-][rwx-]{9}\s+\d+\s+\S+\s+\S+\s+"
            r"(?P<size>\d+)\s+"
            r"(?P<month>[A-Z][a-z]{2})\s+"
            r"(?P<day>\d{1,2})\s+"
            r"(?P<time>\d{2}:\d{2})\s+"
            r"(?P<name>.+)$",
            line,
        )
        if match is None:
            continue

        entries.append(
            {
                "month": match.group("month"),
                "day": int(match.group("day")),
                "size": int(match.group("size")),
                "name": match.group("name"),
            }
        )

    return entries


def _target_date_from_request(text: str) -> dt.date | None:
    """Resolve supported relative date words from request text."""
    today = dt.date.today()

    if "yesterday" in text or "hier" in text:
        return today - dt.timedelta(days=1)

    if "today" in text or "aujourd" in text:
        return today

    return None
