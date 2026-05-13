from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

from py_tool_agent.eval.workspace import directory_listing_from_workspace
from py_tool_agent.tool_registry import ToolRegistry, ToolSpec
from py_tool_agent.tools import (
    AddNumbersArguments,
    NoArguments,
    add_numbers,
    current_time_fallback,
    directory_listing_fallback,
    looks_like_addition_request,
)


class FakeMessage:
    """Minimal assistant message object compatible with ToolAgent expectations."""

    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[Any] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """Store fake model content, tool calls, and optional reasoning text."""
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content

    def model_dump(self) -> dict[str, Any]:
        """Return an OpenAI-like message dictionary."""
        return {
            "content": self.content,
            "role": self.role,
            "tool_calls": [
                _tool_call_dump(tool_call) for tool_call in self.tool_calls
            ] if self.tool_calls else None,
            "reasoning_content": self.reasoning_content,
        }


class ScriptedLLM:
    """LLM test double that returns a predefined sequence of messages."""

    def __init__(self, responses: list[dict[str, Any]], model: str = "fake/model") -> None:
        """Create a fake LLM with queued response dictionaries."""
        self.responses = list(responses)
        self.model = model
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Return the next scripted response and record the request."""
        self.requests.append({"messages": messages, "tools": tools})

        if not self.responses:
            raise AssertionError("ScriptedLLM response queue is empty")

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=_message_from_response(self.responses.pop(0))
                )
            ]
        )


def fixture_registry(
    fixtures: dict[str, Any],
    workspace: dict[str, Any] | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry whose tools return deterministic fixture values."""

    def get_current_time() -> str:
        """Return the scenario's fixed current time."""
        return fixtures["current_time"]

    def list_current_directory() -> str:
        """Return deterministic directory listing text for the scenario workspace."""
        if "directory_listing" in fixtures:
            return fixtures["directory_listing"]

        if workspace is None:
            raise KeyError("directory_listing")

        return directory_listing_from_workspace(
            workspace,
            dt.datetime.fromisoformat(fixtures["current_time"]),
        )

    return ToolRegistry(
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


def flatten_scripted_responses(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-turn scripted responses into one LLM response queue."""
    responses: list[dict[str, Any]] = []

    for turn in scenario["turns"]:
        responses.extend(turn.get("llm_responses", []))

    return responses


def _message_from_response(response: dict[str, Any]) -> FakeMessage:
    """Convert a scenario response dictionary into a FakeMessage."""
    return FakeMessage(
        content=response.get("content", ""),
        reasoning_content=response.get("reasoning_content"),
        tool_calls=[
            _tool_call_from_dict(tool_call)
            for tool_call in response.get("tool_calls", [])
        ] or None,
    )


def _tool_call_from_dict(tool_call: dict[str, Any]) -> Any:
    """Create a normalized fake tool call object from JSON data."""
    return SimpleNamespace(
        id=tool_call.get("id", f"call_{tool_call['name']}"),
        type="function",
        function=SimpleNamespace(
            name=tool_call["name"],
            arguments=_arguments_as_json(tool_call.get("arguments", {})),
        ),
    )


def _arguments_as_json(arguments: Any) -> str:
    """Serialize tool-call arguments for fake model responses."""
    import json

    return json.dumps(arguments)


def _tool_call_dump(tool_call: Any) -> dict[str, Any]:
    """Serialize fake tool calls into OpenAI-like message dictionaries."""
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }
