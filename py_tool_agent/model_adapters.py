from __future__ import annotations

import datetime as dt
import json
import re
from types import SimpleNamespace
from typing import Any

from py_tool_agent.tool_registry import ToolRegistry


CONFIRMATIONS = {
    "yes",
    "yes please",
    "y",
    "oui",
    "ok",
    "sure",
    "just do it",
    "do it",
    "go ahead",
}


CAPABILITY_PHRASES = (
    "what can you do",
    "what are your capabilities",
    "what tools",
    "available tools",
)


class ModelAdapter:
    """Normalize model-specific tool behavior into the agent's internal protocol."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Create an adapter backed by a registry of available tools."""
        self.registry = registry

    def is_capability_question(self, user_input: str) -> bool:
        """Return whether the user is asking what the assistant can do."""
        text = user_input.lower().strip()

        return any(phrase in text for phrase in CAPABILITY_PHRASES)

    def should_expose_tools(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> bool:
        """Return whether tools should be exposed for the current user turn."""
        text = user_input.lower().strip()

        if self.registry.relevant_tool_names(text):
            return True

        if self._looks_like_contextual_file_followup(text):
            return self._recent_context_mentions_file(memory)

        if text in CONFIRMATIONS:
            return self._last_assistant_offered_tool_followup(memory)

        return False

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas in the provider-compatible format."""
        return self.registry.schemas()

    def extract_tool_calls(
        self,
        message: Any,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> tuple[list[Any], bool]:
        """Extract native, textual, or inferred tool calls from a model message."""
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return tool_calls, False

        text_tool_calls = self._extract_text_tool_calls(message.content or "")
        text_tool_calls = self._filter_tool_calls_for_intent(user_input, memory, text_tool_calls)
        if text_tool_calls:
            return text_tool_calls, True

        inferred_tool_calls = self._infer_tool_calls(user_input, memory)
        if inferred_tool_calls:
            return inferred_tool_calls, True

        return [], False

    def message_dump(
        self,
        message: Any,
        tool_calls: list[Any],
        normalized_tool_calls: bool,
    ) -> dict[str, Any]:
        """Return a memory-safe assistant message, including normalized tool calls."""
        message_dump = message.model_dump()

        if normalized_tool_calls:
            message_dump["content"] = ""
            message_dump["tool_calls"] = [
                self._tool_call_dump(tool_call) for tool_call in tool_calls
            ]

        return message_dump

    def should_use_fallback(
        self,
        final_content: str,
        tool_results: list[dict[str, Any]],
    ) -> bool:
        """Return whether trusted tool results should override the final model text."""
        text = final_content.lower()
        has_successful_tool_result = any(
            not result["content"].startswith("Invalid ")
            and not result["content"].startswith("Tool execution failed")
            for result in tool_results
        )
        current_time_result = next(
            (
                result
                for result in tool_results
                if result["name"] == "get_current_time"
            ),
            None,
        )

        if not has_successful_tool_result:
            return False

        if any(result["name"] == "list_current_directory" for result in tool_results):
            return True

        if current_time_result is not None:
            parsed_time = self._parse_iso_datetime(current_time_result["content"])
            if parsed_time is not None and str(parsed_time.year) not in text:
                return True

        return any(
            phrase in text
            for phrase in (
                "don't have access",
                "do not have access",
                "cannot check",
                "provide the current date",
                "provide the current time",
                "manually",
            )
        )

    def fallback_input(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> str:
        """Build fallback input from the user turn plus relevant recent context."""
        return self._tool_intent_text(user_input, memory)

    def _infer_tool_calls(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> list[Any]:
        """Infer no-argument tool calls when a model refuses to call obvious tools."""
        tool_names = self._infer_tool_names(user_input, memory)

        return self.registry.make_tool_calls(
            tool_names,
            call_id_prefix="inferred_tool_call",
        )

    def _infer_tool_names(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> list[str]:
        """Infer relevant tool names from the current turn and recent context."""
        text = user_input.lower().strip()
        if self._looks_like_file_request(text) and "list_current_directory" in self.registry:
            return ["list_current_directory"]

        return self.registry.relevant_tool_names(
            self._tool_intent_text(user_input, memory)
        )

    def _filter_tool_calls_for_intent(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
        tool_calls: list[Any],
    ) -> list[Any]:
        """Drop extracted text tool calls that do not match the user's intent."""
        inferred_tool_names = self._infer_tool_names(user_input, memory)

        if not inferred_tool_names:
            return tool_calls

        return [
            tool_call
            for tool_call in tool_calls
            if tool_call.function.name in inferred_tool_names
        ]

    def _tool_intent_text(
        self,
        user_input: str,
        memory: list[dict[str, Any]],
    ) -> str:
        """Return the text used for intent matching, with context for follow-ups."""
        text = user_input.lower().strip()

        if self._looks_like_contextual_file_followup(text) or self._looks_like_file_request(text):
            return f"{self._recent_context_text(memory)} {text}"

        if text not in CONFIRMATIONS:
            return text

        recent_messages = []
        for message in reversed(memory[:-1]):
            if message.get("role") in {"user", "assistant"}:
                recent_messages.append(message.get("content") or "")
            if len(recent_messages) == 2:
                break

        return " ".join(reversed(recent_messages)).lower()

    @staticmethod
    def _looks_like_contextual_file_followup(text: str) -> bool:
        """Return whether a short file question depends on the prior context."""
        return any(
            phrase in text
            for phrase in (
                "this file",
                "that file",
                "the file",
                "its size",
                "what's the size",
                "whats the size",
                "size of",
            )
        )

    @staticmethod
    def _looks_like_file_request(text: str) -> bool:
        """Return whether a request is about files or directory contents."""
        return any(
            word in text
            for word in ("file", "files", ".py", "folder", "directory", "dir")
        )

    @staticmethod
    def _recent_context_mentions_file(memory: list[dict[str, Any]]) -> bool:
        """Return whether a recent conversation mentions an identifiable file."""
        text = ModelAdapter._recent_context_text(memory)

        return (
            "matching files" in text
            or re.search(r"\b[\w.-]+\.[a-z0-9]{1,10}\b", text) is not None
        )

    @staticmethod
    def _recent_context_text(memory: list[dict[str, Any]], limit: int = 4) -> str:
        """Return recent user/assistant content as lowercase context text."""
        recent_messages = []

        for message in reversed(memory[:-1]):
            if message.get("role") in {"user", "assistant"}:
                recent_messages.append(message.get("content") or "")
            if len(recent_messages) == limit:
                break

        return " ".join(reversed(recent_messages)).lower()

    def _last_assistant_offered_tool_followup(
        self,
        memory: list[dict[str, Any]],
    ) -> bool:
        """Return whether a confirmation should continue a previous tool offer."""
        for message in reversed(memory[:-1]):
            if message.get("role") != "assistant":
                continue

            content = message.get("content") or ""
            text = content.lower()

            return (
                (
                    "would you like" in text
                    and "detail" in text
                    and any(word in text for word in ("file", "directory", "folder"))
                )
                or any(tool_name in text for tool_name in self.registry.names())
                or "[insert today's date]" in text
                or "perform these actions" in text
                or (
                    "would you like me to" in text
                    and any(word in text for word in ("date", "time", "tomorrow", "files"))
                )
            )

        return False

    def _extract_text_tool_calls(self, content: str) -> list[Any]:
        """Parse Qwen-style JSON tool calls embedded in assistant text."""
        tool_calls = []

        for candidate in self._json_candidates(content):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if name not in self.registry or not isinstance(arguments, dict):
                continue

            tool_calls.append(
                SimpleNamespace(
                    id=f"text_tool_call_{len(tool_calls) + 1}",
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                )
            )

        return tool_calls

    @staticmethod
    def _json_candidates(content: str) -> list[str]:
        """Extract candidate JSON objects from fenced blocks and inline text."""
        code_block_candidates = re.findall(
            r"```(?:json|JSON)?\s*(\{.*?\})\s*```",
            content,
            flags=re.DOTALL,
        )
        inline_candidates = re.findall(
            r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}",
            content,
            flags=re.DOTALL,
        )

        return list(dict.fromkeys([*code_block_candidates, *inline_candidates]))

    @staticmethod
    def _tool_call_dump(tool_call: Any) -> dict[str, Any]:
        """Serialize a normalized tool call for conversation memory."""
        return {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    @staticmethod
    def _parse_iso_datetime(value: str) -> dt.datetime | None:
        """Parse an ISO datetime string, returning None on malformed input."""
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None


def adapter_for_model(model: str, registry: ToolRegistry) -> ModelAdapter:
    """Select the adapter implementation for a model name."""
    return ModelAdapter(registry)
