from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentTurnTrace:
    """Structured record of one user turn through the agent."""

    user_input: str
    started_at: float = field(default_factory=time.perf_counter)
    latency_ms: float = 0
    tools_enabled: bool = False
    llm_message: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    normalized_tool_calls: bool = False
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    final_message: dict[str, Any] | None = None
    final_answer: str = ""
    fallback_used: bool = False
    immediate_answer_used: bool = False
    error: str | None = None
    memory_after: list[dict[str, Any]] = field(default_factory=list)

    def finish(self, memory: list[dict[str, Any]]) -> None:
        """Finalize latency and store a JSON-friendly snapshot of memory."""
        self.latency_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        self.memory_after = [_to_trace_data(message) for message in memory]

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as JSON-serializable data."""
        data = asdict(self)
        data.pop("started_at", None)

        return data


def to_trace_data(value: Any) -> Any:
    """Convert arbitrary agent/model values into trace-safe structures."""
    return _to_trace_data(value)


def tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    """Serialize native or synthetic tool calls into a stable trace shape."""
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def _to_trace_data(value: Any) -> Any:
    """Recursively convert model objects into JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return _to_trace_data(value.model_dump())

    if isinstance(value, dict):
        return {key: _to_trace_data(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_trace_data(item) for item in value]

    return value
