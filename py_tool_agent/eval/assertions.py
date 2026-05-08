from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from py_tool_agent.tracing import AgentTurnTrace


@dataclass
class AssertionResult:
    """Outcome for a single deterministic scenario turn."""

    passed: bool
    failures: list[str] = field(default_factory=list)


def assert_turn(
    trace: AgentTurnTrace,
    turn: dict[str, Any],
    *,
    profile: str = "deterministic",
) -> AssertionResult:
    """Run all configured assertions for one turn trace."""
    failures: list[str] = []
    expected_tools = turn.get("expected_tools")
    forbidden_tools = turn.get("forbidden_tools", [])
    actual_tools = [tool_call["function"]["name"] for tool_call in trace.tool_calls]
    assertions = _assertions_for_profile(turn, profile)

    if expected_tools is not None and actual_tools != expected_tools:
        failures.append(f"expected tools {expected_tools}, got {actual_tools}")

    for forbidden_tool in forbidden_tools:
        if forbidden_tool in actual_tools:
            failures.append(f"forbidden tool used: {forbidden_tool}")

    for expected_text in assertions.get("answer_contains", []):
        if expected_text.lower() not in trace.final_answer.lower():
            failures.append(f"answer does not contain {expected_text!r}")

    for forbidden_text in assertions.get("answer_not_contains", []):
        if forbidden_text.lower() in trace.final_answer.lower():
            failures.append(f"answer unexpectedly contains {forbidden_text!r}")

    if assertions.get("memory_valid", True):
        failures.extend(validate_tool_memory(trace.memory_after))

    if assertions.get("grounded_in_tool_results"):
        failures.extend(validate_grounded_answer(trace))

    if assertions.get("date_answer_matches_tool_result"):
        failures.extend(validate_date_answer(trace))

    if assertions.get("fallback_used") is not None:
        expected_fallback = assertions["fallback_used"]
        if trace.fallback_used != expected_fallback:
            failures.append(
                f"expected fallback_used={expected_fallback}, got {trace.fallback_used}"
            )

    return AssertionResult(passed=not failures, failures=failures)


def _assertions_for_profile(turn: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return assertions for a deterministic or live eval profile."""
    profile_assertions = turn.get(f"{profile}_assertions")
    if profile_assertions is not None:
        return profile_assertions

    return turn.get("assertions", {})


def validate_tool_memory(memory: list[dict[str, Any]]) -> list[str]:
    """Validate assistant tool calls and tool-result messages are paired."""
    failures = []
    pending_tool_call_ids: set[str] = set()

    for index, message in enumerate(memory):
        role = message.get("role")

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            pending_tool_call_ids = {
                tool_call["id"] for tool_call in tool_calls if "id" in tool_call
            }
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id not in pending_tool_call_ids:
                failures.append(
                    f"tool message at index {index} has unmatched tool_call_id "
                    f"{tool_call_id!r}"
                )
            continue

        if role != "system":
            pending_tool_call_ids = set()

    return failures


def validate_grounded_answer(trace: AgentTurnTrace) -> list[str]:
    """Check that obvious file names, sizes, and access claims match tool results."""
    failures = []
    answer = trace.final_answer.lower()
    tool_text = "\n".join(result["content"] for result in trace.tool_results).lower()

    if any(phrase in answer for phrase in ("don't have access", "do not have access", "cannot check")):
        if any(_tool_result_succeeded(result) for result in trace.tool_results):
            failures.append("answer denies access despite successful tool result")

    for filename in re.findall(r"[\w.-]+\.py", answer):
        if filename.lower() not in tool_text:
            failures.append(f"answer mentions ungrounded Python file {filename!r}")

    for size in re.findall(r"\b\d+\s+bytes\b", answer):
        number = size.split()[0]
        if number not in tool_text:
            failures.append(f"answer mentions ungrounded byte size {size!r}")

    return failures


def validate_date_answer(trace: AgentTurnTrace) -> list[str]:
    """Check relative date answers against the get_current_time tool result."""
    failures = []
    current_time = _current_time_from_trace(trace)

    if current_time is None:
        failures.append("no successful get_current_time result found")
        return failures

    answer = trace.final_answer
    user_input = trace.user_input.lower()
    today = current_time.date()

    if any(word in user_input for word in ("today", "current date")):
        _assert_date_present(failures, answer, today, "today")

    if "tomorrow" in user_input:
        _assert_date_present(
            failures,
            answer,
            today + dt.timedelta(days=1),
            "tomorrow",
        )

    if "yesterday" in user_input:
        _assert_date_present(
            failures,
            answer,
            today - dt.timedelta(days=1),
            "yesterday",
        )

    return failures


def _assert_date_present(
    failures: list[str],
    answer: str,
    expected_date: dt.date,
    label: str,
) -> None:
    """Append a failure when neither ISO nor readable date text is present."""
    iso_date = expected_date.isoformat()
    readable_date = expected_date.strftime("%B %-d, %Y")

    if iso_date not in answer and readable_date not in answer:
        failures.append(f"{label} date {iso_date} not found in answer")


def _current_time_from_trace(trace: AgentTurnTrace) -> dt.datetime | None:
    """Return the parsed get_current_time result from a trace, if present."""
    for result in trace.tool_results:
        if result["name"] != "get_current_time":
            continue

        try:
            return dt.datetime.fromisoformat(result["content"])
        except ValueError:
            return None

    return None


def _tool_result_succeeded(result: dict[str, Any]) -> bool:
    """Return whether a tool result does not represent a validation/runtime error."""
    content = result["content"]

    return not (
        content.startswith("Invalid ")
        or content.startswith("Tool execution failed")
        or content.startswith("Unknown tool")
    )
