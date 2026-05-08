# src/py_tool_agent/agent.py
"""
Small tool-using agent.

The agent owns conversation orchestration only:
- decide whether tools are relevant through the model adapter
- ask the LLM for a tool call or direct answer
- execute validated registered tools
- ask the LLM for the final natural-language answer
- apply deterministic fallbacks when weaker local models ignore valid tool results
"""

from __future__ import annotations

import json
from typing import Any

from py_tool_agent.llm import LLMClient
from py_tool_agent.model_adapters import ModelAdapter, adapter_for_model
from py_tool_agent.tool_registry import ToolRegistry
from py_tool_agent.tools import DEFAULT_TOOL_REGISTRY
from py_tool_agent.tracing import AgentTurnTrace, to_trace_data, tool_call_to_dict


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


SYSTEM_PROMPT = """
Identity:
- You are a helpful, professional assistant.
- You are being used to test local Ollama models, including Qwen models.
- Do not mention implementation details, model internals, or tooling unless the user asks.

Response style:
- Be clear, direct, and practical.
- Keep answers concise unless the user asks for detail.
- Ask a brief clarifying question when the request is ambiguous.
- Do not invent facts, tool results, files, or command outputs.
- If you cannot complete a request, explain the limitation plainly.

Tool rules:
- You may use tools when they are available, but tool use is optional.
- Call a tool only when the user asks for information or an action that the tool can provide.
- If the user greets you, asks a general question, or asks what you can do, answer naturally without calling a tool.
- Never call a tool just to demonstrate capabilities.
- Never call a tool that is not listed in the available tools.
- Never claim that you checked files, directories, time, or calculations unless you used a tool and received a result.
- Do not write shell commands as a substitute for using an available tool.
- After receiving a tool result, produce a final answer in natural language.
"""


FINAL_ANSWER_PROMPT = (
    "Use the preceding tool result messages to answer the user's request. "
    "Return a concise natural language answer. Do not call tools."
)


class ToolAgent:
    """Coordinate LLM turns, tool execution, and deterministic fallbacks."""

    def __init__(
        self,
        llm: LLMClient,
        max_steps: int = 5,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
        adapter: ModelAdapter | None = None,
        verbose: bool = True,
    ) -> None:
        """Create an agent with an LLM client, tool registry, and model adapter."""
        self.llm = llm
        self.max_steps = max_steps
        self.registry = registry
        self.adapter = adapter or adapter_for_model(getattr(llm, "model", ""), registry)
        self.verbose = verbose
        self.memory: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()}
        ]
        self.last_trace: AgentTurnTrace | None = None
        self.trace_history: list[AgentTurnTrace] = []

    def run(self, user_input: str) -> str:
        """Handle one user turn and return the assistant's final answer."""
        trace = AgentTurnTrace(user_input=user_input)
        self.memory.append({"role": "user", "content": user_input})

        if self.adapter.is_capability_question(user_input):
            answer = self._append_assistant_answer(self.registry.capability_text())
            trace.final_answer = answer
            self._finish_trace(trace)

            return answer

        for _ in range(self.max_steps):
            tools_enabled = self.adapter.should_expose_tools(user_input, self.memory)
            trace.tools_enabled = tools_enabled
            tool_schemas = self.adapter.tool_schemas() if tools_enabled else None

            self._display_info(
                "Tool access",
                tools_enabled,
                color=GREEN if tools_enabled else YELLOW,
            )

            response = self._chat(
                messages=self.memory,
                tools=tool_schemas,
                failure_context=(
                    "LLM request failed while tools were "
                    f"{'enabled' if tools_enabled else 'disabled'}"
                ),
            )
            if isinstance(response, str):
                trace.error = response
                trace.final_answer = response
                self._finish_trace(trace)

                return response

            message = response.choices[0].message
            trace.llm_message = to_trace_data(message)
            tool_calls, normalized_tool_calls = ([], False)
            if tools_enabled:
                tool_calls, normalized_tool_calls = self.adapter.extract_tool_calls(
                    message,
                    user_input,
                    self.memory,
                )
            else:
                native_tool_calls = getattr(message, "tool_calls", None)
                if native_tool_calls:
                    self._display_info(
                        "Ignored tool calls",
                        native_tool_calls,
                        color=YELLOW,
                    )

            self._display_info("LLM message", message, color=CYAN)
            self._display_info("Tool calls", tool_calls or "none", color=MAGENTA)
            trace.tool_calls = [tool_call_to_dict(tool_call) for tool_call in tool_calls]
            trace.normalized_tool_calls = normalized_tool_calls

            if not tools_enabled:
                self.memory.append(message.model_dump())
                answer = message.content or ""
                trace.final_answer = answer
                self._finish_trace(trace)

                return answer

            self.memory.append(
                self.adapter.message_dump(
                    message,
                    tool_calls,
                    normalized_tool_calls,
                )
            )

            if not tool_calls:
                answer = message.content or ""
                trace.final_answer = answer
                self._finish_trace(trace)

                return answer

            tool_results = self._execute_tool_calls(tool_calls)
            trace.tool_results = [to_trace_data(result) for result in tool_results]
            immediate_answer = self._immediate_tool_answer(user_input, tool_results)
            if immediate_answer:
                self.memory.append({"role": "assistant", "content": immediate_answer})
                trace.final_answer = immediate_answer
                trace.fallback_used = True
                trace.immediate_answer_used = True
                self._finish_trace(trace)

                return immediate_answer

            final_answer, fallback_used, final_message = self._final_answer(
                user_input,
                tool_results,
            )
            self.memory.append({"role": "assistant", "content": final_answer})
            trace.final_answer = final_answer
            trace.fallback_used = fallback_used
            trace.final_message = final_message
            self._finish_trace(trace)

            return final_answer

        answer = "J’ai atteint la limite de boucle sans réponse finale fiable."
        trace.final_answer = answer
        self._finish_trace(trace)

        return answer

    def _execute_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Validate and execute normalized tool calls through the registry."""
        tool_results = []

        for tool_call in tool_calls:
            tool_result = self.registry.execute(tool_call)
            tool_results.append(tool_result)
            self.memory.append(tool_result)
            self._display_info("Tool result", tool_result, color=GREEN)

        return tool_results

    def _immediate_tool_answer(
        self,
        user_input: str,
        tool_results: list[dict[str, Any]],
    ) -> str | None:
        """Return a deterministic answer when a tool result is already sufficient."""
        if not any(result["name"] == "list_current_directory" for result in tool_results):
            return None

        return self.registry.fallback_answer(
            self.adapter.fallback_input(user_input, self.memory),
            tool_results,
        )

    def _final_answer(
        self,
        user_input: str,
        tool_results: list[dict[str, Any]],
    ) -> tuple[str, bool, dict[str, Any] | None]:
        """Ask the model to summarize tool results, then fall back if needed."""
        final_response = self._chat(
            messages=[
                *self.memory,
                {"role": "system", "content": FINAL_ANSWER_PROMPT},
            ],
            tools=None,
            failure_context="LLM request failed after tool execution",
        )
        if isinstance(final_response, str):
            return final_response, False, None

        final_message = final_response.choices[0].message
        self._display_info("Final message", final_message, color=CYAN)
        final_message_data = to_trace_data(final_message)

        if final_message.content and not self.adapter.should_use_fallback(
            final_message.content,
            tool_results,
        ):
            return final_message.content, False, final_message_data

        fallback_answer = self.registry.fallback_answer(
            self.adapter.fallback_input(user_input, self.memory),
            tool_results,
        )
        if fallback_answer:
            return fallback_answer, True, final_message_data

        if final_message.content:
            return final_message.content, False, final_message_data

        return self._raw_tool_result_answer(tool_results), True, final_message_data

    def _chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        failure_context: str,
    ) -> Any | str:
        """Call the LLM and convert provider exceptions into user-facing text."""
        try:
            return self.llm.chat(messages=messages, tools=tools)
        except Exception as exc:
            return f"{failure_context}: {type(exc).__name__}: {exc}"

    def _append_assistant_answer(self, answer: str) -> str:
        """Store a deterministic assistant answer in memory and return it."""
        self.memory.append({"role": "assistant", "content": answer})

        return answer

    def _finish_trace(self, trace: AgentTurnTrace) -> None:
        """Finalize and retain a trace for the completed user turn."""
        trace.finish(self.memory)
        self.last_trace = trace
        self.trace_history.append(trace)

    @staticmethod
    def _raw_tool_result_answer(tool_results: list[dict[str, Any]]) -> str:
        """Format raw tool results when neither model nor registry can summarize."""
        if not tool_results:
            return "The model returned an empty final answer after tool execution."

        formatted_results = "\n\n".join(
            f"Tool result from {result['name']}:\n{result['content']}"
            for result in tool_results
        )

        return (
            "The model returned an empty final answer after tool execution.\n\n"
            f"{formatted_results}"
        )

    def _display_info(self, title: str, value: Any, *, color: str = CYAN) -> None:
        """Print a labeled, colorized diagnostic panel to the console."""
        if not self.verbose:
            return

        print(f"{DIM}╭─{RESET} {color}{BOLD}{title}{RESET}")
        print(f"{DIM}╰─{RESET} {ToolAgent._format_console_value(value)}")

    @staticmethod
    def _format_console_value(value: Any) -> str:
        """Convert arbitrary diagnostic values into readable console text."""
        if isinstance(value, bool):
            return f"{GREEN}yes{RESET}" if value else f"{YELLOW}no{RESET}"

        if value is None:
            return f"{DIM}none{RESET}"

        try:
            value = ToolAgent._to_console_data(value)
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _to_console_data(value: Any) -> Any:
        """Recursively convert model objects into JSON-serializable structures."""
        if hasattr(value, "model_dump"):
            return value.model_dump()

        if isinstance(value, dict):
            return {
                key: ToolAgent._to_console_data(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [ToolAgent._to_console_data(item) for item in value]

        return value
