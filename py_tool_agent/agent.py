# src/py_tool_agent/agent.py
""" 
very basic agent workflow :

User message
   ↓
Add to conversation memory
   ↓
Ask LLM: answer directly or call tool?
   ↓
If tool call:
   - validate tool name
   - validate arguments
   - execute Python function
   - inject result back into conversation
   - ask LLM for final answer
   ↓
Return final answer 

KEY POINT : the LLM MUST NEVER execute code freely it could only ask calling tools explicitly registered
"""


from __future__ import annotations

import datetime as dt
import json
import re
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

from py_tool_agent.llm import LLMClient
from py_tool_agent.tools import TOOLS, TOOL_ARGUMENT_MODELS, TOOL_SCHEMAS


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

Available capabilities:
- get_current_time: get the current local date and time.
- add_numbers: add two numbers.
- list_current_directory: list files in the current working directory with ls -al style metadata.
"""


class ToolAgent:
    def __init__(
        self,
        llm: LLMClient,
        max_steps: int = 5,
    ) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.memory: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()}
        ]

    
    def run(self, user_input: str) -> str:
        self.memory.append({"role": "user", "content": user_input})

        if self._is_capability_question(user_input):
            answer = self._capability_answer()
            self.memory.append({"role": "assistant", "content": answer})
            return answer

        for _ in range(self.max_steps):
            tools_enabled = self._should_expose_tools(user_input)
            tools = TOOL_SCHEMAS if tools_enabled else None

            self._display_info(
                "Tool access",
                tools_enabled,
                color=GREEN if tools_enabled else YELLOW,
            )

            try:
                response = self.llm.chat(
                    messages=self.memory,
                    tools=tools,
                )
            except Exception as exc:
                return (
                    "LLM request failed while tools were "
                    f"{'enabled' if tools_enabled else 'disabled'}: "
                    f"{type(exc).__name__}: {exc}"
                )

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            extracted_text_tool_calls = False
            if tools_enabled and not tool_calls:
                tool_calls = self._extract_text_tool_calls(message.content or "")
                tool_calls = self._filter_tool_calls_for_intent(user_input, tool_calls)
                extracted_text_tool_calls = bool(tool_calls)
            if tools_enabled and not tool_calls:
                tool_calls = self._infer_tool_calls(user_input)
                extracted_text_tool_calls = bool(tool_calls)

            self._display_info("LLM message", message, color=CYAN)
            self._display_info("Tool calls", tool_calls or "none", color=MAGENTA)

            # Important:
            # If tools were not exposed, never execute tool calls,
            # even if the model produced some.
            if not tools_enabled:
                if tool_calls:
                    self._display_info(
                        "Ignored tool calls",
                        tool_calls,
                        color=YELLOW,
                    )

                    # Do NOT append the broken tool_call message to memory.
                    # Ask again for a plain natural language answer.
                    correction_messages = [
                        *self.memory,
                        {
                            "role": "system",
                            "content": (
                                "Direct answer mode is active. "
                                "Do not call tools. "
                                "Answer the user's last message in natural language only."
                            ),
                        },
                    ]

                    try:
                        retry_response = self.llm.chat(
                            messages=correction_messages,
                            tools=None,
                        )
                    except Exception as exc:
                        return (
                            "LLM retry failed after disabling tools: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    retry_message = retry_response.choices[0].message
                    self.memory.append(retry_message.model_dump())

                    return retry_message.content or ""

                self.memory.append(message.model_dump())
                return message.content or ""

            # From here, tools are enabled.
            self.memory.append(
                self._message_dump(message, tool_calls, extracted_text_tool_calls)
            )

            if not tool_calls:
                return message.content or ""

            tool_results: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                tool_results.append(tool_result)
                self.memory.append(tool_result)
                self._display_info("Tool result", tool_result, color=GREEN)

            try:
                final_messages = [
                    *self.memory,
                    {
                        "role": "system",
                        "content": (
                            "Use the preceding tool result messages to answer "
                            "the user's request. Return a concise natural "
                            "language answer. Do not call tools."
                        ),
                    },
                ]

                final_response = self.llm.chat(
                    messages=final_messages,
                    tools=None,
                )
            except Exception as exc:
                return (
                    "LLM request failed after tool execution: "
                    f"{type(exc).__name__}: {exc}"
                )

            final_message = final_response.choices[0].message
            self._display_info("Final message", final_message, color=CYAN)

            if final_message.content:
                if self._should_use_fallback(final_message.content, tool_results):
                    fallback_answer = self._fallback_tool_answer(user_input, tool_results)
                    self.memory.append({"role": "assistant", "content": fallback_answer})

                    return fallback_answer

                self.memory.append(final_message.model_dump())
                return final_message.content

            fallback_answer = self._fallback_tool_answer(user_input, tool_results)
            self.memory.append({"role": "assistant", "content": fallback_answer})

            return fallback_answer

        return "J’ai atteint la limite de boucle sans réponse finale fiable."

    @staticmethod
    def _is_capability_question(user_input: str) -> bool:
        text = user_input.lower().strip()

        return any(
            phrase in text
            for phrase in (
                "what can you do",
                "what are your capabilities",
                "what tools",
                "available tools",
                "help me with",
            )
        )

    @staticmethod
    def _capability_answer() -> str:
        return (
            "I can answer general questions directly. I can also use registered "
            "tools to get the current local date and time, add two numbers, and "
            "list files in the current working directory with ls -al style metadata."
        )

    @staticmethod
    def _display_info(title: str, value: Any, *, color: str = CYAN) -> None:
        print(f"{DIM}╭─{RESET} {color}{BOLD}{title}{RESET}")
        print(f"{DIM}╰─{RESET} {ToolAgent._format_console_value(value)}")

    @staticmethod
    def _format_console_value(value: Any) -> str:
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

    @staticmethod
    def _message_dump(
        message: Any,
        tool_calls: Any,
        extracted_text_tool_calls: bool,
    ) -> dict[str, Any]:
        message_dump = message.model_dump()

        if extracted_text_tool_calls:
            message_dump["content"] = ""
            message_dump["tool_calls"] = [
                ToolAgent._tool_call_dump(tool_call) for tool_call in tool_calls
            ]

        return message_dump

    @staticmethod
    def _tool_call_dump(tool_call: Any) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    @staticmethod
    def _extract_text_tool_calls(content: str) -> list[Any]:
        candidates = ToolAgent._json_candidates(content)
        tool_calls = []

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if name not in TOOLS or not isinstance(arguments, dict):
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

    def _infer_tool_calls(self, user_input: str) -> list[Any]:
        tool_names = self._infer_tool_names(user_input)

        return [
            SimpleNamespace(
                id=f"inferred_tool_call_{index}",
                type="function",
                function=SimpleNamespace(name=name, arguments="{}"),
            )
            for index, name in enumerate(tool_names, start=1)
        ]

    def _infer_tool_names(self, user_input: str) -> list[str]:
        text = self._tool_intent_text(user_input)
        tool_names: list[str] = []

        if any(word in text for word in ("date", "time", "tomorrow", "demain", "heure")):
            tool_names.append("get_current_time")

        if any(
            phrase in text
            for phrase in (
                "list files",
                "show files",
                "files from today",
                "current directory",
                "working directory",
                "dossier courant",
                "répertoire courant",
                "repertoire courant",
            )
        ):
            tool_names.append("list_current_directory")

        return list(dict.fromkeys(tool_names))

    def _filter_tool_calls_for_intent(
        self,
        user_input: str,
        tool_calls: list[Any],
    ) -> list[Any]:
        inferred_tool_names = self._infer_tool_names(user_input)

        if not inferred_tool_names:
            return tool_calls

        return [
            tool_call
            for tool_call in tool_calls
            if tool_call.function.name in inferred_tool_names
        ]

    def _tool_intent_text(self, user_input: str) -> str:
        text = user_input.lower().strip()
        confirmations = {"yes", "yes please", "y", "oui", "ok", "sure", "just do it", "do it", "go ahead"}

        if text not in confirmations:
            return text

        recent_messages = []
        for message in reversed(self.memory[:-1]):
            if message.get("role") in {"user", "assistant"}:
                recent_messages.append(message.get("content") or "")
            if len(recent_messages) == 2:
                break

        return " ".join(reversed(recent_messages)).lower()

    @staticmethod
    def _json_candidates(content: str) -> list[str]:
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

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        function_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"

        if function_name not in TOOLS:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": f"Unknown tool: {function_name}",
            }

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": f"Invalid JSON arguments: {exc}",
            }

        argument_model = TOOL_ARGUMENT_MODELS[function_name]
        try:
            validated_arguments = argument_model.model_validate(arguments).model_dump()
        except ValidationError as exc:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": (
                    "Invalid tool arguments: "
                    + json.dumps(
                        exc.errors(include_url=False, include_input=False),
                        ensure_ascii=False,
                    )
                ),
            }

        try:
            result = TOOLS[function_name](**validated_arguments)
        except Exception as exc:
            result = f"Tool execution failed: {type(exc).__name__}: {exc}"

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result),
        }

    @staticmethod
    def _fallback_tool_answer(
        user_input: str,
        tool_results: list[dict[str, Any]],
    ) -> str:
        if not tool_results:
            return "The model returned an empty final answer after tool execution."

        current_time_result = next(
            (
                result
                for result in tool_results
                if result["name"] == "get_current_time"
            ),
            None,
        )
        if current_time_result:
            parsed_time = ToolAgent._parse_iso_datetime(current_time_result["content"])
            if parsed_time:
                text = user_input.lower()
                lines = [f"The current date is {parsed_time.date().isoformat()}."]

                if "tomorrow" in text or "demain" in text:
                    tomorrow = parsed_time.date() + dt.timedelta(days=1)
                    lines.append(f"Tomorrow's date is {tomorrow.isoformat()}.")

                return " ".join(lines)

        if len(tool_results) == 1:
            result = tool_results[0]
            return (
                "The model returned an empty final answer after tool execution.\n\n"
                f"Tool result from {result['name']}:\n{result['content']}"
            )

        formatted_results = "\n\n".join(
            f"Tool result from {result['name']}:\n{result['content']}"
            for result in tool_results
        )

        return (
            "The model returned an empty final answer after tool execution.\n\n"
            f"{formatted_results}"
        )

    @staticmethod
    def _parse_iso_datetime(value: str) -> dt.datetime | None:
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _should_use_fallback(
        final_content: str,
        tool_results: list[dict[str, Any]],
    ) -> bool:
        text = final_content.lower()
        has_successful_time_result = any(
            result["name"] == "get_current_time"
            and ToolAgent._parse_iso_datetime(result["content"]) is not None
            for result in tool_results
        )

        if not has_successful_time_result:
            return False

        return any(
            phrase in text
            for phrase in (
                "don't have access",
                "do not have access",
                "provide the current date",
                "provide the current time",
                "manually",
            )
        )

    def _should_expose_tools(self, user_input: str) -> bool:
        text = user_input.lower().strip()
        tool_triggers = [
            "heure",
            "date",
            "time",
            "current time",
            "what time",
            "additionne",
            "ajoute",
            "calcule",
            "combien font",
            "add",
            "sum",
            "calculate",
            "compute",
            "liste les fichiers",
            "lister les fichiers",
            "affiche les fichiers",
            "dossier courant",
            "répertoire courant",
            "repertoire courant",
            "list files",
            "show files",
            "files",
            "directory",
            "folder",
            "current directory",
            "working directory",
            "today",
            "modified",
            "recent",
        ]

        if any(trigger in text for trigger in tool_triggers):
            return True

        confirmations = {
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
        if text in confirmations:
            return self._last_assistant_offered_tool_followup()

        return False

    def _last_assistant_offered_tool_followup(self) -> bool:
        for message in reversed(self.memory[:-1]):
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
                or any(tool_name in text for tool_name in TOOLS)
                or "[insert today's date]" in text
                or "perform these actions" in text
                or (
                    "would you like me to" in text
                    and any(word in text for word in ("date", "time", "tomorrow", "files"))
                )
            )

        return False
