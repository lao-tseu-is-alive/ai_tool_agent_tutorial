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

import json
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

        for _ in range(self.max_steps):
            tools_enabled = self._should_expose_tools(user_input)
            tools = TOOL_SCHEMAS if tools_enabled else None

            self._display_info(
                "Tool access",
                tools_enabled,
                color=GREEN if tools_enabled else YELLOW,
            )

            response = self.llm.chat(
                messages=self.memory,
                tools=tools,
            )

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

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

                    retry_response = self.llm.chat(
                        messages=correction_messages,
                        tools=None,
                    )

                    retry_message = retry_response.choices[0].message
                    self.memory.append(retry_message.model_dump())

                    return retry_message.content or ""

                self.memory.append(message.model_dump())
                return message.content or ""

            # From here, tools are enabled.
            self.memory.append(message.model_dump())

            if not tool_calls:
                return message.content or ""

            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                self.memory.append(tool_result)

            final_response = self.llm.chat(
                messages=self.memory,
                tools=None,
            )

            final_message = final_response.choices[0].message
            self.memory.append(final_message.model_dump())

            return final_message.content or ""

        return "J’ai atteint la limite de boucle sans réponse finale fiable."

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

        confirmations = {"yes", "yes please", "y", "oui", "ok", "sure"}
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
                "would you like" in text
                and "detail" in text
                and any(word in text for word in ("file", "directory", "folder"))
            )

        return False
