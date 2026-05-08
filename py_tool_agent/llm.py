# src/py_tool_agent/llm.py

from __future__ import annotations

import os
from typing import Any

from litellm import completion


class LLMClient:
    """
    Thin wrapper around LiteLLM.

    Goal:
    - use Ollama locally during development
    - switch to OpenRouter later without changing the agent logic
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        api_base: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "LLMClient":
        model = os.getenv("AGENT_MODEL", "ollama/gemma4:26b")

        api_base = None
        if model.startswith("ollama/"):
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

        return cls(
            model=model,
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
            api_base=api_base,
            timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "20")),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model_for_request(),
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return completion(**kwargs)

    def _model_for_request(self) -> str:
        if self.model.startswith("ollama/"):
            return self.model.replace("ollama/", "ollama_chat/", 1)

        return self.model
