import ollama
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMBackend(ABC):
    @property    
    @abstractmethod
    def model_name(self):
        ...
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        ...

class ToolAgent:
    def __init__(self, llm: LLMBackend, tools: Dict[str, Any]):
        self.llm = llm
        self.tools = tools

    def run(self, user_input: str) -> str:
        # 1. Build a prompt describing tools
        # 2. Ask  LLM if any tools should be called with what arguments
        # 3. call the tool with python if needed
        # 4. get back received answer (hnading errors)
        return "TODO"


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, str]]) -> str:
        res = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.2},
        )
        return res["message"]["content"]