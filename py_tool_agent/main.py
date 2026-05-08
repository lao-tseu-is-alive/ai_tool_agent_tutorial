# src/py_tool_agent/main.py

from __future__ import annotations

from dotenv import load_dotenv

from py_tool_agent.agent import ToolAgent
from py_tool_agent.llm import LLMClient


def main() -> None:
    """Run the interactive command-line loop for the tool agent."""
    load_dotenv()

    llm = LLMClient.from_env()
    agent = ToolAgent(llm=llm)

    print(f"Agent started with model: {llm.model}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input(">>> ").strip()
        except KeyboardInterrupt:
            print("\nbye")
            break

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        answer = agent.run(user_input)
        print(answer)


if __name__ == "__main__":
    main()
