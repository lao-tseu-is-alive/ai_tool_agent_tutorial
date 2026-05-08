from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv

from py_tool_agent.agent import ToolAgent
from py_tool_agent.eval.assertions import assert_turn
from py_tool_agent.eval.runner import (
    ScenarioEvalResult,
    TurnEvalResult,
    build_metadata,
    load_scenarios,
    write_report,
    _markdown_summary,
)
from py_tool_agent.llm import LLMClient


def run_live_scenarios(
    path: Path,
    *,
    llm: LLMClient,
    verbose: bool = False,
    workdir: Path | None = None,
) -> list[ScenarioEvalResult]:
    """Run scenario prompts against a real LLM client."""
    with _working_directory(workdir):
        return [
            run_live_scenario(scenario, llm=llm, verbose=verbose)
            for scenario in load_scenarios(path)
        ]


def run_live_scenario(
    scenario: dict,
    *,
    llm: LLMClient,
    verbose: bool = False,
) -> ScenarioEvalResult:
    """Run one scenario against the selected real model."""
    agent = ToolAgent(llm=llm, verbose=verbose)
    turns: list[TurnEvalResult] = []

    for turn in scenario["turns"]:
        agent.run(turn["input"])
        trace = agent.last_trace
        if trace is None:
            raise AssertionError("agent did not produce a trace")

        assertion_result = assert_turn(trace, turn, profile="live")
        turns.append(
            TurnEvalResult(
                input=turn["input"],
                passed=assertion_result.passed,
                failures=assertion_result.failures,
                trace=trace.to_dict(),
            )
        )

    return ScenarioEvalResult(
        id=scenario["id"],
        description=scenario.get("description", ""),
        model=llm.model,
        passed=all(turn.passed for turn in turns),
        turns=turns,
    )


def main() -> None:
    """Run live behavioral evals against the selected model."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run live agent eval scenarios.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/eval_scenarios.json"),
        help="Path to the scenario JSON file.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL", "ollama/qwen3"),
        help="Real model to evaluate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show agent diagnostic panels while running live evals.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory used as the process cwd while live tools run.",
    )
    args = parser.parse_args()

    scenarios_path = args.scenarios.resolve()
    out_dir = args.out.resolve() if args.out else None
    workdir = args.workdir.resolve() if args.workdir else None
    llm = _llm_for_model(args.model)
    results = run_live_scenarios(
        scenarios_path,
        llm=llm,
        verbose=args.verbose,
        workdir=workdir,
    )
    metadata = build_metadata(scenarios_path, llm.model, runner="live-llm")

    if out_dir:
        write_report(results, out_dir, metadata)

    print(_markdown_summary(results, metadata))
    if not all(result.passed for result in results):
        raise SystemExit(1)


def _llm_for_model(model: str) -> LLMClient:
    """Create an LLM client for the selected live eval model."""
    api_base = None
    if model.startswith("ollama/"):
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    return LLMClient(
        model=model,
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
        api_base=api_base,
        timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "20")),
    )


@contextlib.contextmanager
def _working_directory(path: Path | None):
    """Temporarily switch cwd while tools execute."""
    if path is None:
        yield
        return

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
