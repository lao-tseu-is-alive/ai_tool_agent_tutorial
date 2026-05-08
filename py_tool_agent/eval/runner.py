from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from py_tool_agent.agent import ToolAgent
from py_tool_agent.eval.assertions import assert_turn
from py_tool_agent.eval.fixtures import ScriptedLLM, fixture_registry, flatten_scripted_responses


PASS_MARK = "✅"
FAIL_MARK = "❌"


@dataclass
class TurnEvalResult:
    """Evaluation result for one scenario turn."""

    input: str
    passed: bool
    failures: list[str]
    trace: dict[str, Any]


@dataclass
class ScenarioEvalResult:
    """Evaluation result for one scenario."""

    id: str
    description: str
    model: str
    passed: bool
    turns: list[TurnEvalResult] = field(default_factory=list)


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load scenario definitions from a JSON file."""
    return json.loads(path.read_text())


def run_scenarios(path: Path, model: str = "deterministic-fixture") -> list[ScenarioEvalResult]:
    """Run all deterministic scenarios from a file."""
    return [run_scenario(scenario, model=model) for scenario in load_scenarios(path)]


def run_scenario(
    scenario: dict[str, Any],
    *,
    model: str = "deterministic-fixture",
) -> ScenarioEvalResult:
    """Run one scenario with fixture-backed tools and scripted model responses."""
    llm = ScriptedLLM(
        flatten_scripted_responses(scenario),
        model=model,
    )
    agent = ToolAgent(
        llm=llm,
        registry=fixture_registry(scenario.get("fixtures", {})),
        verbose=False,
    )
    turns: list[TurnEvalResult] = []

    for turn in scenario["turns"]:
        agent.run(turn["input"])
        trace = agent.last_trace
        if trace is None:
            raise AssertionError("agent did not produce a trace")

        assertion_result = assert_turn(trace, turn)
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
        model=model,
        passed=all(turn.passed for turn in turns),
        turns=turns,
    )


def write_report(results: list[ScenarioEvalResult], out_dir: Path) -> None:
    """Write machine-readable and Markdown evaluation reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_data = [asdict(result) for result in results]
    (out_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, ensure_ascii=False)
    )
    (out_dir / "summary.md").write_text(_markdown_summary(results))


def main() -> None:
    """Run deterministic scenario evals from the command line."""
    parser = argparse.ArgumentParser(description="Run deterministic agent eval scenarios.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/eval_scenarios.json"),
        help="Path to the scenario JSON file.",
    )
    parser.add_argument(
        "--model",
        default="deterministic-fixture",
        help="Model label to store in the report.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory for JSON and Markdown reports.",
    )
    args = parser.parse_args()

    results = run_scenarios(args.scenarios, model=args.model)
    if args.out:
        write_report(results, args.out)

    print(_markdown_summary(results))
    if not all(result.passed for result in results):
        raise SystemExit(1)


def _markdown_summary(results: list[ScenarioEvalResult]) -> str:
    """Build a compact Markdown summary for humans."""
    passed = sum(1 for result in results if result.passed)
    overall_status = PASS_MARK if passed == len(results) else FAIL_MARK
    lines = [
        "# Agent Eval Summary",
        "",
        f"{overall_status} Scenarios: {passed}/{len(results)} passed",
        "",
    ]

    for result in results:
        status = PASS_MARK if result.passed else FAIL_MARK
        lines.append(f"## {status} {result.id}")
        if result.description:
            lines.append(result.description)
        lines.append("")

        for index, turn in enumerate(result.turns, start=1):
            turn_status = PASS_MARK if turn.passed else FAIL_MARK
            tools = [
                tool_call["function"]["name"]
                for tool_call in turn.trace.get("tool_calls", [])
            ]
            lines.append(f"- Turn {index}: {turn_status} `{turn.input}`")
            lines.append(f"  - tools: {tools}")
            lines.append(f"  - answer: {turn.trace.get('final_answer', '')}")
            if turn.failures:
                for failure in turn.failures:
                    lines.append(f"  - failure: {failure}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
