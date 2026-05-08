from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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


@dataclass
class EvalRunMetadata:
    """Metadata that makes eval reports comparable across runs."""

    created_at: str
    model: str
    provider_model: str
    scenarios_path: str
    runner: str
    temperature: float | None = None
    timeout_seconds: float | None = None
    context_size: int | None = None
    api_base: str | None = None


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load scenario definitions from a JSON file."""
    data = json.loads(path.read_text())

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]

    raise ValueError(
        "scenario file must be a list of scenarios or an object with a "
        "'scenarios' list"
    )


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

        assertion_result = assert_turn(trace, turn, profile="deterministic")
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


def write_report(
    results: list[ScenarioEvalResult],
    out_dir: Path,
    metadata: EvalRunMetadata,
) -> None:
    """Write machine-readable and Markdown evaluation reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "run": asdict(metadata),
        "scenarios": [asdict(result) for result in results],
    }
    (out_dir / "results.json").write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False)
    )
    (out_dir / "summary.md").write_text(_markdown_summary(results, metadata))


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
    metadata = build_metadata(args.scenarios, args.model)
    if args.out:
        write_report(results, args.out, metadata)

    print(_markdown_summary(results, metadata))
    if not all(result.passed for result in results):
        raise SystemExit(1)


def build_metadata(
    scenarios_path: Path,
    model: str,
    *,
    runner: str = "deterministic-fixture",
) -> EvalRunMetadata:
    """Build run metadata from CLI args and environment configuration."""
    is_ollama_model = model.startswith("ollama/")

    return EvalRunMetadata(
        created_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        model=model,
        provider_model=_provider_model_name(model),
        scenarios_path=str(scenarios_path),
        temperature=_float_env("AGENT_TEMPERATURE", default=0.2 if is_ollama_model else None),
        timeout_seconds=_float_env("AGENT_REQUEST_TIMEOUT", default=20 if is_ollama_model else None),
        context_size=_int_env("AGENT_CONTEXT_SIZE"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        if is_ollama_model
        else None,
        runner=runner,
    )


def _markdown_summary(
    results: list[ScenarioEvalResult],
    metadata: EvalRunMetadata,
) -> str:
    """Build a compact Markdown summary for humans."""
    passed = sum(1 for result in results if result.passed)
    overall_status = PASS_MARK if passed == len(results) else FAIL_MARK
    lines = [
        "# Agent Eval Summary",
        "",
        f"- created_at: `{metadata.created_at}`",
        f"- model: `{metadata.model}`",
        f"- provider_model: `{metadata.provider_model}`",
        f"- runner: `{metadata.runner}`",
        f"- temperature: `{metadata.temperature}`",
        f"- timeout_seconds: `{metadata.timeout_seconds}`",
        f"- context_size: `{metadata.context_size}`",
        f"- api_base: `{metadata.api_base}`",
        f"- scenarios: `{metadata.scenarios_path}`",
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


def _provider_model_name(model: str) -> str:
    """Return the provider-facing model name used by LiteLLM."""
    if model.startswith("ollama/"):
        return model.replace("ollama/", "ollama_chat/", 1)

    return model


def _float_env(name: str, *, default: float | None = None) -> float | None:
    """Return an environment variable parsed as float, if available."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return None


def _int_env(name: str) -> int | None:
    """Return an environment variable parsed as int, if available."""
    value = os.getenv(name)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
