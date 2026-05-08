from __future__ import annotations

import unittest
from pathlib import Path

from py_tool_agent.eval.runner import run_scenarios


class EvalScenarioTests(unittest.TestCase):
    """Regression tests for deterministic behavioral eval scenarios."""

    def test_eval_scenarios_pass(self) -> None:
        """Run every scenario and fail with readable turn-level messages."""
        results = run_scenarios(Path("tests/eval_scenarios.json"))
        failures = []

        for result in results:
            for index, turn in enumerate(result.turns, start=1):
                if turn.passed:
                    continue
                failures.append(
                    f"{result.id} turn {index} {turn.input!r}: {turn.failures}"
                )

        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
