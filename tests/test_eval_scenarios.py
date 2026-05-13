from __future__ import annotations

import datetime as dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_tool_agent.eval.runner import run_scenarios
from py_tool_agent.eval.workspace import prepare_workspace


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

    def test_prepare_workspace_uses_scenario_config(self) -> None:
        """Create live workspace files from scenario workspace metadata."""
        with TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "scenarios.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "workspace": {
                            "reference_time": "2026-05-08T14:01:32",
                            "files": [
                                {
                                    "path": "backend.py",
                                    "content": "print('example')\n",
                                    "size_bytes": 1121,
                                    "modified_at": {
                                        "date": "yesterday",
                                        "time": "16:15:00",
                                    },
                                }
                            ],
                        },
                        "scenarios": [],
                    }
                ),
                encoding="utf-8",
            )
            workspace = prepare_workspace(
                Path(temp_dir) / "workspace",
                scenario_path,
            )

            backend = workspace / "backend.py"
            metadata = backend.stat()
            modified_at = dt.datetime.fromtimestamp(metadata.st_mtime)

            self.assertTrue(backend.exists())
            self.assertEqual(1121, metadata.st_size)
            self.assertEqual(
                dt.datetime(2026, 5, 7, 16, 15),
                modified_at.replace(microsecond=0),
            )


if __name__ == "__main__":
    unittest.main()
