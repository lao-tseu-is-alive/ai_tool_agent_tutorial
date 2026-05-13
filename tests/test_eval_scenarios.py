from __future__ import annotations

import datetime as dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_tool_agent.eval.runner import run_scenarios
from py_tool_agent.eval.workspace import prepare_workspace
from py_tool_agent.model_adapters import ModelAdapter
from py_tool_agent.tools import DEFAULT_TOOL_REGISTRY


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
        scenario_path = Path("tests/eval_scenarios.json")
        scenario_data = json.loads(scenario_path.read_text(encoding="utf-8"))
        file_config = scenario_data["workspace"]["files"][0]
        modified_at_config = file_config["modified_at"]

        with TemporaryDirectory() as temp_dir:
            before_prepare = dt.datetime.now()
            workspace = prepare_workspace(
                Path(temp_dir) / "workspace",
                scenario_path,
            )
            after_prepare = dt.datetime.now()

            backend = workspace / file_config["path"]
            metadata = backend.stat()
            modified_at = dt.datetime.fromtimestamp(metadata.st_mtime)
            possible_expected_dates = {
                before_prepare.date() - dt.timedelta(days=1),
                after_prepare.date() - dt.timedelta(days=1),
            }

            self.assertTrue(backend.exists())
            self.assertEqual(file_config["size_bytes"], metadata.st_size)
            self.assertIn(modified_at.date(), possible_expected_dates)
            self.assertEqual(
                dt.time.fromisoformat(modified_at_config["time"]),
                modified_at.time().replace(microsecond=0),
            )

    def test_file_followup_uses_filename_from_recent_context(self) -> None:
        """Expose directory tools when a follow-up refers to a named prior file."""
        adapter = ModelAdapter(DEFAULT_TOOL_REGISTRY)
        memory = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Any python files from yesterday?"},
            {"role": "assistant", "content": "Matching files: app_worker.py."},
            {"role": "user", "content": "What's the size of this file?"},
        ]

        self.assertTrue(
            adapter.should_expose_tools("What's the size of this file?", memory)
        )

    def test_file_followup_ignores_vague_file_context(self) -> None:
        """Do not expose directory tools from a generic prior file mention."""
        adapter = ModelAdapter(DEFAULT_TOOL_REGISTRY)
        memory = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Can you explain what a file is?"},
            {"role": "assistant", "content": "A file stores data."},
            {"role": "user", "content": "What's the size of this file?"},
        ]

        self.assertFalse(
            adapter.should_expose_tools("What's the size of this file?", memory)
        )


if __name__ == "__main__":
    unittest.main()
