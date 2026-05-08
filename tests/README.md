# Tests

This directory contains deterministic behavioral tests for the tool agent.

The shortest path is:

```bash
./scripts/test.sh
```

That script runs:

1. Python compile checks.
2. The deterministic unittest suite.
3. The eval runner, which writes `eval_runs/latest/results.json` and `eval_runs/latest/summary.md`.

## Running The Unittest Directly

Use the dotted Python module name:

```bash
uv run python -m unittest tests.test_eval_scenarios
```

Do not use a slash path with `python -m unittest`:

```bash
uv run python -m unittest tests/test_eval_scenarios
```

That form is interpreted as an import name and fails with:

```text
ModuleNotFoundError: No module named 'tests/test_eval_scenarios'
```

## Scenario File

`eval_scenarios.json` defines scripted conversations and assertions.

Use these supporting files when adding scenarios:

- `eval_scenarios.example.jsonc` is a commented teaching example.
- `eval_scenarios.schema.json` documents and validates the real JSON shape in editors.

Each scenario can validate:

- which tools were used
- which tools must not be used
- final answer text
- grounding against tool results
- date calculations from `get_current_time`
- assistant/tool message pairing in agent memory

The current scenario protects the date-chaining and directory-listing behavior
seen while testing Ollama models.

## Adding A Simple Scenario

Add a new object to the top-level array in `eval_scenarios.json`:

```json
{
  "id": "simple_date_01",
  "description": "Checks today and tomorrow date handling.",
  "fixtures": {
    "current_time": "2026-05-08T14:01:32",
    "directory_listing": "total 0"
  },
  "turns": [
    {
      "input": "What is today's date and tomorrow's date?",
      "llm_responses": [
        {
          "content": "",
          "tool_calls": [
            {
              "id": "call_time_1",
              "name": "get_current_time",
              "arguments": {}
            }
          ]
        },
        {
          "content": "Today is May 14, 2025, and tomorrow is May 15, 2025."
        }
      ],
      "expected_tools": [
        "get_current_time"
      ],
      "forbidden_tools": [
        "add_numbers"
      ],
      "assertions": {
        "answer_contains": [
          "2026-05-08",
          "2026-05-09"
        ],
        "answer_not_contains": [
          "2025"
        ],
        "date_answer_matches_tool_result": true,
        "memory_valid": true,
        "fallback_used": true
      }
    }
  ]
}
```

The important sections are:

- `fixtures`: fixed tool outputs for deterministic tests.
- `turns`: user messages run sequentially against the same agent.
- `llm_responses`: fake model messages consumed by the agent.
- `expected_tools`: exact ordered tool calls expected after adapter normalization.
- `forbidden_tools`: tools that should never be used for the turn.
- `assertions`: final-answer, grounding, date, fallback, and memory checks.

JSON does not support comments, so keep comments in `.jsonc` examples or docs,
not in `eval_scenarios.json`.
