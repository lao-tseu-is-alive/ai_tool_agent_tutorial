# Tests

This directory contains deterministic behavioral tests for the tool agent.

The shortest path is:

```bash
./scripts/test.sh
```

That script runs:

1. Python compile checks.
2. The deterministic unittest suite.
3. The eval runner, which writes `results.json` and `summary.md`.

By default the report is written to a timestamped folder:

```text
eval_runs/2026-05-08T13-45-12Z/
```

The script also refreshes `eval_runs/latest/` for quick local inspection.

The report includes metadata such as:

- creation time
- model label
- provider-facing model name
- scenario path
- temperature, timeout, context size, and API base when configured

For this deterministic phase, the model label defaults to
`deterministic-fixture`. You can override only the label:

```bash
EVAL_MODEL_LABEL=my-local-baseline ./scripts/test.sh
```

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

The real scenario file starts with `$schema` so schema-aware editors can offer
completion and validation automatically:

```json
{
  "$schema": "./eval_scenarios.schema.json",
  "scenarios": []
}
```

Each scenario can validate:

- which tools were used
- which tools must not be used
- final answer text
- grounding against tool results
- date calculations from `get_current_time`
- assistant/tool message pairing in agent memory

The current scenario protects the date-chaining and directory-listing behavior
seen while testing Ollama models.

## Deterministic vs Live

The same scenario file is used in two ways:

- `./scripts/test.sh` uses `llm_responses` and a fake LLM. This tests the agent
  contract, fallbacks, tracing, and assertions.
- `./scripts/eval_live.sh` ignores `llm_responses` and calls the selected real
  model. This tests actual model behavior.

Live evals run tools from a generated workspace under `eval_runs/live/workspace`.
That workspace is recreated before every live run so directory-listing tests do
not depend on mutable project files such as the real `backend.py`.

The generated workspace is configured by the top-level `workspace` object in
`eval_scenarios.json`. Relative modification dates such as `yesterday` are
resolved from the local time when the workspace is prepared unless
`workspace.reference_time` is set explicitly for reproducible debugging:

```json
{
  "workspace": {
    "files": [
      {
        "path": "backend.py",
        "content": "print('example')\n",
        "size_bytes": 128,
        "modified_at": {
          "date": "yesterday",
          "time": "16:15:00"
        }
      }
    ]
  }
}
```

For live evals, the important fields are:

- `input`
- `expected_tools`
- `forbidden_tools`
- `assertions`
- `live_assertions` when live behavior needs a slightly different expectation

For deterministic evals, `llm_responses` is also required when the turn reaches
the fake LLM.

## Adding A Simple Scenario

Add a new object to the top-level `scenarios` array in `eval_scenarios.json`:

```json
{
  "$schema": "./eval_scenarios.schema.json",
  "scenarios": [
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
  ]
}
```

The important sections are:

- `fixtures`: fixed tool outputs for deterministic tests.
- `scenarios`: scenario objects run by the eval runner.
- `turns`: user messages run sequentially against the same agent.
- `llm_responses`: fake model messages consumed by the agent.
- `expected_tools`: exact ordered tool calls expected after adapter normalization.
- `forbidden_tools`: tools that should never be used for the turn.
- `assertions`: final-answer, grounding, date, fallback, and memory checks.

JSON does not support comments, so keep comments in `.jsonc` examples or docs,
not in `eval_scenarios.json`.
