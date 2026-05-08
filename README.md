# AI Tool Agent Tutorial

A small, practical LLM agent for learning how tool use really works with local
Ollama models.

This project is intentionally compact, but it is no longer a toy script. It now
has a registry-based tool system, strict Pydantic validation, model adapters for
messy local-model behavior, readable console diagnostics, and deterministic
fallbacks when a model ignores a valid tool result.

## Why This Exists

Local models do not all behave the same way with tools.

Some models return clean `tool_calls`. Some write JSON-looking tool calls inside
plain text. Some call the tool correctly, then hallucinate while summarizing the
result. This project shows how to build an agent that stays useful anyway.

The main rule is simple:

> The model may decide, but tools and validation define reality.

## Features

- Interactive CLI agent powered by LiteLLM.
- Works with local Ollama models through `ollama_chat/...` routing.
- Registry pattern for adding tools in one central place.
- Strict Pydantic validation for all tool arguments.
- Tool schemas generated from the same Pydantic models used at runtime.
- Model adapter layer for native tool calls, text-embedded JSON tool calls, and inferred tool use.
- Deterministic fallbacks for time/date and directory-listing answers.
- Colorized console diagnostics showing tool access, model messages, tool calls, and tool results.
- Safe directory listing implemented with Python APIs instead of arbitrary shell input.

## Current Tools

The default registry includes:

| Tool | Purpose |
| --- | --- |
| `get_current_time` | Returns the current local date and time. |
| `add_numbers` | Adds two validated numeric arguments. |
| `list_current_directory` | Lists the current directory with `ls -al` style metadata. |

## Project Layout

```text
py_tool_agent/
  agent.py            # Orchestrates conversation, tool execution, and fallbacks
  llm.py              # LiteLLM wrapper and Ollama model normalization
  main.py             # Interactive CLI entrypoint
  model_adapters.py   # Normalizes model-specific tool behavior
  tool_registry.py    # Tool metadata, schemas, validation, execution, fallbacks
  tools.py            # Concrete tool functions and default registry
  tracing.py          # Per-turn trace capture for evals and debugging
  eval/
    assertions.py     # Deterministic behavioral assertions
    fixtures.py       # Scripted LLM and deterministic tool fixtures
    runner.py         # Scenario runner and report writer
tests/
  README.md           # How to run deterministic tests and evals
  eval_scenarios.example.jsonc
  eval_scenarios.json # Behavioral regression scenarios
  eval_scenarios.schema.json
  test_eval_scenarios.py
scripts/
  test.sh             # One-command compile, test, and eval report runner
```

## Quickstart

Install dependencies:

```bash
uv sync
```

Start Ollama and make sure your model is available:

```bash
ollama pull gemma4:26b
ollama pull qwen2.5-coder:latest
ollama pull qwen3
```

Choose a model:

```bash
export AGENT_MODEL=ollama/gemma4:26b
```

Run the agent:

```bash
uv run -m py_tool_agent.main
```

Example:

```text
>>> give me the current date and calculate the date of tomorrow
The current date is 2026-05-08. Tomorrow's date is 2026-05-09.

>>> any python files from yesterday in current folder
Matching files in the current directory: backend.py.

>>> what's the size of this file
The size of backend.py is 1121 bytes.
```

## Configuration

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_MODEL` | `ollama/qwen3` | Model name passed to LiteLLM. |
| `AGENT_TEMPERATURE` | `0.2` | Sampling temperature. |
| `AGENT_REQUEST_TIMEOUT` | `20` | Request timeout in seconds. |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama server URL. |

For Ollama models, `llm.py` normalizes:

```text
ollama/model-name -> ollama_chat/model-name
```

That keeps chat/tool behavior aligned with LiteLLM's Ollama chat provider.

## How Tool Use Works

The agent flow is:

1. Add the user message to conversation memory.
2. Ask the model adapter whether tools should be exposed.
3. Send the model the relevant tool schemas.
4. Normalize tool calls:
   - native provider `tool_calls`
   - JSON tool calls embedded in text
   - inferred obvious tool calls for weaker local models
5. Validate arguments with Pydantic.
6. Execute only registered Python functions.
7. Ask the model for a final answer.
8. Override bad final answers with deterministic fallbacks when needed.

The LLM never executes arbitrary code. It can only request registered tools.

## Adding a Tool

Add the Python function and argument model in `tools.py`:

```python
class MultiplyArguments(ToolArguments):
    a: float
    b: float


def multiply_numbers(a: float, b: float) -> float:
    return a * b
```

Register it once in `DEFAULT_TOOL_REGISTRY`:

```python
ToolSpec(
    name="multiply_numbers",
    description="multiply two numbers",
    function=multiply_numbers,
    args_model=MultiplyArguments,
    intent_keywords=("multiply", "product"),
)
```

The registry then provides:

- the callable function
- the JSON schema
- strict validation
- intent routing
- execution
- optional fallback behavior

## Model Compatibility Notes

This project was shaped by testing multiple Ollama models:

- `qwen3`: generally better at native tool calls.
- `gemma4:26b`: can call tools but may hallucinate while summarizing results.
- `qwen2.5-coder:latest`: may emit JSON tool calls in plain text or ask permission instead of calling tools.

The adapter layer exists because those differences should not leak into the
agent loop or individual tools.

## Deterministic Behavioral Evals

The project includes a first evaluation harness for checking agent behavior
without relying on an external judge model.

The recommended test path is:

```bash
./scripts/test.sh
```

It compile-checks the package, runs the deterministic unittest suite, and writes
an eval report to `eval_runs/latest`.

The output uses `✅` for passed steps and `❌` for failures.

Scenarios live in `tests/eval_scenarios.json`. The supporting files are:

- `tests/eval_scenarios.example.jsonc`: commented example for learning the format
- `tests/eval_scenarios.schema.json`: JSON schema for editor validation and documentation
- `tests/README.md`: step-by-step notes for adding a scenario

Each turn can define:

- scripted model responses
- fixed tool fixtures, such as current time and directory listing output
- expected tools, forbidden tools, and final-answer assertions
- memory validation for assistant tool calls and matching `role: tool` messages
- groundedness checks for file names, file sizes, date answers, and access-denial hallucinations

Run only the regression test:

```bash
uv run python -m unittest tests.test_eval_scenarios
```

Important: `unittest` expects a dotted Python module name. Do not use a slash
path here:

```bash
uv run python -m unittest tests/test_eval_scenarios
```

Generate a readable JSON and Markdown report:

```bash
uv run python -m py_tool_agent.eval.runner \
  --scenarios tests/eval_scenarios.json \
  --out eval_runs/latest
```

Current phase-1 scenario coverage:

- greeting should not expose tools
- date request should call `get_current_time`
- wrong model date summaries are overridden by deterministic fallback
- yesterday chaining remains grounded in the time tool result
- Python files from yesterday are filtered from directory metadata
- file-size follow-up stays grounded in the same directory listing, preventing regressions like `1234` instead of `1121`

The next natural step is a cross-model runner that executes the same scenario
prompts against real Ollama models and writes comparative reliability reports.

## Design Principles

- Keep tool definitions centralized.
- Treat model-generated arguments as untrusted input.
- Prefer deterministic code for facts derived from tool results.
- Keep provider/model quirks in adapters.
- Keep the agent loop readable.
- Never let the model invent tool results.

## Development Checks

Compile-check the package:

```bash
python -m py_compile py_tool_agent/*.py py_tool_agent/eval/*.py
```

Check that every class/function has a docstring:

```bash
python - <<'PY'
import ast
from pathlib import Path

missing = []
for path in Path("py_tool_agent").glob("*.py"):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                missing.append((str(path), node.lineno, node.name))

for path, line, name in missing:
    print(f"{path}:{line} {name}")
print(f"missing={len(missing)}")
PY
```

## Cognitive Loop

The project still follows the original learning loop:

1. Perceive the user request.
2. Update short-term conversation memory.
3. Decide whether tools are needed.
4. Act through validated registered tools.
5. Ground the final answer in tool results.
6. Loop until a reliable answer is produced.
