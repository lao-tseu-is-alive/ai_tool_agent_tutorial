#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${1:-"$ROOT_DIR/eval_runs/latest"}"

cd "$ROOT_DIR"

trap 'echo "❌ Test run failed. Check the command output above."' ERR

echo "🔎 Compile-checking Python modules"
uv run python -m py_compile py_tool_agent/*.py py_tool_agent/eval/*.py

echo "✅ Compile check passed"

echo "🧪 Running deterministic behavioral regression tests"
uv run python -m unittest tests.test_eval_scenarios

echo "✅ Regression tests passed"

echo "📝 Writing deterministic eval report to $REPORT_DIR"
uv run python -m py_tool_agent.eval.runner \
  --scenarios tests/eval_scenarios.json \
  --out "$REPORT_DIR"

echo "✅ Eval report written"
echo "✅ Done"
