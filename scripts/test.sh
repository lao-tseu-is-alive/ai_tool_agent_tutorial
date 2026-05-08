#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
REPORT_DIR="${1:-"$ROOT_DIR/eval_runs/$RUN_ID"}"
LATEST_DIR="$ROOT_DIR/eval_runs/latest"

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
  --model "${EVAL_MODEL_LABEL:-deterministic-fixture}" \
  --out "$REPORT_DIR"

echo "✅ Eval report written"

if [[ "$REPORT_DIR" != "$LATEST_DIR" ]]; then
  mkdir -p "$LATEST_DIR"
  cp "$REPORT_DIR/results.json" "$LATEST_DIR/results.json"
  cp "$REPORT_DIR/summary.md" "$LATEST_DIR/summary.md"
  echo "✅ Latest report updated at $LATEST_DIR"
fi

echo "✅ Done"
