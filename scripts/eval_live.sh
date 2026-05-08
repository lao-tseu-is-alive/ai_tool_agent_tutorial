#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"

cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${1:-"${AGENT_MODEL:-ollama/qwen3}"}"
SAFE_MODEL="${MODEL//\//_}"
SAFE_MODEL="${SAFE_MODEL//:/_}"
REPORT_DIR="${2:-"$ROOT_DIR/eval_runs/live/$RUN_ID-$SAFE_MODEL"}"
LATEST_DIR="$ROOT_DIR/eval_runs/live/latest"

trap 'echo "❌ Live eval failed. Check the command output above."' ERR

echo "🔎 Compile-checking Python modules"
uv run python -m py_compile py_tool_agent/*.py py_tool_agent/eval/*.py
echo "✅ Compile check passed"

echo "🧪 Running live eval with model: $MODEL"
uv run python -m py_tool_agent.eval.live_runner \
  --scenarios tests/eval_scenarios.json \
  --model "$MODEL" \
  --out "$REPORT_DIR"

echo "✅ Live eval report written to $REPORT_DIR"

mkdir -p "$LATEST_DIR"
cp "$REPORT_DIR/results.json" "$LATEST_DIR/results.json"
cp "$REPORT_DIR/summary.md" "$LATEST_DIR/summary.md"
echo "✅ Latest live report updated at $LATEST_DIR"
echo "✅ Done"
