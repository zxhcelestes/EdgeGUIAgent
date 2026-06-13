#!/usr/bin/env bash
# start_dev.sh — starts all processes for local development
# Usage:
#   ./start_dev.sh                          # screenshot/dom mode, local Ollama only
#   ANTHROPIC_API_KEY=sk-ant-... ./start_dev.sh  # enables Claude hybrid mode
#   OLLAMA_MODEL=qwen2.5vl:3b ./start_dev.sh     # use a different local model

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Check prerequisites ──────────────────────────────────────────────────────
command -v ollama  >/dev/null 2>&1 || echo "! ollama not found — local screenshot/dom modes will not work"
command -v python3 >/dev/null 2>&1 || { echo "x python3 required"; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "x node required"; exit 1; }

echo "───────────────────────────────────────────────"
echo "  GUI Agent Dev Startup"
echo "  OLLAMA_MODEL   : ${OLLAMA_MODEL:-qwen2.5vl:7b}"
echo "  ANTHROPIC API  : ${ANTHROPIC_API_KEY:+set (hybrid mode available)}"
echo "  STEP_DELAY     : ${STEP_DELAY:-0.5}"
echo "───────────────────────────────────────────────"

# ── 1. Pull model if needed ──────────────────────────────────────────────────
if command -v ollama >/dev/null 2>&1; then
  MODEL="${OLLAMA_MODEL:-qwen2.5vl:7b}"
  if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "Pulling $MODEL via ollama…"
    ollama pull "$MODEL"
  fi
fi

# ── 2. Python agent server ───────────────────────────────────────────────────
echo "Starting Python agent server on :8000…"
cd "$ROOT/agent"
pip install -q -r requirements.txt
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5vl:7b}" \
STEP_DELAY="${STEP_DELAY:-0.5}" \
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
uvicorn server:app --host 127.0.0.1 --port 8000 --reload &
AGENT_PID=$!

# ── 3. Electron app ──────────────────────────────────────────────────────────
echo "Starting Electron app…"
cd "$ROOT/electron-app"
npm install --silent
ELECTRON_DEV=1 npx electron . &
ELECTRON_PID=$!

echo ""
echo "✓ Agent server PID: $AGENT_PID"
echo "✓ Electron PID:     $ELECTRON_PID"
echo ""
echo "Press Ctrl+C to stop all processes."

trap "echo 'Shutting down…'; kill $AGENT_PID $ELECTRON_PID 2>/dev/null" INT TERM
wait
