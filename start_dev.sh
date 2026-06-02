#!/usr/bin/env bash
# start_dev.sh — starts all three processes for local development
# Usage:
#   ./start_dev.sh                     # screenshot mode, local Ollama
#   GEMINI_API_KEY=sk-... ./start_dev.sh  # enables hybrid mode
#   OLLAMA_MODEL=avil/UI-TARS ./start_dev.sh # use a different model

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Check prerequisites ──────────────────────────────────────────────────────
command -v ollama  >/dev/null 2>&1 || echo "! ollama not found — screenshot mode will use Gemini fallback"
command -v python3 >/dev/null 2>&1 || { echo "x python3 required"; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "x node required"; exit 1; }

echo "───────────────────────────────────────────────"
echo "  GUI Agent Dev Startup"
echo "  OLLAMA_MODEL : ${OLLAMA_MODEL:-0000/ui-tars-1.5-7b}"
echo "  GEMINI API   : ${GEMINI_API_KEY:+set}"
echo "───────────────────────────────────────────────"

# ── 1. Pull model if needed ──────────────────────────────────────────────────
if command -v ollama >/dev/null 2>&1; then
  MODEL="${OLLAMA_MODEL:-qwen2.5vl:3b}"
  if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "Pulling $MODEL via ollama…"
    ollama pull "$MODEL"
  fi
fi

# ── 2. Python agent server ───────────────────────────────────────────────────
echo "Starting Python agent server on :8000…"
cd "$ROOT/agent"
pip install -q -r requirements.txt
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
