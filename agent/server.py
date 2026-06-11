"""
Agent HTTP Server — FastAPI app that Electron calls to trigger runs
and that pushes status back to the renderer via SSE.

Endpoints:
  POST /run           { task, start_url, mode, max_steps }
  GET  /status/stream → SSE stream of live step updates
  GET  /results       → list of past RunResult dicts
  GET  /health        → ollama/Claude status + running flag

Model selection via environment variables:
  OLLAMA_MODEL=qwen2.5vl:3b   (default, M2 16GB)
  OLLAMA_MODEL=qwen2.5vl:7b   (recommended, 24GB+ VRAM)
  OLLAMA_MODEL=qwen2.5vl:72b  (high accuracy, A100 class)
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from executor import AgentExecutor, ElectronBridge, RunResult
from vlm_client import OllamaVLMClient, ClaudeVLMClient


# ── Config ────────────────────────────────────────────────────────────────────
# Override any of these via environment variables:
#
#   OLLAMA_MODEL=qwen2.5vl:7b uvicorn server:app ...
#   OLLAMA_MODEL=qwen2.5vl:7b STEP_DELAY=0.1 uvicorn server:app ...

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ELECTRON_URL = os.getenv("ELECTRON_URL",  "http://localhost:7788")

# Step delay — 7B is faster per token, can afford shorter delay
# 3b on M2:      0.3s (inference already slow, no point waiting more)
# 7b on 24GB:    0.5s (faster inference, slightly longer wait for page settle)
# 7b+ on A100:   1.0s (very fast inference, more wait helps page rendering)
STEP_DELAY   = float(os.getenv("STEP_DELAY", "0.3"))

# Max steps — 7B is more capable, may need fewer steps
MAX_STEPS    = int(os.getenv("MAX_STEPS", "20"))


# ── Global state ──────────────────────────────────────────────────────────────

_status_queue: asyncio.Queue = asyncio.Queue()
_results: list[dict] = []
_running = False

local_client:  Optional[OllamaVLMClient]  = None
remote_client: Optional[ClaudeVLMClient]  = None
bridge:        Optional[ElectronBridge]   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_client, remote_client, bridge

    print(f"[server] model:      {OLLAMA_MODEL}")
    print(f"[server] step_delay: {STEP_DELAY}s")
    print(f"[server] max_steps:  {MAX_STEPS}")

    local_client = OllamaVLMClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    if ANTHROPIC_KEY:
        remote_client = ClaudeVLMClient(api_key=ANTHROPIC_KEY)
    bridge = ElectronBridge(base_url=ELECTRON_URL)

    yield

    local_client.close()
    if remote_client:
        remote_client.close()
    bridge.close()


app = FastAPI(title="GUI Agent Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    start_url: Optional[str] = None
    mode: str = "screenshot"
    max_steps: int = MAX_STEPS


class RunResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ollama_ok = local_client.is_available() if local_client else False
    return {
        "status":     "ok",
        "ollama":     ollama_ok,
        "model":      OLLAMA_MODEL,
        "claude":     bool(ANTHROPIC_KEY),
        "running":    _running,
        "step_delay": STEP_DELAY,
    }


@app.post("/run", response_model=RunResponse)
async def run_task(req: RunRequest):
    global _running

    if _running:
        return RunResponse(status="busy", message="Agent is already running a task.")

    _running = True
    asyncio.create_task(_run_agent(req))
    return RunResponse(status="started", message=f"Started: {req.task}")


async def _run_agent(req: RunRequest):
    global _running

    loop = asyncio.get_event_loop()

    def _sync_run():
        print(f"[agent] run started")
        print(f"[agent] task:      {req.task}")
        print(f"[agent] mode:      {req.mode}")
        print(f"[agent] model:     {OLLAMA_MODEL}")
        print(f"[agent] start_url: {req.start_url}")

        executor = AgentExecutor(
            bridge=bridge,
            local_client=local_client  if req.mode != "hybrid" else None,
            remote_client=remote_client if req.mode == "hybrid"  else None,
            mode=req.mode,
            max_steps=req.max_steps,
            step_delay_s=STEP_DELAY,
        )

        def _push(payload):
            asyncio.run_coroutine_threadsafe(
                _status_queue.put(payload), loop
            )
        executor.bridge.push_status = _push
        return executor.run(task=req.task, start_url=req.start_url)

    try:
        result: RunResult = await loop.run_in_executor(None, _sync_run)
        print(f"[agent] run finished: success={result.success}")
        _results.append(result.to_dict())
        await _status_queue.put({"type": "done", "result": result.to_dict()})
    except Exception as e:
        print(f"[agent] error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await _status_queue.put({"type": "error", "message": str(e)})
    finally:
        _running = False


@app.get("/status/stream")
async def status_stream():
    async def _generator():
        while True:
            try:
                payload = await asyncio.wait_for(_status_queue.get(), timeout=30.0)
                yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.TimeoutError:
                yield "data: {\"type\":\"ping\"}\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")


@app.get("/results")
async def get_results():
    return {"results": _results}
