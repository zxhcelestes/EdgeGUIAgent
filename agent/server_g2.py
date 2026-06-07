"""
Agent HTTP Server v2 — Planner + GUI-G2 Grounder Architecture

Identical to server.py except it uses vlm_client_g2.OllamaVLMClient,
which adds GUI-G2 grounding for click actions on CUDA environments.
On macOS / no CUDA, automatically falls back to planner-only mode.

Usage:
    # Standard (planner-only on macOS, planner+grounder on CUDA)
    uvicorn server_g2:app --host 127.0.0.1 --port 8000 --reload

    # With 7B planner (24GB+ VRAM)
    OLLAMA_MODEL=qwen2.5vl:7b uvicorn server_g2:app ...

    # Custom grounder path
    GROUNDER_MODEL_PATH=/path/to/GUI-G2-3B uvicorn server_g2:app ...

    # Download GUI-G2-3B (CUDA environments only):
    huggingface-cli download inclusionAI/GUI-G2-3B --local-dir ./models/GUI-G2-3B
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
from vlm_client_g2 import OllamaVLMClient, GeminiVLMClient


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL",        "qwen2.5vl:3b")
OLLAMA_URL          = os.getenv("OLLAMA_URL",          "http://localhost:11434")
GEMINI_KEY          = os.getenv("GEMINI_API_KEY",      "")
ELECTRON_URL        = os.getenv("ELECTRON_URL",        "http://localhost:7788")
GROUNDER_MODEL_PATH = os.getenv("GROUNDER_MODEL_PATH", "./models/GUI-G2-3B")
STEP_DELAY          = float(os.getenv("STEP_DELAY",    "0.3"))
MAX_STEPS           = int(os.getenv("MAX_STEPS",       "20"))


# ── Global state ──────────────────────────────────────────────────────────────

_status_queue: asyncio.Queue = asyncio.Queue()
_results: list[dict] = []
_running = False

local_client:  Optional[OllamaVLMClient] = None
remote_client: Optional[GeminiVLMClient] = None
bridge:        Optional[ElectronBridge]  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_client, remote_client, bridge

    print(f"[server_g2] planner model:    {OLLAMA_MODEL}")
    print(f"[server_g2] grounder path:    {GROUNDER_MODEL_PATH}")
    print(f"[server_g2] step_delay:       {STEP_DELAY}s")
    print(f"[server_g2] max_steps:        {MAX_STEPS}")

    local_client = OllamaVLMClient(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        grounder_model_path=GROUNDER_MODEL_PATH,
    )
    if GEMINI_KEY:
        remote_client = GeminiVLMClient(api_key=GEMINI_KEY)
    bridge = ElectronBridge(base_url=ELECTRON_URL)

    # Pre-load GUI-G2 only if CUDA is available
    if local_client._grounder_available:
        print("[startup] Pre-loading GUI-G2 model on CUDA...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, local_client._grounder._ensure_loaded)
        print("[startup] GUI-G2 ready")

    yield

    local_client.close()
    if remote_client:
        remote_client.close()
    bridge.close()


app = FastAPI(title="GUI Agent Server v2 (Planner+Grounder)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

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
    ollama_ok    = local_client.is_available() if local_client else False
    grounder_ok  = local_client._grounder_available if local_client else False
    return {
        "status":     "ok",
        "ollama":     ollama_ok,
        "model":      OLLAMA_MODEL,
        "grounder":   grounder_ok,
        "gemini":     bool(GEMINI_KEY),
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
        print(f"[agent] grounder:  {local_client._grounder_available}")
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
            asyncio.run_coroutine_threadsafe(_status_queue.put(payload), loop)
        executor.bridge.push_status = _push
        return executor.run(task=req.task, start_url=req.start_url)

    try:
        result: RunResult = await loop.run_in_executor(None, _sync_run)
        print(f"[agent] run finished: success={result.success}")
        _results.append(result.to_dict())
        await _status_queue.put({"type": "done", "result": result.to_dict()})
    except Exception as e:
        print(f"[agent] error: {e}")
        import traceback; traceback.print_exc()
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
