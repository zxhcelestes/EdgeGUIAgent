"""
Agent HTTP Server — FastAPI app that Electron calls to trigger runs
and that pushes status back to the renderer via SSE.

Endpoints:
  POST /run          { task, start_url, mode }  → starts agent run
  GET  /status/stream                          → SSE stream of live updates
  GET  /results                                → list of past RunResult dicts
  GET  /health
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
from vlm_client import OllamaVLMClient, GeminiVLMClient


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
GEMINI_KEY= os.getenv("GEMINI_API_KEY", "")
ELECTRON_URL = os.getenv("ELECTRON_URL", "http://localhost:7788")


# ── Global state ─────────────────────────────────────────────────────────────

_status_queue: asyncio.Queue = asyncio.Queue()
_results: list[dict] = []
_running = False

local_client: Optional[OllamaVLMClient] = None
remote_client: Optional[GeminiVLMClient] = None
bridge: Optional[ElectronBridge] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_client, remote_client, bridge

    local_client = OllamaVLMClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    if GEMINI_KEY:
        remote_client = GeminiVLMClient(api_key=GEMINI_KEY)
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


# ── Models ────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    start_url: Optional[str] = None
    mode: str = "screenshot"    # "screenshot" | "hybrid" | "dom"
    max_steps: int = 20


class RunResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ollama_ok = local_client.is_available() if local_client else False
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "gemini": bool(GEMINI_KEY),
        "running": _running,
    }


@app.post("/run", response_model=RunResponse)
async def run_task(req: RunRequest):
    global _running

    if _running:
        return RunResponse(status="busy", message="Agent is already running a task.")

    _running = True

    # Run in background so we can return immediately
    asyncio.create_task(_run_agent(req))
    return RunResponse(status="started", message=f"Started: {req.task}")


async def _run_agent(req: RunRequest):
    global _running

    loop = asyncio.get_event_loop()

    def _sync_run():
        print(f"[agent] starting task: {req.task}")
        print(f"[agent] mode: {req.mode}, start_url: {req.start_url}")
        executor = AgentExecutor(
            bridge=bridge,
            local_client=local_client if req.mode != "hybrid" else None,
            remote_client=remote_client if req.mode in ("hybrid",) else None,
            mode=req.mode,
            max_steps=req.max_steps,
            step_delay_s=0.3,
        )
        def _push(payload):
            print(f"[agent] step update: {payload.get('step')} action={payload.get('action', {}).get('type')}")
            asyncio.run_coroutine_threadsafe(
                _status_queue.put(payload), loop
            )
        executor.bridge.push_status = _push
        return executor.run(task=req.task, start_url=req.start_url)

    try:
        print("[agent] run started")
        result: RunResult = await loop.run_in_executor(None, _sync_run)
        print(f"[agent] run finished: success={result.success}")
        _results.append(result.to_dict())
        await _status_queue.put({"type": "done", "result": result.to_dict()})
    except Exception as e:
        print(f"[agent] error: {e}")
        import traceback
        traceback.print_exc()
        _results.append({
            "task": req.task,
            "mode": req.mode,
            "success": False,
            "step_count": 0,
            "total_time_s": 0.0,
            "avg_latency_s": 0.0,
            "failure_reason": str(e),
        })
        err = {"type": "error", "message": str(e)}
        await _status_queue.put(err)
    finally:
        _running = False

@app.get("/status/stream")
async def status_stream():
    """Server-Sent Events stream of agent step updates."""
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
