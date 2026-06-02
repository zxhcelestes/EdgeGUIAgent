"""
AgentExecutor — runs the perceive → plan → act loop.
Communicates with the Electron renderer via HTTP (localhost:7788).
"""

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from vlm_client import (
    AgentAction,
    ActionType,
    OllamaVLMClient,
    GeminiVLMClient,
    build_dom_context,
)


@dataclass
class StepRecord:
    step: int
    thought: str
    action: dict
    latency_s: float
    screenshot_b64: Optional[str] = None
    dom_element_count: int = 0
    success: Optional[bool] = None


@dataclass
class RunResult:
    task: str
    mode: str                        # "screenshot" | "hybrid" | "dom"
    success: bool
    steps: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "mode": self.mode,
            "success": self.success,
            "step_count": len(self.steps),
            "total_time_s": round(self.total_time_s, 3),
            "avg_latency_s": round(
                sum(s.latency_s for s in self.steps) / max(len(self.steps), 1), 3
            ),
            "failure_reason": self.failure_reason,
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "action": s.action,
                    "latency_s": round(s.latency_s, 3),
                    "dom_element_count": s.dom_element_count,
                }
                for s in self.steps
            ],
        }


# ── Electron bridge ────────────────────────────────────────────────────────────

ELECTRON_BASE = "http://localhost:7788"


class ElectronBridge:
    """Thin HTTP client that talks to the Electron main process."""

    def __init__(self, base_url: str = ELECTRON_BASE):
        self._client = httpx.Client(
            timeout=15.0,
            transport=httpx.HTTPTransport(proxy=None),  # localhost, no proxy needed
        )
        self.base_url = base_url

    def get_screenshot(self) -> tuple[bytes, int, int]:
        """Returns (png_bytes, width, height)."""
        r = self._client.get(f"{self.base_url}/screenshot")
        r.raise_for_status()
        data = r.json()
        png = base64.b64decode(data["image"])
        return png, data["width"], data["height"]

    def get_dom_elements(self) -> list[dict]:
        r = self._client.get(f"{self.base_url}/dom")
        r.raise_for_status()
        return r.json().get("elements", [])

    def get_current_url(self) -> str:
        try:
            r = self._client.get(f"{self.base_url}/current-url", timeout=3.0)
            return r.json().get("url", "")
        except Exception:
            return ""

    def execute_action(self, action: AgentAction) -> dict:
        r = self._client.post(
            f"{self.base_url}/action",
            json=action.to_dict(),
        )
        r.raise_for_status()
        return r.json()

    def navigate(self, url: str):
        r = self._client.post(f"{self.base_url}/navigate", json={"url": url})
        r.raise_for_status()

    def push_status(self, payload: dict):
        """Send live status update to the renderer UI."""
        try:
            self._client.post(f"{self.base_url}/status", json=payload, timeout=2.0)
        except Exception:
            pass   # non-critical

    def close(self):
        self._client.close()


# ── Success detection ─────────────────────────────────────────────────────────

def _check_success(task: str, url: str) -> bool:
    """
    Heuristic URL-based success detection.
    Returns True when the current URL strongly suggests the task is complete.
    """
    task_lower = task.lower()
    url_lower = url.lower()

    # Search tasks: results page loaded
    if any(kw in task_lower for kw in ["search for", "search on", "find"]):
        if any(kw in url_lower for kw in ["search", "q=", "results", "?s=", "query"]):
            return True

    # GitHub trending
    if "trending" in task_lower and "github.com/trending" in url_lower:
        return True

    # GitHub repo page
    if "github" in task_lower and "open issues" in task_lower:
        if "github.com/" in url_lower and "/issues" not in url_lower:
            # landed on a repo page (not issues list)
            parts = url_lower.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                return True

    # Hacker News story
    if "hacker news" in task_lower and "item?id=" in url_lower:
        return True

    # Wikipedia article
    if "wikipedia" in task_lower and "wikipedia.org/wiki/" in url_lower:
        return True

    # Excalidraw
    if "excalidraw" in task_lower and "excalidraw.com" in url_lower:
        return True

    return False


# ── Executor ──────────────────────────────────────────────────────────────────

class AgentExecutor:
    def __init__(
        self,
        bridge: ElectronBridge,
        local_client: Optional[OllamaVLMClient] = None,
        remote_client: Optional[GeminiVLMClient] = None,
        mode: str = "screenshot",      # "screenshot" | "hybrid" | "dom"
        max_steps: int = 20,
        step_delay_s: float = 1.0,
    ):
        self.bridge = bridge
        self.local_client = local_client
        self.remote_client = remote_client
        self.mode = mode
        self.max_steps = max_steps
        self.step_delay_s = step_delay_s

    def _get_vlm_client(self):
        if self.mode == "hybrid":
            return self.remote_client or self.local_client
        return self.local_client or self.remote_client

    def run(self, task: str, start_url: Optional[str] = None) -> RunResult:
        result = RunResult(task=task, mode=self.mode, success=False)
        history: list[str] = []
        t_start = time.perf_counter()

        if start_url:
            self.bridge.navigate(start_url)
            time.sleep(2.0)   # wait for initial page load

        client = self._get_vlm_client()
        if client is None:
            result.failure_reason = "No VLM client available"
            return result

        for step_num in range(1, self.max_steps + 1):
            # ── Perceive ──
            for _ in range(3):
                screenshot, w, h = self.bridge.get_screenshot()
                if w > 0 and h > 0:
                    break
                print(f"[executor] screenshot size 0, retrying...")
                time.sleep(1.0)

            dom_elements: list[dict] = []
            dom_context: Optional[str] = None

            if self.mode in ("hybrid", "dom"):
                dom_elements = self.bridge.get_dom_elements()
                # print(f"[executor] dom elements: {[e.get('text') for e in dom_elements]}")
                dom_context = build_dom_context(dom_elements, w, h)
                # print(f"[executor] dom context sent to model:\n{dom_context}")  # 加这行

            # ── URL-based success check (before planning) ──
            current_url = self.bridge.get_current_url()
            print(f"[executor] step {step_num} url: {current_url}")
            if _check_success(task, current_url):
                print(f"[executor] URL-based success detected: {current_url}")
                result.success = True
                result.total_time_s = time.perf_counter() - t_start
                self.bridge.push_status({
                    "type": "done",
                    "result": result.to_dict(),
                })
                return result

            # ── Plan ──
            action, latency = client.get_action(
                screenshot_bytes=screenshot,
                task=task,
                history=history,
                dom_context=dom_context,
                screen_w=w,
                screen_h=h,
            )

            if action is None:
                action = AgentAction(type=ActionType.FAIL, thought="no action returned")

            step = StepRecord(
                step=step_num,
                thought=action.thought or "",
                action=action.to_dict(),
                latency_s=latency,
                screenshot_b64=base64.b64encode(screenshot).decode(),
                dom_element_count=len(dom_elements),
            )
            result.steps.append(step)

            # Push live update to renderer
            self.bridge.push_status({
                "step": step_num,
                "thought": action.thought,
                "action": action.to_dict(),
                "latency_s": round(latency, 3),
                "screenshot": step.screenshot_b64,
            })

            # ── Terminal states ──
            if action.type == ActionType.DONE:
                result.success = True
                break
            if action.type == ActionType.FAIL:
                result.failure_reason = action.thought or "model returned fail"
                break

            # ── Loop detection ──
            # if len(history) >= 3:
            #     last_3 = [s.split(":")[0] for s in history[-3:]]
            #     if len(set(last_3)) == 1 and last_3[0] == action.type.value:
            #         print(f"[executor] loop detected: repeated {action.type.value} 3+ times")
            #         result.failure_reason = f"stuck in loop: repeated {action.type.value}"
            #         break

            # ── Act ──
            try:
                self.bridge.execute_action(action)

                if action.type == ActionType.TYPE and action.text:
                    enter_action = AgentAction(type=ActionType.KEY, key="Enter")
                    self.bridge.execute_action(enter_action)
                    print(f"[executor] auto Enter after type")
            except httpx.HTTPError as e:
                result.failure_reason = f"action execution error: {e}"
                break

            history.append(f"{action.type.value}: x={action.x}, y={action.y}, text={action.text}")

            # Extra wait after navigation actions to allow page to load
            # if action.type in (ActionType.KEY, ActionType.CLICK, ActionType.NAVIGATE):
            #     time.sleep(2.0)
            time.sleep(self.step_delay_s)

        else:
            result.failure_reason = f"max steps ({self.max_steps}) reached"

        result.total_time_s = time.perf_counter() - t_start
        return result