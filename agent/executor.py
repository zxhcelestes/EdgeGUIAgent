"""
AgentExecutor — runs the perceive → plan → act loop.
Communicates with the Electron renderer via HTTP (localhost:7788).
"""

import base64
import re
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
    mode: str
    success: bool
    steps: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    failure_reason: Optional[str] = None
    final_answer: Optional[str] = None  # extracted content from done action text field

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
            "final_answer": self.final_answer,
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
            transport=httpx.HTTPTransport(proxy=None),
        )
        self.base_url = base_url

    def get_screenshot(self) -> tuple[bytes, int, int]:
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
        try:
            self._client.post(f"{self.base_url}/status", json=payload, timeout=2.0)
        except Exception:
            pass

    def close(self):
        self._client.close()


# ── Success evaluator ─────────────────────────────────────────────────────────

def _evaluate_completion(
    task: str,
    screenshot_bytes: bytes,
    current_url: str,
    client,
    screen_w: int,
    screen_h: int,
) -> tuple[bool, str]:
    """
    Calls client.evaluate() to verify task completion.
    Called when:
      (a) the planner outputs 'done' — verifies before accepting
      (b) URL-based heuristic detects likely completion — confirms before marking success
    """
    if hasattr(client, "evaluate"):
        return client.evaluate(screenshot_bytes, task, current_url, screen_w, screen_h)
    return False, "client does not support evaluation"


# ── URL-based completion heuristics ──────────────────────────────────────────

def _url_suggests_completion(task: str, url: str) -> bool:
    """
    Lightweight check: does the current URL strongly suggest the task is done?
    Used as a trigger for the evaluator — does NOT directly mark success.
    """
    task_lower = task.lower()
    url_lower  = url.lower()

    # Search tasks
    if any(kw in task_lower for kw in ["search for", "search on", "search github"]):
        if any(kw in url_lower for kw in ["q=", "search", "results"]):
            return True

    # GitHub Trending
    if "trending" in task_lower and "github.com/trending" in url_lower:
        return True

    # Wikipedia cross-page: landed on Attention article
    if "wikipedia" in task_lower and "wikipedia.org/wiki/" in url_lower:
        if "attention" in url_lower or "mechanism" in url_lower:
            return True

    # Hacker News story page
    if "hacker news" in task_lower and "item?id=" in url_lower:
        return True

    # HuggingFace model page
    if "huggingface" in task_lower and "download" in task_lower:
        if "huggingface.co/" in url_lower and "/models" not in url_lower and "search" not in url_lower:
            return True

    # GitHub repo page (for repo inspection tasks)
    if "github" in task_lower and "readme" in task_lower:
        if "github.com/" in url_lower and "/search" not in url_lower:
            parts = url_lower.replace("https://github.com/", "").split("/")
            if len(parts) >= 2 and parts[0] and parts[1]:
                return True

    return False


# ── Loop detection helpers ────────────────────────────────────────────────────

def _extract_coords(history_entry: str) -> Optional[tuple[float, float]]:
    """Extract (x, y) from a history entry string."""
    m = re.search(r"x=([\d.]+),\s*y=([\d.]+)", history_entry)
    if m:
        try:
            return round(float(m.group(1)), 2), round(float(m.group(2)), 2)
        except ValueError:
            pass
    return None


def _is_coord_loop(history: list[str], current_action: AgentAction) -> bool:
    """
    Returns True if the last 3 history entries + current action all have
    the same (x, y) coordinates — same spot clicked repeatedly with no effect.
    """
    if len(history) < 3:
        return False
    current_coords = (
        round(current_action.x or 0.0, 2),
        round(current_action.y or 0.0, 2),
    )
    if current_coords == (0.0, 0.0):
        return False
    recent_coords = [_extract_coords(h) for h in history[-3:]]
    if None in recent_coords:
        return False
    return all(c == current_coords for c in recent_coords)


# ── Executor ──────────────────────────────────────────────────────────────────

class AgentExecutor:
    def __init__(
        self,
        bridge: ElectronBridge,
        local_client: Optional[OllamaVLMClient] = None,
        remote_client: Optional[GeminiVLMClient] = None,
        mode: str = "screenshot",
        max_steps: int = 20,
        step_delay_s: float = 0.3,
    ):
        self.bridge        = bridge
        self.local_client  = local_client
        self.remote_client = remote_client
        self.mode          = mode
        self.max_steps     = max_steps
        self.step_delay_s  = step_delay_s

    def _get_vlm_client(self):
        if self.mode == "hybrid":
            return self.remote_client or self.local_client
        return self.local_client or self.remote_client

    def run(self, task: str, start_url: Optional[str] = None) -> RunResult:
        result  = RunResult(task=task, mode=self.mode, success=False)
        history: list[str] = []
        t_start = time.perf_counter()

        if start_url:
            self.bridge.navigate(start_url)
            time.sleep(2.0)

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
                dom_context  = build_dom_context(dom_elements, w, h)

            current_url = self.bridge.get_current_url()
            print(f"[executor] step {step_num} url: {current_url}")

            # ── URL heuristic: trigger evaluator proactively ──
            if _url_suggests_completion(task, current_url):
                print(f"[executor] URL suggests completion — running evaluator...")
                complete, reason = _evaluate_completion(
                    task, screenshot, current_url, client, w, h
                )
                if complete:
                    print(f"[executor] evaluator confirmed via URL heuristic: {reason}")
                    result.success = True
                    result.total_time_s = time.perf_counter() - t_start
                    self.bridge.push_status({"type": "done", "result": result.to_dict()})
                    return result
                else:
                    print(f"[executor] evaluator rejected URL heuristic: {reason} — continuing")

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

            self.bridge.push_status({
                "step":       step_num,
                "thought":    action.thought,
                "action":     action.to_dict(),
                "latency_s":  round(latency, 3),
                "screenshot": step.screenshot_b64,
            })

            # ── Terminal states ──
            if action.type == ActionType.DONE:
                print(f"[executor] model said done — running evaluator...")
                complete, reason = _evaluate_completion(
                    task, screenshot, current_url, client, w, h
                )
                if complete:
                    print(f"[executor] evaluator confirmed: {reason}")
                    result.success = True
                    if action.text and action.text.strip():
                        result.final_answer = action.text.strip()
                        print(f"[executor] final_answer: {result.final_answer[:100]}")
                    break
                else:
                    print(f"[executor] evaluator rejected: {reason} — continuing")

            if action.type == ActionType.FAIL:
                result.failure_reason = action.thought or "model returned fail"
                break

            # ── Loop detection: same coordinates ──
            if _is_coord_loop(history, action):
                print(f"[executor] coord loop detected: same (x,y) repeated 3+ times")
                result.failure_reason = (
                    f"stuck in loop: same coordinates "
                    f"({action.x:.2f}, {action.y:.2f}) repeated"
                )
                break

            # ── Fallback loop detection: same action type ──
            if len(history) >= 3:
                last_3_types = [s.split(":")[0] for s in history[-3:]]
                if (
                    len(set(last_3_types)) == 1
                    and last_3_types[0] == action.type.value
                    and action.type not in (ActionType.SCROLL,)
                ):
                    print(f"[executor] type loop detected: repeated {action.type.value}")
                    result.failure_reason = f"stuck in loop: repeated {action.type.value}"
                    break

            # ── Act ──
            try:
                self.bridge.execute_action(action)
            except httpx.HTTPError as e:
                result.failure_reason = f"action execution error: {e}"
                break

            # Auto Enter after type
            if action.type == ActionType.TYPE and action.text:
                enter_action = AgentAction(type=ActionType.KEY, key="Enter")
                try:
                    self.bridge.execute_action(enter_action)
                    print(f"[executor] auto Enter after type")
                except Exception:
                    pass

            history.append(
                f"{action.type.value}: x={action.x}, y={action.y}, text={action.text}"
            )

            if action.type == ActionType.NAVIGATE:
                time.sleep(2.0)
            time.sleep(self.step_delay_s)

        else:
            result.failure_reason = f"max steps ({self.max_steps}) reached"

        result.total_time_s = time.perf_counter() - t_start
        return result
