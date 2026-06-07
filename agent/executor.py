"""
AgentExecutor — runs the perceive → plan → act loop.
Communicates with the Electron renderer via HTTP (localhost:7788).
"""

import base64
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
    Calls client.evaluate() if available, otherwise returns (False, 'not supported').
    Both OllamaVLMClient and GeminiVLMClient implement evaluate().
    Called only when the planner outputs 'done' — verifies before accepting.
    """
    if hasattr(client, "evaluate"):
        return client.evaluate(screenshot_bytes, task, current_url, screen_w, screen_h)
    return False, "client does not support evaluation"


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
                # VLM evaluator confirms completion before accepting
                print(f"[executor] model said done — running evaluator...")
                complete, reason = _evaluate_completion(
                    task, screenshot, current_url, client, w, h
                )
                if complete:
                    print(f"[executor] evaluator confirmed: {reason}")
                    result.success = True
                    break
                else:
                    print(f"[executor] evaluator rejected: {reason} — continuing")
                    # Don't break — keep going, model may have been premature

            if action.type == ActionType.FAIL:
                result.failure_reason = action.thought or "model returned fail"
                break

            # ── Loop detection ──
            if len(history) >= 3:
                last_3 = [s.split(":")[0] for s in history[-3:]]
                print(f"[executor] loop check: {last_3}, current: {action.type.value}")
                if len(set(last_3)) == 1 and last_3[0] == action.type.value:
                    print(f"[executor] loop detected: repeated {action.type.value} 3+ times")
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
