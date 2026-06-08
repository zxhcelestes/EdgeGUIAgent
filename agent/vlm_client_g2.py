"""
VLM Client v2 — Planner + Grounder Architecture

Architecture:
  Planner  (Qwen2.5-VL via Ollama, 3B or 7B)
      → decides action type + describes the target element in natural language
  Grounder (GUI-G2-3B via transformers, CUDA required)
      → maps the element description to precise (x, y) coordinates on screen

For non-click actions (type, scroll, navigate, key), the planner output is used
directly without calling the grounder.

NOTE: GUI-G2 requires CUDA. On macOS MPS, the grounder is automatically disabled
and the system falls back to planner-only mode.

Model selection:
  OLLAMA_MODEL=qwen2.5vl:3b   (default, M2 16GB)
  OLLAMA_MODEL=qwen2.5vl:7b   (recommended, 24GB+ VRAM)
"""

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx
from PIL import Image

from gui_g2_client import GUIG2GrounderClient


# ── Types ─────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    CLICK    = "click"
    TYPE     = "type"
    SCROLL   = "scroll"
    NAVIGATE = "navigate"
    DONE     = "done"
    FAIL     = "fail"
    KEY      = "key"


@dataclass
class AgentAction:
    type: ActionType
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    amount: Optional[int] = None
    url: Optional[str] = None
    key: Optional[str] = None
    thought: Optional[str] = None
    target_description: Optional[str] = None  # used by grounder, not sent to Electron
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if "type" in d:
            d["type"] = d["type"].value if hasattr(d["type"], "value") else str(d["type"])
        d.pop("target_description", None)
        d.pop("raw", None)
        return d


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a GUI automation agent. You observe screenshots of a web browser and decide the next action.

For CLICK actions, describe the target element clearly in natural language instead of guessing coordinates.
For all other actions, output coordinates normally.

Always respond with a JSON object:
{
  "thought": "<brief reasoning>",
  "action": {
    "type": "<click|type|scroll|navigate|key|done|fail>",
    "target": "<natural language description of the element to click>",
    "x": <0.0-1.0, only for non-click actions>,
    "y": <0.0-1.0, only for non-click actions>,
    "text": "<string, for type actions>",
    "direction": "<up|down, for scroll>",
    "amount": <integer, for scroll>,
    "url": "<string, for navigate>",
    "key": "<string, for key>"
  }
}

Rules:
- For click: always fill "target" with a clear description. Omit x/y.
- For type/scroll/key/navigate/done/fail: fill x/y/text/etc as normal.
- Use "done" when the task goal is visibly achieved on screen. Examples:
  - Search task → done when results page is fully loaded and visible
  - Navigation task → done when target page URL is loaded
  - Extraction task → done when the required information is visible on screen
- Use "fail" only if truly stuck after multiple retries.
- Respond with JSON only, no markdown, no extra text.
- Your entire response must be a single valid JSON object.
- When outputting "done" for extraction tasks, put the extracted information in the "text" field.
  Example: {"type": "done", "text": "The download count is 1.2M"}
- To fill a search box or input field: use "type" directly with the coordinates of the input.
  Do NOT first click then type — a single type action handles both focusing and typing.
"""

DOM_CONTEXT_TEMPLATE = """
--- DOM CONTEXT (interactable elements) ---
{dom_json}
--- END DOM CONTEXT ---
Use center x/y values for non-click actions. For click, prefer using "target" description.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _parse_planner_response(raw: str, screen_w: int, screen_h: int) -> AgentAction:
    print(f"[planner] raw output: {raw[:300]}")

    if not raw or not raw.strip():
        return AgentAction(type=ActionType.FAIL, thought="empty response", raw=raw)

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return AgentAction(type=ActionType.FAIL, thought="no JSON found", raw=raw)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"[planner] json error: {e}")
        return AgentAction(type=ActionType.FAIL, thought="json decode error", raw=raw)

    action_data     = data.get("action", {})
    thought         = data.get("thought", "")
    action_type_str = action_data.get("type", "fail").lower()

    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.FAIL

    target = action_data.get("target") or action_data.get("description")

    x = action_data.get("x")
    y = action_data.get("y")
    if isinstance(x, list): x = x[0] if x else None
    if isinstance(y, list): y = y[0] if y else None
    try:
        x = float(x) if x is not None else None
        y = float(y) if y is not None else None
    except (TypeError, ValueError):
        x, y = None, None
    if x is not None and x > 1.5: x = x / screen_w
    if y is not None and y > 1.5: y = y / screen_h

    return AgentAction(
        type=action_type,
        x=x, y=y,
        text=action_data.get("text"),
        direction=action_data.get("direction"),
        amount=action_data.get("amount"),
        url=action_data.get("url"),
        key=action_data.get("key"),
        thought=thought,
        target_description=target,
        raw=raw,
    )


# ── Planner + Grounder VLM client ─────────────────────────────────────────────

class OllamaVLMClient:
    """
    Two-stage client:
      1. Ollama planner (3B or 7B) — decides action type, describes click targets
      2. GUI-G2 grounder (CUDA only) — maps description to precise coordinates

    Falls back to planner-only if GUI-G2 is unavailable (macOS, no CUDA).
    Supports 7B model via OLLAMA_MODEL env var for 24GB+ environments.
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        grounder_model_path: str = "./models/GUI-G2-3B",
    ):
        self.model    = model
        self.base_url = base_url
        self.timeout  = timeout
        self._client  = httpx.Client(
            timeout=timeout,
            transport=httpx.HTTPTransport(proxy=None),
        )

        # Grounder disabled on macOS (MPS bfloat16 unsupported, CPU too slow)
        self._grounder = GUIG2GrounderClient(grounder_model_path)
        self._grounder_available = False  # set True only on CUDA environments
        if self._grounder.is_available():
            import torch
            if torch.cuda.is_available():
                self._grounder_available = True
                print(f"[planner+grounder] GUI-G2 enabled (CUDA) at {grounder_model_path}")
            else:
                print(f"[planner+grounder] GUI-G2 found but disabled (no CUDA) — planner-only mode")
        else:
            print(f"[planner+grounder] GUI-G2 not found at {grounder_model_path} — planner-only mode")

        print(f"[planner+grounder] planner model: {model}")

    def is_available(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def get_action(
        self,
        screenshot_bytes: bytes,
        task: str,
        history: list[str],
        dom_context: Optional[str],
        screen_w: int,
        screen_h: int,
    ) -> tuple[AgentAction, float]:

        # ── Stage 1: Planner ──
        history_text = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history))
        prompt_parts = [f"Task: {task}"]
        if history_text:
            prompt_parts.append(f"Previous steps:\n{history_text}")
        if dom_context:
            prompt_parts.append(dom_context)
        prompt_parts.append("What is the next action?")

        b64 = _encode_image(screenshot_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n\n".join(prompt_parts),
                    "images": [b64],
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        t0 = time.perf_counter()
        for attempt in range(2):
            try:
                resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 500 and attempt == 0:
                    print("[planner] Ollama 500, retrying in 5s...")
                    time.sleep(5.0)
                    continue
                raise
        planner_latency = time.perf_counter() - t0

        action = _parse_planner_response(
            resp.json()["message"]["content"], screen_w, screen_h
        )

        # ── Stage 2: Grounder (CUDA only, click actions only) ──
        if (
            action.type == ActionType.CLICK
            and action.target_description
            and self._grounder_available
        ):
            print(f"[grounder] grounding: '{action.target_description}'")
            gx, gy, g_latency = self._grounder.ground(
                screenshot_bytes=screenshot_bytes,
                element_description=action.target_description,
                screen_w=screen_w,
                screen_h=screen_h,
            )
            total_latency = planner_latency + g_latency

            if gx is not None:
                print(f"[grounder] result: ({gx:.3f}, {gy:.3f}) in {g_latency:.1f}s")
                action.x = gx
                action.y = gy
                action.thought = f"{action.thought or ''} | grounder: ({gx:.3f}, {gy:.3f})"
            else:
                print("[grounder] grounding failed, using planner fallback coords")
                total_latency = planner_latency
        else:
            total_latency = planner_latency

        return action, total_latency

    def evaluate(
        self,
        screenshot_bytes: bytes,
        task: str,
        current_url: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[bool, str]:
        """
        Dedicated evaluator call — separate system prompt, no action format.
        Returns (is_complete, reason).
        """
        from vlm_client import EVAL_SYSTEM_PROMPT, _parse_eval_response
        prompt = (
            f"Task: {task}\n"
            f"Current URL: {current_url}\n\n"
            "Has this task been fully completed based on what you see in the screenshot?"
        )
        b64 = _encode_image(screenshot_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            print(f"[eval] raw: {raw[:150]}")
            return _parse_eval_response(raw)
        except Exception as e:
            print(f"[eval] error: {e}")
            return False, f"evaluator error: {e}"

    def close(self):
        self._client.close()


# ── Gemini client (hybrid mode) ───────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class GeminiVLMClient:
    """Remote Gemini Flash planner (hybrid mode). No grounder needed — strong coord prediction."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        if not _GENAI_AVAILABLE:
            raise ImportError("google-genai not installed")
        self.model  = model
        self.client = genai.Client(api_key=api_key)

    def is_available(self) -> bool:
        return True

    def get_action(
        self,
        screenshot_bytes: bytes,
        task: str,
        history: list[str],
        dom_context: Optional[str],
        screen_w: int,
        screen_h: int,
    ) -> tuple[AgentAction, float]:
        from vlm_client import SYSTEM_PROMPT, _parse_action_response
        history_text = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history))
        prompt_parts = [f"Task: {task}"]
        if history_text:
            prompt_parts.append(f"Previous steps:\n{history_text}")
        if dom_context:
            prompt_parts.append(dom_context)
        prompt_parts.append("What is the next action?")

        image = Image.open(io.BytesIO(screenshot_bytes))
        t0 = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=["\n\n".join(prompt_parts), image],
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        latency = time.perf_counter() - t0
        action = _parse_action_response(response.text, screen_w, screen_h)
        return action, latency

    def evaluate(
        self,
        screenshot_bytes: bytes,
        task: str,
        current_url: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[bool, str]:
        """Dedicated evaluator call for Gemini hybrid mode."""
        from vlm_client import EVAL_SYSTEM_PROMPT, _parse_eval_response
        prompt = (
            f"Task: {task}\n"
            f"Current URL: {current_url}\n\n"
            "Has this task been fully completed based on what you see in the screenshot?"
        )
        image = Image.open(io.BytesIO(screenshot_bytes))
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
                config=genai_types.GenerateContentConfig(
                    system_instruction=EVAL_SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            print(f"[eval] raw: {response.text[:150]}")
            return _parse_eval_response(response.text)
        except Exception as e:
            print(f"[eval] gemini error: {e}")
            return False, f"evaluator error: {e}"

    def close(self):
        pass


# ── DOM context builder ───────────────────────────────────────────────────────

def build_dom_context(elements: list[dict], screen_w: int, screen_h: int) -> str:
    screen_w = screen_w or 1280
    screen_h = screen_h or 800

    form_els = [e for e in elements if e.get("tag") in ("input", "button", "textarea", "select")][:10]
    used_texts = {e.get("text", "") for e in form_els}
    nav_links = [
        e for e in elements
        if e.get("tag") == "a"
        and 0 < len(e.get("text") or "") < 30
        and e.get("text") not in used_texts
    ][:5]

    compact = []
    for el in form_els + nav_links:
        rect = el.get("rect", {})
        cx = (rect.get("left", 0) + rect.get("width",  0) / 2) / screen_w
        cy = (rect.get("top",  0) + rect.get("height", 0) / 2) / screen_h
        compact.append({
            "tag":    el.get("tag"),
            "text":   (el.get("text") or "")[:40],
            "center": {"x": round(cx, 3), "y": round(cy, 3)},
        })

    return DOM_CONTEXT_TEMPLATE.format(dom_json=json.dumps(compact, indent=2))
