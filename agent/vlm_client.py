"""
VLM Client — wraps Ollama (qwen2.5vl:3b) and Gemini Flash as hybrid fallback.
Supports pure-screenshot mode and screenshot+DOM hybrid mode.
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
from google import genai
from google.genai import types


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
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if "type" in d:
            d["type"] = d["type"].value if hasattr(d["type"], "value") else str(d["type"])
        return d


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a GUI automation agent. You observe screenshots of a web browser and output the next action to complete the given task.

Always respond with a JSON object in this exact format:
{
  "thought": "<brief reasoning about current state and next step>",
  "action": {
    "type": "<click|type|scroll|navigate|key|done|fail>",
    "x": <0.0-1.0 normalized>,
    "y": <0.0-1.0 normalized>,
    "text": "<string>",
    "direction": "<up|down>",
    "amount": <integer>,
    "url": "<string>",
    "key": "<string>"
  }
}

Rules:
- Coordinates are NORMALIZED (0.0 = left/top, 1.0 = right/bottom).
- x and y must be single numbers, never lists or arrays.
- Use "done" when the task is fully complete. Examples:
  - Task says "search for X" → output done when search results page is visible.
  - Task says "navigate to X" → output done when the target page is loaded.
  - If you can see the expected result on screen, output done immediately.
- Use "fail" if the task is impossible or you are stuck after multiple retries.
- Be precise: prefer clicking visible buttons/links over typing URLs.
- If DOM context is provided, use the center x/y values directly as coordinates.
- Do NOT keep pressing Enter or clicking if the page has already changed.
- Respond with JSON only. No markdown fences, no extra text.
- Your entire response must be a single valid JSON object starting with { and ending with }.
"""

DOM_CONTEXT_TEMPLATE = """
--- DOM CONTEXT (interactable elements, max 15) ---
{dom_json}
--- END DOM CONTEXT ---

Use normalized_center x/y values directly as your action coordinates.
"""


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_action_response(raw: str, screen_w: int, screen_h: int) -> AgentAction:
    """
    Parse model output into AgentAction.
    Handles:
      - Clean JSON
      - JSON wrapped in markdown fences
      - Natural language fallback: "Action: click: x=1249, y=80"
      - Empty response
    """
    print(f"[vlm] raw output: {raw[:300]}")

    # ── Empty response guard ──
    if not raw or not raw.strip():
        print("[vlm] empty response from model")
        return AgentAction(type=ActionType.FAIL, thought="empty response from model", raw=raw)

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # ── Primary: JSON parse ──
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group(0))

            action_data     = data.get("action", {})
            thought         = data.get("thought", "")
            action_type_str = action_data.get("type", "fail").lower()

            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                action_type = ActionType.FAIL

            x = action_data.get("x")
            y = action_data.get("y")

            # Model sometimes outputs [x, y] list — take first element
            if isinstance(x, list):
                x = x[0] if x else None
            if isinstance(y, list):
                y = y[0] if y else None

            # Ensure numeric
            try:
                x = float(x) if x is not None else None
                y = float(y) if y is not None else None
            except (TypeError, ValueError):
                x, y = None, None

            # Normalize pixel coords to 0-1
            if x is not None and x > 1.5:
                x = x / screen_w
            if y is not None and y > 1.5:
                y = y / screen_h

            return AgentAction(
                type=action_type,
                x=x,
                y=y,
                text=action_data.get("text"),
                direction=action_data.get("direction"),
                amount=action_data.get("amount"),
                url=action_data.get("url"),
                key=action_data.get("key"),
                thought=thought,
                raw=raw,
            )
        except json.JSONDecodeError as e:
            print(f"[vlm] json error: {e}")

    # ── Fallback: natural language format ──
    action_match  = re.search(
        r"Action:\s*(\w+).*?x[=:]\s*([\d.]+).*?y[=:]\s*([\d.]+)",
        cleaned, re.IGNORECASE
    )
    text_match    = re.search(r'text[=:]\s*["\']?([^"\'\n,}]+)', cleaned, re.IGNORECASE)
    thought_match = re.search(r"Thought:\s*(.+?)(?:\n|Action:|$)", cleaned, re.IGNORECASE | re.DOTALL)

    if action_match:
        action_type_str = action_match.group(1).lower()
        x = float(action_match.group(2))
        y = float(action_match.group(3))
        if x > 1.5:
            x = x / screen_w
        if y > 1.5:
            y = y / screen_h
        thought = thought_match.group(1).strip() if thought_match else ""
        text    = text_match.group(1).strip() if text_match else None
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.FAIL
        print(f"[vlm] fallback parsed: {action_type_str} x={x:.3f} y={y:.3f}")
        return AgentAction(type=action_type, x=x, y=y, text=text, thought=thought, raw=raw)

    print(f"[vlm] parse failed completely: {cleaned[:100]}")
    return AgentAction(type=ActionType.FAIL, thought="parse error", raw=raw)


# ── Ollama client (qwen2.5vl:3b) ─────────────────────────────────────────────

class OllamaVLMClient:
    """
    Calls a local Ollama instance with Ollama-native image format.
    Uses proxy=None to bypass system proxy for localhost requests.
    Retries once on 500 errors (memory pressure crashes).
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        self.model    = model
        self.base_url = base_url
        self.timeout  = timeout
        self._client  = httpx.Client(
            timeout=timeout,
            transport=httpx.HTTPTransport(proxy=None),
        )

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
                {"role": "system", "content": SYSTEM_PROMPT},
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
                    print(f"[vlm] Ollama 500, retrying in 5s...")
                    time.sleep(5.0)
                    continue
                raise
        latency = time.perf_counter() - t0

        raw    = resp.json()["message"]["content"]
        action = _parse_action_response(raw, screen_w, screen_h)
        return action, latency

    def close(self):
        self._client.close()


# ── Gemini client (hybrid / fallback) ────────────────────────────────────────

class GeminiVLMClient:
    """Uses Gemini 1.5 Flash as remote planner (free tier via AI Studio)."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
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
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        latency = time.perf_counter() - t0

        action = _parse_action_response(response.text, screen_w, screen_h)
        return action, latency

    def close(self):
        pass


# ── DOM context builder ───────────────────────────────────────────────────────

def build_dom_context(elements: list[dict], screen_w: int, screen_h: int) -> str:
    screen_w = screen_w or 1280
    screen_h = screen_h or 800

    # Priority 1: form elements (input, button, textarea, select)
    form_els = [
        e for e in elements
        if e.get("tag") in ("input", "button", "textarea", "select")
    ][:8]

    # Priority 2: top-level nav links (short text, near top of page)
    nav_links = [
        e for e in elements
        if e.get("tag") == "a"
        and 0 < len(e.get("text") or "") < 30
        and e.get("rect", {}).get("top", 999) < 100
    ][:8]

    # Priority 3: other short links not in nav
    used_texts = {e.get("text") for e in form_els + nav_links}
    other_links = [
        e for e in elements
        if e.get("tag") == "a"
        and 0 < len(e.get("text") or "") < 30
        and e.get("text") not in used_texts
        and e.get("rect", {}).get("top", 999) >= 100
    ][:4]

    selected = form_els + nav_links + other_links

    compact = []
    for el in selected:
        rect = el.get("rect", {})
        cx = (rect.get("left", 0) + rect.get("width", 0) / 2) / screen_w
        cy = (rect.get("top", 0) + rect.get("height", 0) / 2) / screen_h
        compact.append({
            "tag": el.get("tag"),
            "text": (el.get("text") or "")[:40],
            "center": {"x": round(cx, 3), "y": round(cy, 3)},
        })

    return DOM_CONTEXT_TEMPLATE.format(dom_json=json.dumps(compact, indent=2))