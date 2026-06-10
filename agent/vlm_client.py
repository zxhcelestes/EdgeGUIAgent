"""
VLM Client — wraps Ollama (qwen2.5vl:3b/7b) and Gemini Flash as hybrid fallback.
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
        excluded = {'thought', 'raw'}
        d = {k: v for k, v in self.__dict__.items() if v is not None and k not in excluded}
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
- When outputting "done" for extraction tasks, put the extracted information in the "text" field.
  Example: {"type": "done", "text": "The download count is 1.2M"}
- To fill a search box or input field: use "type" directly with the coordinates of the input.
  Do NOT first click then type — a single type action handles both focusing and typing.
"""

DOM_CONTEXT_TEMPLATE = """
--- DOM CONTEXT (interactable elements) ---
{dom_json}
--- END DOM CONTEXT ---

Use the center x/y values directly as your action coordinates.
For input elements: use "type" action directly with the center coordinates — do NOT click first.
"""


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_action_response(raw: str, screen_w: int, screen_h: int) -> AgentAction:
    print(f"[vlm] raw output: {raw[:300]}")

    if not raw or not raw.strip():
        print("[vlm] empty response from model")
        return AgentAction(type=ActionType.FAIL, thought="empty response from model", raw=raw)

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Primary: JSON parse
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
                raw=raw,
            )
        except json.JSONDecodeError as e:
            print(f"[vlm] json error: {e}")

    # Fallback: natural language format
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
        if x > 1.5: x = x / screen_w
        if y > 1.5: y = y / screen_h
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


# ── Shared evaluator prompt ───────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """You are evaluating whether a GUI agent has completed a task.
Look at the screenshot carefully. Respond with JSON only:
{"complete": true or false, "reason": "<one sentence explanation>"}
General rules:
- Focus on whether the GOAL of the task is achieved, not on the current UI state.
- Ignore irrelevant UI elements (search bars, navigation menus, ads) when judging completion.
- A task is complete if the required information is clearly visible anywhere on screen,
  even if a search box is also visible or partially filled.
Task-type specific rules:
- Navigation task ("go to X", "navigate to X"):
  true if the target page is fully loaded and visible.
- Search task ("search for X"):
  true if search results are displayed on screen.
- Extraction task ("report X", "find X", "what is X"):
  true if the specific information (a number, a sentence, a name, a date, code content,
  or any other requested data) is clearly readable on screen — regardless of what else
  is visible. The agent does NOT need to have spoken the answer aloud; it only needs
  to be visible on the page.
- Multi-step task:
  true only if ALL steps are complete, not just the most recent one.
Do not output anything except the JSON object."""


def _parse_eval_response(raw: str) -> tuple[bool, str]:
    """Parse evaluator response into (complete, reason)."""
    if not raw or not raw.strip():
        return False, "empty response"
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{[\s\S]*?\}", cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            return bool(data.get("complete", False)), data.get("reason", "")
        except json.JSONDecodeError:
            pass
    if re.search(r'"complete"\s*:\s*true', raw, re.IGNORECASE):
        return True, "fallback parse"
    return False, "parse error"


# ── Ollama client (qwen2.5vl:3b/7b) ──────────────────────────────────────────

class OllamaVLMClient:
    """Calls a local Ollama instance. Uses Ollama native image format."""

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

    def evaluate(
        self,
        screenshot_bytes: bytes,
        task: str,
        current_url: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[bool, str]:
        """Dedicated evaluator call with its own system prompt."""
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


# ── Gemini client (hybrid / fallback) ────────────────────────────────────────

class GeminiVLMClient:
    """Uses Gemini Flash as remote planner (free tier via AI Studio)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
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

    def evaluate(
        self,
        screenshot_bytes: bytes,
        task: str,
        current_url: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[bool, str]:
        """Dedicated evaluator call for Gemini hybrid mode."""
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
                config=types.GenerateContentConfig(
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

    form_els = [
        e for e in elements
        if e.get("tag") in ("input", "button", "textarea", "select")
    ][:10]
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
            "action": "type" if el.get("tag") in ("input", "textarea") else "click",
        })

    return DOM_CONTEXT_TEMPLATE.format(dom_json=json.dumps(compact, indent=2))
