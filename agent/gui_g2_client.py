"""
GUI-G2 Grounder Client

Wraps the GUI-G2-3B model (inclusionAI/GUI-G2-3B) for precise GUI element grounding.
GUI-G2 takes a screenshot + natural language description of a target element,
and outputs normalized (x, y) coordinates.

This is used in conjunction with a planner VLM:
  Planner (Qwen2.5-VL via Ollama) → decides WHAT to do and describes the target
  Grounder (GUI-G2)               → finds WHERE the target is on screen

Model: inclusionAI/GUI-G2-3B (Qwen2.5-VL-3B fine-tuned with Gaussian rewards)
Paper: GUI-G²: Gaussian Reward Modeling for GUI Grounding (AAAI 2026)
Repo:  https://github.com/ZJU-REAL/GUI-G2
"""

import base64
import io
import re
import time
from typing import Optional

from PIL import Image

# Lazy imports — only load when model is actually used
_model = None
_processor = None
_model_path: str = ""


def _load_model(model_path: str):
    """Load GUI-G2 model lazily on first use."""
    global _model, _processor, _model_path

    if _model is not None and _model_path == model_path:
        return _model, _processor

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[gui-g2] Loading model from {model_path}...")
    t0 = time.perf_counter()

    # Use MPS on Apple Silicon if available, else CPU
    if torch.backends.mps.is_available():
        # device_map = {"": "mps"}
        device_map = {"": "cpu"}
        torch_dtype = torch.float32
        print("[gui-g2] Using MPS (Apple Silicon)")
    else:
        device_map = "auto"
        torch_dtype = torch.float32
        print("[gui-g2] Using CPU")

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        # flash_attention_2 requires CUDA — skip on MPS/CPU
    ).eval()

    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    _model_path = model_path

    print(f"[gui-g2] Model loaded in {time.perf_counter() - t0:.1f}s")
    return _model, _processor


# ── Grounding prompt ──────────────────────────────────────────────────────────
# GUI-G2 uses this exact prompt format from the paper / official inference code

GROUNDING_PROMPT_TEMPLATE = (
    "Outline the position coordinates of the element corresponding to the "
    "following description: {description}"
)


def _parse_coordinates(output_text: str, screen_w: int, screen_h: int) -> Optional[tuple[float, float]]:
    """
    Parse GUI-G2 output into normalized (x, y) coordinates.

    GUI-G2 outputs coordinates in one of these formats:
      - "(0.48, 0.05)"            — already normalized
      - "[[512, 40]]"             — pixel absolute
      - "(512, 40)"               — pixel absolute
      - "x=512, y=40"
    """
    print(f"[gui-g2] raw output: {output_text[:100]}")

    # Try normalized tuple: (0.48, 0.05)
    m = re.search(r"\(\s*(0\.\d+|\d+\.\d+)\s*,\s*(0\.\d+|\d+\.\d+)\s*\)", output_text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        # If both < 1.5 assume already normalized
        if x <= 1.5 and y <= 1.5:
            return x, y

    # Try pixel list format: [[512, 40]] or [512, 40]
    m = re.search(r"\[+\s*(\d+)\s*,\s*(\d+)\s*\]+", output_text)
    if m:
        x = float(m.group(1)) / screen_w
        y = float(m.group(2)) / screen_h
        return x, y

    # Try plain tuple with pixels: (512, 40)
    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", output_text)
    if m:
        x = float(m.group(1)) / screen_w
        y = float(m.group(2)) / screen_h
        return x, y

    # Try x=N, y=N format
    mx = re.search(r"x\s*=\s*(\d+\.?\d*)", output_text, re.IGNORECASE)
    my = re.search(r"y\s*=\s*(\d+\.?\d*)", output_text, re.IGNORECASE)
    if mx and my:
        x, y = float(mx.group(1)), float(my.group(1))
        if x > 1.5:
            x /= screen_w
        if y > 1.5:
            y /= screen_h
        return x, y

    print(f"[gui-g2] failed to parse coordinates from: {output_text[:100]}")
    return None


class GUIG2GrounderClient:
    """
    Local GUI-G2 grounder. Takes a screenshot + element description,
    returns normalized (x, y) coordinates.

    Usage:
        grounder = GUIG2GrounderClient("./models/GUI-G2-3B")
        x, y, latency = grounder.ground(screenshot_bytes, "the search button", w, h)
    """

    def __init__(self, model_path: str = "./models/GUI-G2-3B"):
        self.model_path = model_path
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._processor = _load_model(self.model_path)

    def is_available(self) -> bool:
        """Check if model directory exists."""
        import os
        return os.path.isdir(self.model_path)

    def ground(
        self,
        screenshot_bytes: bytes,
        element_description: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[Optional[float], Optional[float], float]:
        """
        Ground an element description to screen coordinates.

        Returns:
            (x, y, latency_s) where x/y are normalized 0-1, or (None, None, latency)
            if grounding fails.
        """
        self._ensure_loaded()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("[gui-g2] qwen_vl_utils not found, install with: pip install qwen-vl-utils")
            return None, None, 0.0

        prompt = GROUNDING_PROMPT_TEMPLATE.format(description=element_description)

        # Save screenshot to temp file (qwen_vl_utils needs a path or URL)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(screenshot_bytes)
            tmp_path = f.name

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(next(self._model.parameters()).device)

            t0 = time.perf_counter()
            import torch
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=64)
            latency = time.perf_counter() - t0

            generated_ids_trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            coords = _parse_coordinates(output_text, screen_w, screen_h)
            if coords:
                return coords[0], coords[1], latency
            return None, None, latency

        finally:
            os.unlink(tmp_path)
