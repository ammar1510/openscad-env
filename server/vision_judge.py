"""
Vision LLM judge for grading OpenSCAD renders.

Renders the .scad file from multiple camera angles, sends the images
to a vision-capable LLM via an OpenAI-compatible API, and parses
structured scores from the response.

Configuration is via environment variables:
    OPENSCAD_VISION_API_BASE  - e.g. "https://api.openai.com" or "http://localhost:8080"
    OPENSCAD_VISION_MODEL     - e.g. "gpt-4o"
    OPENSCAD_VISION_API_KEY   - API key (optional for local endpoints)
"""

import base64
import json
import logging
import os
import re
import subprocess
from typing import Dict, Optional, Tuple

import httpx

from .tasks import Task

logger = logging.getLogger(__name__)

# (translate_x, translate_y, translate_z, rot_x, rot_y, rot_z, distance)
CAMERA_VIEWS = {
    "front": "0,0,0,55,0,25,200",
    "top": "0,0,0,90,0,0,200",
    "side": "0,0,0,55,0,90,200",
}

_DEFAULT_CRITERIA = {
    "shape": "Does the object match the described shape?",
    "proportions": "Do the relative proportions look correct?",
    "completeness": "Are all required features present?",
}

TASK_CRITERIA: Dict[str, Dict[str, str]] = {
    "basic_box": {
        "shape": "Is this a single rectangular box/prism with no extra geometry?",
        "proportions": "Does the box look wider than deep, and shallow in height?",
        "completeness": "Is it a clean solid box with sharp edges and flat faces?",
    },
    "hollow_cylinder": {
        "shape": "Is this a hollow cylinder (tube) with a visible hole through the center?",
        "proportions": (
            "Does the tube have roughly equal width and depth, "
            "with noticeable wall thickness?"
        ),
        "completeness": (
            "Is the hole clearly visible and does the cylinder "
            "have smooth curved walls?"
        ),
    },
    "stacking_blocks": {
        "shape": (
            "Are there 3 distinct rectangular blocks stacked on top of each other?"
        ),
        "proportions": (
            "Do the blocks decrease in size from bottom to top, "
            "forming a stepped pyramid?"
        ),
        "completeness": (
            "Are all 3 blocks visible, centered on each other, "
            "and touching with no gaps?"
        ),
    },
    "phone_stand": {
        "shape": (
            "Does this look like a phone stand with an angled surface "
            "and a slot or groove?"
        ),
        "proportions": (
            "Does the stand have a stable base and an appropriate "
            "angle for holding a phone?"
        ),
        "completeness": (
            "Is there a visible slot for a phone, a stable base, "
            "and would it plausibly hold a phone?"
        ),
    },
}

SCORE_KEYS = ("shape", "proportions", "completeness")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_views(
    scad_path: str,
    output_dir: str,
    imgsize: Tuple[int, int] = (512, 512),
    timeout_s: float = 30,
) -> Dict[str, str]:
    """Render multiple camera views of an OpenSCAD file to PNG."""
    rendered: Dict[str, str] = {}
    for view_name, camera in CAMERA_VIEWS.items():
        png_path = os.path.join(output_dir, f"render_{view_name}.png")
        try:
            subprocess.run(
                [
                    "openscad",
                    "-o",
                    png_path,
                    f"--camera={camera}",
                    f"--imgsize={imgsize[0]},{imgsize[1]}",
                    "--colorscheme=Tomorrow",
                    scad_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                rendered[view_name] = png_path
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to render %s view: %s", view_name, exc)
    return rendered


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(task: Task) -> str:
    criteria = TASK_CRITERIA.get(task.task_id, _DEFAULT_CRITERIA)
    criteria_lines = "\n".join(
        f"- **{name}**: {desc}" for name, desc in criteria.items()
    )
    return (
        "You are grading a 3D model rendered from OpenSCAD code.\n\n"
        f"TASK: {task.description}\n\n"
        "You are shown the model from three camera angles "
        "(front/isometric, top-down, side).\n"
        "Score each criterion from 0.0 to 1.0:\n\n"
        f"{criteria_lines}\n\n"
        "Scoring guide:\n"
        "- 1.0: Clearly correct\n"
        "- 0.5: Recognizable attempt but flawed\n"
        "- 0.0: Completely wrong, empty, or unrecognizable\n\n"
        "Respond ONLY with JSON, no other text:\n"
        '{"shape": 0.X, "proportions": 0.X, "completeness": 0.X}'
    )


# ---------------------------------------------------------------------------
# LLM call + parsing
# ---------------------------------------------------------------------------


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_scores(text: str) -> Tuple[float, Dict[str, float]]:
    """Extract JSON scores from the LLM response text."""
    match = re.search(r"\{[^}]+\}", text)
    if not match:
        return 0.0, {}
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return 0.0, {}

    scores: Dict[str, float] = {}
    for key in SCORE_KEYS:
        if key in raw:
            scores[key] = max(0.0, min(1.0, float(raw[key])))

    if not scores:
        return 0.0, {}

    avg = sum(scores.values()) / len(scores)
    return round(avg, 4), {k: round(v, 4) for k, v in scores.items()}


def judge(
    task: Task,
    rendered_views: Dict[str, str],
    api_base: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 30,
) -> Tuple[float, Dict[str, float]]:
    """Send rendered views to a vision LLM and return parsed scores."""
    if not rendered_views:
        return 0.0, {}

    prompt = _build_prompt(task)

    content: list = [{"type": "text", "text": prompt}]
    for view_name in ("front", "top", "side"):
        path = rendered_views.get(view_name)
        if path is None:
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(path)}",
                    "detail": "low",
                },
            }
        )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    url = f"{api_base.rstrip('/')}/v1/chat/completions"

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_scores(text)
    except Exception as exc:
        logger.warning("Vision judge call failed: %s", exc)
        return 0.0, {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_config_from_env() -> Optional[Dict[str, str]]:
    """Read vision judge config from environment variables."""
    api_base = os.environ.get("OPENSCAD_VISION_API_BASE")
    if not api_base:
        return None
    return {
        "api_base": api_base,
        "model": os.environ.get("OPENSCAD_VISION_MODEL", "gpt-4o"),
        "api_key": os.environ.get("OPENSCAD_VISION_API_KEY", ""),
    }
