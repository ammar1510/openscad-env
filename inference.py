#!/usr/bin/env python3
"""Baseline inference script for the OpenSCAD 3D Modeling Environment.

Uses an OpenAI-compatible LLM to generate OpenSCAD code for each task,
submits it to the environment server, and logs structured results.

Prerequisites:
    1. Start the environment server:
           uv run --project . server
       or via Docker:
           docker build -t openscad-env:latest -f server/Dockerfile .
           docker run -d -p 8000:8000 openscad-env:latest

    2. Set environment variables:
           export API_BASE_URL=https://router.huggingface.co/v1
           export MODEL_NAME=openai/gpt-oss-120b:novita
           export HF_TOKEN=your_token_here

    3. Run:
           python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List

from openai import APITimeoutError, APIConnectionError, RateLimitError

from openai import OpenAI

from openscad_env import OpenSCADAction, OpenSCADEnv

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b:novita")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "https://ammar-shaikh-openscad-env.hf.space")

MAX_STEPS = 3  # max attempts per task
TASKS = ["basic_box", "hollow_cylinder", "stacking_blocks", "phone_stand"]

SYSTEM_PROMPT = (
    "You are an expert OpenSCAD programmer. You write complete, valid OpenSCAD "
    "scripts that create 3D objects matching the given specifications.\n\n"
    "Rules:\n"
    "- Respond ONLY with OpenSCAD code inside a ```openscad code block\n"
    "- The code must be a complete, self-contained script\n"
    "- Use precise dimensions as specified in the task\n"
    "- Use $fn=64 or higher for smooth curved surfaces\n"
    "- Do not include comments explaining the code, just write the code"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_openscad_code(text: str) -> str:
    """Extract OpenSCAD code from a model response."""
    # Try fenced code block first
    blocks = re.findall(
        r"```(?:openscad|scad)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if blocks:
        return blocks[0].strip()
    # Fallback: return the whole response stripped
    return text.strip()


def build_initial_prompt(task_description: str) -> str:
    """Build the first user prompt for a task."""
    return (
        f"Write OpenSCAD code to create the following 3D object:\n\n"
        f"{task_description}\n\n"
        "Respond with the complete OpenSCAD script in a ```openscad code block."
    )


def build_feedback_prompt(
    step: int,
    compile_success: bool,
    compile_error: str,
    dimensions: dict,
    volume: float,
    score: float,
    score_breakdown: dict,
) -> str:
    """Build a feedback prompt after a failed or low-scoring attempt."""
    parts = [f"Attempt {step} results:"]

    if not compile_success:
        parts.append(f"COMPILATION FAILED: {compile_error}")
        parts.append("Fix the compilation error and try again.")
    else:
        parts.append(f"Dimensions: {json.dumps(dimensions)}")
        parts.append(f"Volume: {volume:.1f} mm^3")
        parts.append(f"Score: {score:.4f}")
        if score_breakdown:
            parts.append(f"Breakdown: {json.dumps(score_breakdown)}")
        if score < 0.9:
            parts.append(
                "The score is below 0.9. Adjust dimensions/geometry to "
                "better match the target and try again."
            )

    parts.append(
        "\nRespond with an improved OpenSCAD script in a ```openscad code block."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def log_start(task_id: str, task_description: str) -> None:
    print(
        f"[START] task_id={task_id} "
        f"description={json.dumps(task_description[:100])}"
    )


def log_step(
    task_id: str,
    step: int,
    reward: float,
    done: bool,
    score_breakdown: dict,
) -> None:
    print(
        f"[STEP] task_id={task_id} step={step} "
        f"reward={reward:.4f} done={done} "
        f"breakdown={json.dumps(score_breakdown)}"
    )


def log_end(task_id: str, final_reward: float, steps_taken: int) -> None:
    print(
        f"[END] task_id={task_id} "
        f"final_reward={final_reward:.4f} steps={steps_taken}"
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_task(
    env: OpenSCADEnv,
    client: OpenAI,
    task_id: str,
) -> float:
    """Run inference on a single task. Returns the best score achieved."""
    result = env.reset(task_id=task_id)
    obs = result.observation

    log_start(task_id, obs.task_description)

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(obs.task_description)},
    ]

    best_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        # Query the LLM with retry logic
        assistant_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )
                assistant_text = response.choices[0].message.content.strip()
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                # print(f"[DEBUG] LLM call failed (attempt {attempt+1}/3): {e}, retrying in {wait}s")
                time.sleep(wait)

        assert assistant_text is not None
        messages.append({"role": "assistant", "content": assistant_text})

        code = extract_openscad_code(assistant_text)

        # Submit to environment
        result = env.step(OpenSCADAction(code=code))
        obs = result.observation

        # if not obs.compile_success:
        #     print(f"[DEBUG] task_id={task_id} step={step} compile_error={obs.compile_error}")
        #     print(f"[DEBUG] extracted code (first 200 chars): {code[:200]}")

        log_step(
            task_id=task_id,
            step=step,
            reward=result.reward or 0.0,
            done=result.done,
            score_breakdown=obs.score_breakdown,
        )

        current_score = result.reward or 0.0
        best_score = max(best_score, current_score)

        # Stop early if we got a good score
        if current_score >= 0.9:
            break

        # Build feedback for next attempt
        if step < MAX_STEPS:
            feedback = build_feedback_prompt(
                step=step,
                compile_success=obs.compile_success,
                compile_error=obs.compile_error,
                dimensions=obs.dimensions,
                volume=obs.volume,
                score=obs.score,
                score_breakdown=obs.score_breakdown,
            )
            messages.append({"role": "user", "content": feedback})

    log_end(task_id, best_score, step)
    return best_score


def main() -> None:
    if not API_KEY:
        print(
            "ERROR: HF_TOKEN or API_KEY must be set.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Config: model={MODEL_NAME} env={ENV_URL} tasks={TASKS}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=60.0)
    env = OpenSCADEnv(base_url=ENV_URL).sync()

    scores = {}
    start_time = time.time()

    for task_id in TASKS:
        try:
            score = run_task(env, client, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[END] task_id={task_id} final_reward=0.0000 error={e}")
            scores[task_id] = 0.0

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    for tid, score in scores.items():
        print(f"  {tid:20s}  {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':20s}  {avg:.4f}")
    print(f"  {'TIME':20s}  {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
