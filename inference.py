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


def log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env=openscad model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # truncate action to avoid very long lines
    action_short = action[:80].replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
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

    log_start(task_id)

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(obs.task_description)},
    ]

    best_score = 0.0
    rewards: List[float] = []
    steps_taken = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
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
                    time.sleep(wait)

            assert assistant_text is not None
            messages.append({"role": "assistant", "content": assistant_text})

            code = extract_openscad_code(assistant_text)

            # Submit to environment
            result = env.step(OpenSCADAction(code=code))
            obs = result.observation

            current_score = result.reward or 0.0
            rewards.append(current_score)
            best_score = max(best_score, current_score)

            error = obs.compile_error if not obs.compile_success else None
            log_step(
                step=step,
                action=code,
                reward=current_score,
                done=result.done,
                error=error,
            )

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
    finally:
        final_score = min(max(best_score, 0.01), 0.99)
        success = best_score >= 0.5
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score


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
            # log_end is already emitted by run_task's finally block
            scores[task_id] = 0.01

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
