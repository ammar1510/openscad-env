---
title: OpenSCAD 3D Modeling Environment
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# OpenSCAD 3D Modeling Environment

An OpenEnv environment where an LLM agent writes OpenSCAD code to create 3D objects matching given specifications. The environment compiles the code, measures the resulting geometry, and grades the submission against task requirements.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Complete OpenSCAD script to compile and evaluate |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | `str` | Natural language description of what to build |
| `task_id` | `str` | Task identifier (e.g. `basic_box`) |
| `available_tasks` | `list[str]` | All available task IDs |
| `compile_success` | `bool` | Whether OpenSCAD compiled successfully |
| `compile_error` | `str` | Compilation error message (empty if success) |
| `dimensions` | `dict` | Bounding box dimensions `{x, y, z}` in mm |
| `volume` | `float` | Mesh volume in mm^3 |
| `is_watertight` | `bool` | Whether the mesh is a valid solid |
| `score` | `float` | Composite grading score 0.0-1.0 |
| `score_breakdown` | `dict` | Per-component score breakdown |

## Tasks

| Task ID | Difficulty | Description |
|---------|-----------|-------------|
| `basic_box` | Easy | 30x20x10mm rectangular box |
| `hollow_cylinder` | Medium | Hollow tube, outer r=15mm, inner r=10mm, h=25mm |
| `stacking_blocks` | Medium | 3 centered blocks stacked (20/15/10mm cubes) |
| `phone_stand` | Hard | Phone stand at 60 degrees with slot |

## Scoring

Rewards are in `[0.0, 1.0]` computed from a weighted rubric:

- **Compilation** (20%): Binary pass/fail
- **Dimensions** (50%): Per-axis accuracy vs target
- **Volume** (30%): Volume accuracy vs target

When vision judge is enabled (via `OPENSCAD_VISION_*` env vars), weights shift to include a 40% vision component scored by a multimodal LLM.

## Setup

### Install dependencies

```bash
uv sync
```

### Run the server locally

```bash
uv run --project . server
```

### Run with Docker

```bash
docker build -t openscad-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 openscad-env:latest
```

### Run inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=openai/gpt-oss-120b:novita
export HF_TOKEN=your_token_here

python inference.py
```

## Client Usage

```python
from openscad_env import OpenSCADEnv, OpenSCADAction

env = OpenSCADEnv(base_url="http://localhost:8000").sync()

result = env.reset(task_id="basic_box")
print(result.observation.task_description)

result = env.step(OpenSCADAction(code="cube([30, 20, 10]);"))
print(f"Score: {result.observation.score}")
print(f"Dimensions: {result.observation.dimensions}")
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | Hugging Face / API authentication token |
| `OPENSCAD_VISION_API_BASE` | No | Vision judge API endpoint |
| `OPENSCAD_VISION_MODEL` | No | Vision model name (default: `gpt-4o`) |
| `OPENSCAD_VISION_API_KEY` | No | Vision API key |
