"""
OpenSCAD 3D Modeling Environment Implementation.

An environment where an LLM agent writes OpenSCAD code to create 3D objects
matching given specifications. The environment compiles the code via the
OpenSCAD CLI, measures the resulting geometry with trimesh, and grades the
submission against task requirements.
"""

import os
import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from openscad_env.models import OpenSCADAction, OpenSCADObservation, OpenSCADState

from .rubrics import OpenSCADRubric
from .tasks import get_task, list_tasks
from .vision_judge import get_config_from_env


class OpenSCADEnvironment(Environment):
    """
    OpenSCAD 3D modeling environment.

    The agent receives a task description on reset(), submits a complete
    OpenSCAD script on step(), and receives a grading score (0.0-1.0).
    Episodes are single-step: reset -> step -> done.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._vision_config = get_config_from_env()
        self._current_task = None
        self._work_dir = tempfile.mkdtemp(prefix="openscad_env_")
        super().__init__(rubric=None)
        self._state = OpenSCADState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="",
            current_code="",
            score=0.0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenSCADObservation:
        """Reset the environment with a task."""
        task_id = kwargs.get("task_id", "basic_box")
        task = get_task(task_id)
        self._current_task = task

        self.rubric = OpenSCADRubric(task, self._vision_config)
        self._reset_rubric()

        self._state = OpenSCADState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_code="",
            score=0.0,
        )

        return OpenSCADObservation(
            task_description=task.description,
            task_id=task.task_id,
            available_tasks=list_tasks(),
            compile_success=False,
            compile_error="",
            dimensions={},
            volume=0.0,
            is_watertight=False,
            score=0.0,
            score_breakdown={},
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: OpenSCADAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OpenSCADObservation:
        """Execute the agent's OpenSCAD code, compile, measure, and grade."""
        if self._current_task is None:
            return OpenSCADObservation(
                task_description="",
                task_id="",
                available_tasks=list_tasks(),
                compile_success=False,
                compile_error="No task selected. Call reset() first.",
                dimensions={},
                volume=0.0,
                is_watertight=False,
                score=0.0,
                score_breakdown={},
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._state.current_code = action.code

        ep_id = self._state.episode_id
        scad_path = os.path.join(self._work_dir, f"{ep_id}.scad")
        stl_path = os.path.join(self._work_dir, f"{ep_id}.stl")

        compile_success, compile_error, compile_warnings = self._compile(
            action.code, scad_path, stl_path, timeout_s
        )

        dimensions, volume, surface_area, is_watertight, component_count = (
            {}, 0.0, 0.0, False, 0,
        )
        if compile_success:
            dimensions, volume, surface_area, is_watertight, component_count, analysis_error = (
                self._analyse(stl_path)
            )
            if analysis_error:
                compile_success = False
                compile_error = analysis_error

        observation = OpenSCADObservation(
            task_description=self._current_task.description,
            task_id=self._current_task.task_id,
            available_tasks=list_tasks(),
            compile_success=compile_success,
            compile_error=compile_error,
            compile_warnings=compile_warnings,
            dimensions=dimensions,
            volume=volume,
            surface_area=surface_area,
            is_watertight=is_watertight,
            component_count=component_count,
            score=0.0,
            score_breakdown={},
            done=True,
            reward=0.0,
        )

        object.__setattr__(observation, "_scad_path", scad_path)
        object.__setattr__(observation, "_stl_path", stl_path)
        object.__setattr__(observation, "_work_dir", self._work_dir)

        reward = self._apply_rubric(action, observation)
        observation.reward = reward
        observation.score = reward

        breakdown: dict = {}
        for name, child in self.rubric.named_children():
            if child.last_score is not None:
                breakdown[name] = round(child.last_score, 4)
        observation.score_breakdown = breakdown

        if hasattr(self.rubric, "vision") and self.rubric.vision.breakdown:
            observation.vision_score = self.rubric.vision.last_score or 0.0
            observation.vision_breakdown = self.rubric.vision.breakdown

        self._state.score = reward

        try:
            object.__delattr__(observation, "_scad_path")
            object.__delattr__(observation, "_stl_path")
            object.__delattr__(observation, "_work_dir")
        except AttributeError:
            pass
        for path in (scad_path, stl_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        return observation

    @property
    def state(self) -> OpenSCADState:
        return self._state

    def close(self) -> None:
        import shutil

        try:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile(
        code: str,
        scad_path: str,
        stl_path: str,
        timeout_s: Optional[float],
    ) -> tuple:
        """Write code to .scad file and run openscad CLI."""
        try:
            with open(scad_path, "w") as f:
                f.write(code)

            result = subprocess.run(
                ["openscad", "-o", stl_path, scad_path],
                capture_output=True,
                text=True,
                timeout=timeout_s or 30,
            )

            success = (
                result.returncode == 0
                and os.path.exists(stl_path)
                and os.path.getsize(stl_path) > 0
            )
            error = ""
            if not success:
                error = (
                    result.stderr.strip()
                    if result.stderr
                    else "Unknown compilation error"
                )

            # Extract WARNING lines from stderr even on success
            warnings = []
            if result.stderr:
                for line in result.stderr.splitlines():
                    stripped = line.strip()
                    if stripped.upper().startswith("WARNING"):
                        warnings.append(stripped)

            return success, error, warnings

        except subprocess.TimeoutExpired:
            return False, "OpenSCAD compilation timed out (>30s)", []
        except FileNotFoundError:
            return False, (
                "OpenSCAD binary not found. Install with: "
                "apt-get install openscad (Linux) or brew install openscad (macOS)"
            ), []
        except Exception as e:
            return False, f"Compilation error: {e}", []

    @staticmethod
    def _analyse(stl_path: str) -> tuple:
        """Load STL with trimesh and extract measurements."""
        try:
            import trimesh

            mesh = trimesh.load(stl_path)
            dimensions = {}
            volume = 0.0
            surface_area = 0.0
            is_watertight = False

            if hasattr(mesh, "bounding_box") and mesh.bounding_box is not None:
                extents = mesh.bounding_box.extents
                dimensions = {
                    "x": round(float(extents[0]), 3),
                    "y": round(float(extents[1]), 3),
                    "z": round(float(extents[2]), 3),
                }
            if mesh.is_volume:
                volume = round(float(mesh.volume), 3)
            if hasattr(mesh, "area"):
                surface_area = round(float(mesh.area), 3)
            is_watertight = bool(mesh.is_watertight)

            components = mesh.split() if hasattr(mesh, "split") else [mesh]
            component_count = len(components)

            return dimensions, volume, surface_area, is_watertight, component_count, None

        except Exception as e:
            return {}, 0.0, 0.0, False, 0, f"STL analysis failed: {e}"
