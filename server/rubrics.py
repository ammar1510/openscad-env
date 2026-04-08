"""
OpenSCAD rubrics for reward computation.

Provides composable Rubric subclasses for each scoring component
(compilation, dimensions, volume, vision judge) wired together
with WeightedSum.
"""

from typing import Any, Dict, Optional

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

from . import vision_judge
from .tasks import Task

# Hackathon requires scores strictly between 0 and 1 (exclusive).
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    """Clamp a score to the open interval (0, 1)."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


# ---------------------------------------------------------------------------
# Component rubrics
# ---------------------------------------------------------------------------


class CompilationRubric(Rubric):
    """Binary: 1.0 if compilation succeeded, 0.0 otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        return _clamp(1.0 if observation.compile_success else 0.0)


class WatertightRubric(Rubric):
    """Binary: 1.0 if mesh is watertight (closed solid), 0.0 otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success:
            return _clamp(0.0)
        return _clamp(1.0 if observation.is_watertight else 0.0)


class ComponentCountRubric(Rubric):
    """Scores how well the component count matches the task expectation.

    1.0 for exact match, linear penalty for deviation, floors at 0.0.
    """

    def __init__(self, task: Task):
        super().__init__()
        self._expected = task.expected_components

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success:
            return _clamp(0.0)
        actual = getattr(observation, "component_count", 0)
        if actual == 0:
            return _clamp(0.0)
        diff = abs(actual - self._expected)
        return _clamp(1.0 - diff / max(self._expected, 1))


class DimensionsRubric(Rubric):
    """Per-axis dimensional accuracy against the task target."""

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success or not observation.dimensions:
            return _clamp(0.0)

        scores = []
        for axis, target_val in self._task.target_dimensions.items():
            actual_val = observation.dimensions.get(axis, 0.0)
            if target_val > 0:
                error = abs(actual_val - target_val) / target_val
                scores.append(
                    max(0.0, 1.0 - error / self._task.dimension_tolerance)
                )

        return _clamp(sum(scores) / len(scores) if scores else 0.0)


class VolumeRubric(Rubric):
    """Volume accuracy against the task target."""

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        tv = self._task.target_volume
        if tv is None or tv == 0:
            return _clamp(1.0)
        if not observation.compile_success:
            return _clamp(0.0)
        if self._task.volume_tolerance <= 0:
            return _clamp(1.0)

        error = abs(observation.volume - tv) / tv
        return _clamp(1.0 - error / self._task.volume_tolerance)


class SurfaceAreaRubric(Rubric):
    """Surface area accuracy against the task target."""

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        target = self._task.target_surface_area
        if target is None or target == 0:
            return _clamp(1.0)
        if not observation.compile_success:
            return _clamp(0.0)
        tol = self._task.surface_area_tolerance
        if tol <= 0:
            return _clamp(1.0)

        actual = getattr(observation, "surface_area", 0.0)
        error = abs(actual - target) / target
        return _clamp(1.0 - error / tol)


class CrossSectionRubric(Rubric):
    """Compares cross-sectional area profile at specified heights.

    Slices the mesh horizontally at each target height and scores
    how well each slice's area matches the expected value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        targets = self._task.target_cross_sections
        if not targets:
            return 1.0
        if not observation.compile_success:
            return 0.0

        stl_path = getattr(observation, "_stl_path", None)
        if not stl_path:
            return 0.0

        try:
            import numpy as np
            import trimesh

            mesh = trimesh.load(stl_path)
        except Exception:
            return 0.0

        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        z_min, z_max = float(bbox_min[2]), float(bbox_max[2])
        z_range = z_max - z_min
        if z_range <= 0:
            return 0.0

        tol = self._task.cross_section_tolerance
        scores = []
        for rel_height, expected_area in targets:
            z = z_min + rel_height * z_range
            try:
                section = mesh.section(
                    plane_origin=[0, 0, z],
                    plane_normal=[0, 0, 1],
                )
                if section is None:
                    scores.append(0.0)
                    continue
                planar, _ = section.to_planar()
                actual_area = float(planar.area)
            except Exception:
                scores.append(0.0)
                continue

            if expected_area <= 0:
                scores.append(1.0 if actual_area <= 0 else 0.0)
                continue

            error = abs(actual_area - expected_area) / expected_area
            scores.append(max(0.0, 1.0 - error / tol))

        return _clamp(sum(scores) / len(scores) if scores else 0.0)


class CodeParsabilityRubric(Rubric):
    """Penalises OpenSCAD compiler warnings (even when compilation succeeds).

    1.0 for zero warnings, -0.25 per warning, floored at 0.0.
    """

    _PENALTY_PER_WARNING = 0.25

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success:
            return _clamp(0.0)
        warnings = getattr(observation, "compile_warnings", [])
        return _clamp(1.0 - len(warnings) * self._PENALTY_PER_WARNING)


class VisionJudgeRubric(Rubric):
    """Renders the .scad file and scores via a vision LLM."""

    def __init__(self, task: Task, vision_config: Dict[str, str]):
        super().__init__()
        self._task = task
        self._config = vision_config
        self.breakdown: Dict[str, float] = {}

    def forward(self, action: Any, observation: Any) -> float:
        self.breakdown = {}
        if not observation.compile_success:
            return _clamp(0.0)

        scad_path = getattr(observation, "_scad_path", None)
        work_dir = getattr(observation, "_work_dir", None)
        if not scad_path or not work_dir:
            return _clamp(0.0)

        rendered = vision_judge.render_views(scad_path, work_dir)
        if not rendered:
            return _clamp(0.0)

        import os

        try:
            score, self.breakdown = vision_judge.judge(
                task=self._task,
                rendered_views=rendered,
                api_base=self._config["api_base"],
                model=self._config["model"],
                api_key=self._config["api_key"],
            )
            return _clamp(score)
        finally:
            for png_path in rendered.values():
                try:
                    os.remove(png_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Top-level composite rubric
# ---------------------------------------------------------------------------

_WEIGHTS_WITH_VISION = {
    "compilation": 0.05,
    "watertight": 0.05,
    "component_count": 0.05,
    "cross_section": 0.10,
    "code_parsability": 0.05,
    "dimensions": 0.15,
    "volume": 0.05,
    "surface_area": 0.05,
    "vision": 0.45,
}
_WEIGHTS_GEOMETRIC = {
    "compilation": 0.05,
    "watertight": 0.05,
    "component_count": 0.10,
    "cross_section": 0.15,
    "code_parsability": 0.05,
    "dimensions": 0.25,
    "volume": 0.20,
    "surface_area": 0.15,
}


class OpenSCADRubric(Rubric):
    """Top-level rubric composing geometric + optional vision scoring."""

    def __init__(
        self,
        task: Task,
        vision_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._task = task
        self._vision_enabled = vision_config is not None

        self.compilation = CompilationRubric()
        self.watertight = WatertightRubric()
        self.component_count = ComponentCountRubric(task)
        self.cross_section = CrossSectionRubric(task)
        self.code_parsability = CodeParsabilityRubric()
        self.dimensions = DimensionsRubric(task)
        self.volume = VolumeRubric(task)
        self.surface_area = SurfaceAreaRubric(task)

        if self._vision_enabled:
            self.vision = VisionJudgeRubric(task, vision_config)
            w = _WEIGHTS_WITH_VISION
            weights = [
                w["compilation"], w["watertight"], w["component_count"],
                w["cross_section"], w["code_parsability"], w["dimensions"],
                w["volume"], w["surface_area"], w["vision"],
            ]
            rubrics = [
                self.compilation, self.watertight, self.component_count,
                self.cross_section, self.code_parsability, self.dimensions,
                self.volume, self.surface_area, self.vision,
            ]
        else:
            w = _WEIGHTS_GEOMETRIC
            weights = [
                w["compilation"], w["watertight"], w["component_count"],
                w["cross_section"], w["code_parsability"], w["dimensions"],
                w["volume"], w["surface_area"],
            ]
            rubrics = [
                self.compilation, self.watertight, self.component_count,
                self.cross_section, self.code_parsability, self.dimensions,
                self.volume, self.surface_area,
            ]

        scorer = WeightedSum(rubrics, weights)
        object.__setattr__(self, "_scorer", scorer)

    def forward(self, action: Any, observation: Any) -> float:
        return _clamp(self._scorer(action, observation))

    def reset(self) -> None:
        self.compilation.last_score = None
        self.watertight.last_score = None
        self.component_count.last_score = None
        self.cross_section.last_score = None
        self.code_parsability.last_score = None
        self.dimensions.last_score = None
        self.volume.last_score = None
        self.surface_area.last_score = None
        if self._vision_enabled:
            self.vision.last_score = None
            self.vision.breakdown = {}
