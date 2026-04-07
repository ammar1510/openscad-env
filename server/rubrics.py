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


# ---------------------------------------------------------------------------
# Component rubrics
# ---------------------------------------------------------------------------


class CompilationRubric(Rubric):
    """Binary: 1.0 if compilation succeeded, 0.0 otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        return 1.0 if observation.compile_success else 0.0


class WatertightRubric(Rubric):
    """Binary: 1.0 if mesh is watertight (closed solid), 0.0 otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success:
            return 0.0
        return 1.0 if observation.is_watertight else 0.0


class ComponentCountRubric(Rubric):
    """Scores how well the component count matches the task expectation.

    1.0 for exact match, linear penalty for deviation, floors at 0.0.
    """

    def __init__(self, task: Task):
        super().__init__()
        self._expected = task.expected_components

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success:
            return 0.0
        actual = getattr(observation, "component_count", 0)
        if actual == 0:
            return 0.0
        diff = abs(actual - self._expected)
        return max(0.0, 1.0 - diff / max(self._expected, 1))


class DimensionsRubric(Rubric):
    """Per-axis dimensional accuracy against the task target."""

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        if not observation.compile_success or not observation.dimensions:
            return 0.0

        scores = []
        for axis, target_val in self._task.target_dimensions.items():
            actual_val = observation.dimensions.get(axis, 0.0)
            if target_val > 0:
                error = abs(actual_val - target_val) / target_val
                scores.append(
                    max(0.0, 1.0 - error / self._task.dimension_tolerance)
                )

        return sum(scores) / len(scores) if scores else 0.0


class VolumeRubric(Rubric):
    """Volume accuracy against the task target."""

    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def forward(self, action: Any, observation: Any) -> float:
        tv = self._task.target_volume
        if tv is None or tv == 0:
            return 1.0
        if not observation.compile_success:
            return 0.0
        if self._task.volume_tolerance <= 0:
            return 1.0

        error = abs(observation.volume - tv) / tv
        return max(0.0, 1.0 - error / self._task.volume_tolerance)


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
            return 0.0

        scad_path = getattr(observation, "_scad_path", None)
        work_dir = getattr(observation, "_work_dir", None)
        if not scad_path or not work_dir:
            return 0.0

        rendered = vision_judge.render_views(scad_path, work_dir)
        if not rendered:
            return 0.0

        import os

        try:
            score, self.breakdown = vision_judge.judge(
                task=self._task,
                rendered_views=rendered,
                api_base=self._config["api_base"],
                model=self._config["model"],
                api_key=self._config["api_key"],
            )
            return score
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
    "compilation": 0.10,
    "watertight": 0.05,
    "component_count": 0.05,
    "dimensions": 0.25,
    "volume": 0.15,
    "vision": 0.40,
}
_WEIGHTS_GEOMETRIC = {
    "compilation": 0.15,
    "watertight": 0.05,
    "component_count": 0.10,
    "dimensions": 0.40,
    "volume": 0.30,
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
        self.dimensions = DimensionsRubric(task)
        self.volume = VolumeRubric(task)

        if self._vision_enabled:
            self.vision = VisionJudgeRubric(task, vision_config)
            w = _WEIGHTS_WITH_VISION
            weights = [
                w["compilation"], w["watertight"], w["component_count"],
                w["dimensions"], w["volume"], w["vision"],
            ]
            rubrics = [
                self.compilation, self.watertight, self.component_count,
                self.dimensions, self.volume, self.vision,
            ]
        else:
            w = _WEIGHTS_GEOMETRIC
            weights = [
                w["compilation"], w["watertight"], w["component_count"],
                w["dimensions"], w["volume"],
            ]
            rubrics = [
                self.compilation, self.watertight, self.component_count,
                self.dimensions, self.volume,
            ]

        scorer = WeightedSum(rubrics, weights)
        object.__setattr__(self, "_scorer", scorer)

    def forward(self, action: Any, observation: Any) -> float:
        return self._scorer(action, observation)

    def reset(self) -> None:
        self.compilation.last_score = None
        self.watertight.last_score = None
        self.component_count.last_score = None
        self.dimensions.last_score = None
        self.volume.last_score = None
        if self._vision_enabled:
            self.vision.last_score = None
            self.vision.breakdown = {}
