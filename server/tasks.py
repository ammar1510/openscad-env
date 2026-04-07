"""
Task definitions for the OpenSCAD environment.

Each task specifies what the agent must build, target dimensions/volume,
and tolerances for grading.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    target_dimensions: Dict[str, float]  # Expected bounding box {x, y, z} in mm
    target_volume: Optional[float] = None  # Expected volume in mm^3
    dimension_tolerance: float = 0.1  # Fraction of target allowed as error
    volume_tolerance: float = 0.15  # Fraction of target volume allowed as error
    expected_components: int = 1  # Number of connected mesh components
    hints: str = ""


TASK_REGISTRY: Dict[str, Task] = {}


def _register(*tasks: Task) -> None:
    for t in tasks:
        TASK_REGISTRY[t.task_id] = t


_register(
    Task(
        task_id="basic_box",
        description=(
            "Create a rectangular box with dimensions 30mm (width/x) by 20mm "
            "(depth/y) by 10mm (height/z). The box should be a simple solid "
            "rectangular prism."
        ),
        difficulty="easy",
        target_dimensions={"x": 30.0, "y": 20.0, "z": 10.0},
        target_volume=6000.0,
        dimension_tolerance=0.05,
        volume_tolerance=0.10,
        hints=(
            "In OpenSCAD, cube([x,y,z]) creates a box. "
            "Example: cube([30, 20, 10]);"
        ),
    ),
    Task(
        task_id="hollow_cylinder",
        description=(
            "Create a hollow cylinder (tube) with outer radius 15mm, inner "
            "radius 10mm, and height 25mm. The tube should be centered on the "
            "origin. Use a high polygon count ($fn=64 or higher) for smooth curves."
        ),
        difficulty="medium",
        target_dimensions={"x": 30.0, "y": 30.0, "z": 25.0},
        target_volume=math.pi * (15**2 - 10**2) * 25,  # ~9817.5
        dimension_tolerance=0.05,
        volume_tolerance=0.15,
        hints=(
            "Use difference() to subtract one cylinder from another. The inner "
            "cylinder should be slightly taller to ensure a clean cut."
        ),
    ),
    Task(
        task_id="stacking_blocks",
        description=(
            "Create a set of 3 stacking blocks centered on top of each other: "
            "a large block (20x20x20mm) at the bottom, a medium block "
            "(15x15x15mm) centered on top of the large one, and a small block "
            "(10x10x10mm) centered on top of the medium one. All blocks should "
            "be touching (no gaps between them)."
        ),
        difficulty="medium",
        target_dimensions={"x": 20.0, "y": 20.0, "z": 45.0},  # 20+15+10
        target_volume=12375.0,  # 20^3 + 15^3 + 10^3
        dimension_tolerance=0.08,
        volume_tolerance=0.12,
        hints=(
            "Use translate([x,y,z]) to position blocks. Center blocks by "
            "offsetting x and y by half the difference in width."
        ),
    ),
    Task(
        task_id="phone_stand",
        description=(
            "Design a phone stand that holds a phone at approximately 60 degrees "
            "from horizontal. Requirements:\n"
            "- The phone slot/groove should be approximately 8mm wide\n"
            "- The base should be at least 40mm long for stability\n"
            "- The stand should be no taller than 80mm\n"
            "- The stand should be a single solid piece\n"
            "Use your creativity for the overall shape, but ensure it meets "
            "the dimensional requirements."
        ),
        difficulty="hard",
        target_dimensions={"x": 40.0, "y": 30.0, "z": 60.0},
        target_volume=None,
        dimension_tolerance=0.25,
        volume_tolerance=0.0,
        hints=(
            "Consider using rotate(), difference(), and hull() for organic "
            "shapes. A wedge shape with a slot cut into it works well."
        ),
    ),
)


def get_task(task_id: str) -> Task:
    """Get a task by ID. Raises ValueError if not found."""
    if task_id not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {available}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    """Return all available task IDs."""
    return list(TASK_REGISTRY.keys())
