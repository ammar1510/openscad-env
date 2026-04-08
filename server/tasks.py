"""
Task definitions for the OpenSCAD environment.

Each task specifies what the agent must build, target dimensions/volume,
and tolerances for grading.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    target_dimensions: Dict[str, float]  # Expected bounding box {x, y, z} in mm
    target_volume: Optional[float] = None  # Expected volume in mm^3
    dimension_tolerance: float = 0.1  # Fraction of target allowed as error
    volume_tolerance: float = 0.15  # Fraction of target volume allowed as error
    target_surface_area: Optional[float] = None  # Expected surface area in mm^2
    surface_area_tolerance: float = 0.15  # Fraction of target allowed as error
    # Cross-section profile: list of (relative_height 0-1, expected_area mm^2)
    target_cross_sections: Optional[List[Tuple[float, float]]] = None
    cross_section_tolerance: float = 0.20  # Fraction of target area allowed as error
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
        target_surface_area=2200.0,  # 2*(30*20 + 30*10 + 20*10)
        dimension_tolerance=0.05,
        volume_tolerance=0.10,
        # Constant 30*20 = 600 at every height
        target_cross_sections=[
            (0.1, 600.0), (0.3, 600.0), (0.5, 600.0),
            (0.7, 600.0), (0.9, 600.0),
        ],
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
        # outer lateral + inner lateral + 2 annular rings
        target_surface_area=math.pi * (2 * 15 * 25 + 2 * 10 * 25 + 2 * (15**2 - 10**2)),  # ~4712.4
        dimension_tolerance=0.05,
        volume_tolerance=0.15,
        # Annular cross-section pi*(15^2-10^2) ≈ 392.7 at every height
        target_cross_sections=[
            (0.1, math.pi * (15**2 - 10**2)),
            (0.3, math.pi * (15**2 - 10**2)),
            (0.5, math.pi * (15**2 - 10**2)),
            (0.7, math.pi * (15**2 - 10**2)),
            (0.9, math.pi * (15**2 - 10**2)),
        ],
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
        # Full SA of each cube minus 2x each contact patch (removed from both surfaces)
        target_surface_area=3700.0,  # 6*20^2 + 6*15^2 + 6*10^2 - 2*15^2 - 2*10^2
        dimension_tolerance=0.08,
        volume_tolerance=0.12,
        # Stepped profile: total height=45. Bottom 20mm → 400, middle 15mm → 225, top 10mm → 100
        target_cross_sections=[
            (0.1, 400.0),   # 4.5mm  → in bottom block (20x20)
            (0.3, 400.0),   # 13.5mm → in bottom block (20x20)
            (0.5, 225.0),   # 22.5mm → in middle block (15x15)
            (0.7, 225.0),   # 31.5mm → in middle block (15x15)
            (0.9, 100.0),   # 40.5mm → in top block (10x10)
        ],
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
