"""
Data models for the OpenSCAD 3D Modeling Environment.

Defines the Action, Observation, and State types for an environment
where an LLM agent writes OpenSCAD code to create 3D objects.
"""

from typing import Dict, List

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class OpenSCADAction(Action):
    """Agent's action: a complete OpenSCAD script."""

    code: str = Field(
        ..., description="Complete OpenSCAD script to compile and evaluate"
    )


class OpenSCADObservation(Observation):
    """Observation returned after reset() and step()."""

    # Task info
    task_description: str = Field(
        default="", description="Natural language description of what to build"
    )
    task_id: str = Field(default="", description="Task identifier, e.g. 'basic_box'")
    available_tasks: List[str] = Field(
        default_factory=list, description="List of all available task IDs"
    )

    # Compilation results
    compile_success: bool = Field(
        default=False,
        description="Whether OpenSCAD compiled the code successfully",
    )
    compile_error: str = Field(
        default="", description="Compilation error message, empty if success"
    )
    compile_warnings: List[str] = Field(
        default_factory=list,
        description="Warning lines from OpenSCAD stderr (even on successful compile)",
    )

    # Geometric measurements
    dimensions: Dict[str, float] = Field(
        default_factory=dict,
        description="Bounding box dimensions {x, y, z} in mm",
    )
    volume: float = Field(default=0.0, description="Mesh volume in mm^3")
    surface_area: float = Field(default=0.0, description="Mesh surface area in mm^2")
    is_watertight: bool = Field(
        default=False, description="Whether the mesh is a valid watertight solid"
    )
    component_count: int = Field(
        default=0, description="Number of connected mesh components"
    )

    # Grading
    score: float = Field(default=0.0, description="Composite grading score 0.0-1.0")
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-component score breakdown"
    )
    vision_score: float = Field(
        default=0.0, description="Vision LLM judge score 0.0-1.0 (0 if disabled)"
    )
    vision_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion vision judge breakdown (shape, proportions, completeness)",
    )


class OpenSCADState(State):
    """Internal environment state."""

    task_id: str = Field(default="", description="Active task identifier")
    current_code: str = Field(
        default="", description="The OpenSCAD code submitted by the agent"
    )
    score: float = Field(default=0.0, description="Score achieved")
