"""
OpenSCAD 3D Modeling Environment for OpenEnv.

An RL environment where an LLM agent writes OpenSCAD code to create
3D objects matching given specifications.

Example:
    >>> from openscad_env import OpenSCADEnv, OpenSCADAction
    >>>
    >>> with OpenSCADEnv(base_url="http://localhost:8000").sync() as client:
    ...     result = client.reset(task_id="basic_box")
    ...     print(result.observation.task_description)
    ...
    ...     result = client.step(OpenSCADAction(code="cube([30, 20, 10]);"))
    ...     print(result.observation.score)
"""

from .client import OpenSCADEnv
from .models import OpenSCADAction, OpenSCADObservation, OpenSCADState

__all__ = ["OpenSCADAction", "OpenSCADObservation", "OpenSCADState", "OpenSCADEnv"]
