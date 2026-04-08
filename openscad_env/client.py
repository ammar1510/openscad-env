"""
OpenSCAD Environment Client.

WebSocket client for connecting to an OpenSCAD environment server.

Example:
    >>> with OpenSCADEnv(base_url="http://localhost:8000").sync() as client:
    ...     result = client.reset(task_id="basic_box")
    ...     print(result.observation.task_description)
    ...
    ...     result = client.step(OpenSCADAction(code="cube([30, 20, 10]);"))
    ...     print(result.observation.score)
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import OpenSCADAction, OpenSCADObservation, OpenSCADState


class OpenSCADEnv(EnvClient[OpenSCADAction, OpenSCADObservation, OpenSCADState]):
    """
    Client for the OpenSCAD 3D Modeling Environment.

    Example:
        >>> with OpenSCADEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(task_id="basic_box")
        ...     result = client.step(OpenSCADAction(code="cube([30, 20, 10]);"))
        ...     print(result.observation.score)

    Example with Docker:
        >>> client = OpenSCADEnv.from_docker_image("openscad-env:latest")
        >>> try:
        ...     result = client.reset(task_id="basic_box")
        ...     result = client.step(OpenSCADAction(code="cube([30, 20, 10]);"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: OpenSCADAction) -> Dict:
        """Convert Action to JSON payload for step message."""
        return {"code": action.code}

    def _parse_result(self, payload: Dict) -> StepResult[OpenSCADObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = OpenSCADObservation(
            task_description=obs_data.get("task_description", ""),
            task_id=obs_data.get("task_id", ""),
            available_tasks=obs_data.get("available_tasks", []),
            compile_success=obs_data.get("compile_success", False),
            compile_error=obs_data.get("compile_error", ""),
            dimensions=obs_data.get("dimensions", {}),
            volume=obs_data.get("volume", 0.0),
            is_watertight=obs_data.get("is_watertight", False),
            score=obs_data.get("score", 0.0),
            score_breakdown=obs_data.get("score_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OpenSCADState:
        """Parse server response into State object."""
        return OpenSCADState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            current_code=payload.get("current_code", ""),
            score=payload.get("score", 0.0),
        )
