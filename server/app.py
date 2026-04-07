"""
FastAPI application for the OpenSCAD 3D Modeling Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import OpenSCADAction, OpenSCADObservation
    from .openscad_environment import OpenSCADEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app

    from models import OpenSCADAction, OpenSCADObservation
    from server.openscad_environment import OpenSCADEnvironment


app = create_app(
    OpenSCADEnvironment,
    OpenSCADAction,
    OpenSCADObservation,
    env_name="openscad_env",
    max_concurrent_envs=4,
)


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
