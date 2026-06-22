"""Single-process launcher for single-service deployments.

The api (FastAPI token/REST side-car) and worker (LiveKit agent) are normally two
separate Railway services per the README — but each gets its own ephemeral local
disk, so the SQLite file the worker writes to is invisible to the api service's
reads, and `/api/summary/...` 404s forever. Running both in this one process means
they share this container's filesystem and therefore the same `data/voice_agent.db`.
"""

from __future__ import annotations

import os
import subprocess
import sys

from livekit.agents import WorkerOptions, cli

from app.agent import entrypoint


def main() -> None:
    port = os.environ.get("PORT", "8000")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", port]
    )
    try:
        # Reverted to 1 idle process — too costly on a limited/trial plan. See the
        # matching comment in app/agent.py's __main__ block.
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1))
    finally:
        api_proc.terminate()
        api_proc.wait()


if __name__ == "__main__":
    main()
