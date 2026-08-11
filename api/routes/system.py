"""System diagnostic route."""

from __future__ import annotations

import platform
import sys

from fastapi import APIRouter

router = APIRouter(tags=["Operations"])


@router.get("/system", summary="System diagnostics")
def system_info() -> dict:
    """Return lightweight runtime diagnostics.

    Useful for confirming the Python version and platform inside a container
    without attaching a shell.
    """
    return {
        "status": "ready",
        "python": sys.version,
        "platform": platform.platform(),
    }
