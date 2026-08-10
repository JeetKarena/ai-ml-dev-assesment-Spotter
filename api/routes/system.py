"""System routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/system")
def system_info():
    return {"status": "ready"}
