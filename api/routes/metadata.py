"""Metadata route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/metadata")
def metadata():
    return {"model": "Spotter", "version": "0.1.0"}
