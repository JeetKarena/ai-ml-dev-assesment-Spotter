"""Metadata route — exposes static model and version information."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import MetadataResponse
from api.config import API_VERSION, MODEL_PATH

router = APIRouter(tags=["Operations"])


@router.get("/metadata", response_model=MetadataResponse, summary="Model metadata")
def metadata() -> MetadataResponse:
    """Return static metadata about the deployed model artifact."""
    return MetadataResponse(
        model="HistGradientBoostingRegressor",
        version=API_VERSION,
        artifact=str(MODEL_PATH),
    )
