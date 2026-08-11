"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse
from api.services.model_service import is_model_available

router = APIRouter(tags=["Operations"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health_check() -> HealthResponse:
    """Return service readiness.

    Checks whether the model artifact exists on disk.  Does not load the
    model — intentionally fast for load-balancer probes.
    """
    return HealthResponse(status="ok" if is_model_available() else "model_unavailable")
