"""FastAPI application factory for the Spotter freight-rate inference service."""

from __future__ import annotations

from fastapi import FastAPI

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.lifespan import lifespan
from api.middleware import add_app_middleware
from api.routes.health import router as health_router
from api.routes.locations import router as locations_router
from api.routes.metadata import router as metadata_router
from api.routes.prediction import router as prediction_router
from api.routes.system import router as system_router


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    add_app_middleware(app)
    app.include_router(health_router)
    app.include_router(prediction_router)
    app.include_router(locations_router)
    app.include_router(metadata_router)
    app.include_router(system_router)
    return app


app = create_app()
