"""FastAPI lifespan lifecycle — load the model once on startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.services.model_service import load_model, is_model_available

logger = logging.getLogger("spotter.api.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the API.

    On startup the model artifact is eagerly loaded into memory so the
    first request does not pay the deserialization cost.  On shutdown the
    reference is released.
    """
    if is_model_available():
        try:
            load_model()  # warms the singleton cache in model_service
            logger.info("Model artifact loaded successfully on startup.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model could not be pre-loaded: %s", exc)
    else:
        logger.warning("Model artifact not found — /predict will return 503 until it is available.")

    app.state.ready = True
    yield
    app.state.ready = False
    logger.info("API shutdown complete.")
