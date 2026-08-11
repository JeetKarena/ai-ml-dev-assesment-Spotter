"""API middleware — logging and error-handling hooks."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.exceptions import SpotterError

logger = logging.getLogger("spotter.api.middleware")


def add_app_middleware(app: FastAPI) -> FastAPI:
    """Register all middleware and global exception handlers on *app*.

    Args:
        app: The FastAPI application instance.

    Returns:
        The same application instance with middleware attached.
    """
    # ------------------------------------------------------------------
    # CORS — lock down in production; permissive here for assessment ease
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request timing — adds X-Process-Time header to every response
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        logger.debug("%s %s — %.4fs", request.method, request.url.path, elapsed)
        return response

    # ------------------------------------------------------------------
    # Domain exception handler — maps SpotterError → structured JSON
    # ------------------------------------------------------------------
    @app.exception_handler(SpotterError)
    async def spotter_exception_handler(request: Request, exc: SpotterError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app
