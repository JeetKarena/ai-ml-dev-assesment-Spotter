"""FastAPI lifespan lifecycle hooks."""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the API."""
    app.state.ready = True
    yield
    app.state.ready = False
