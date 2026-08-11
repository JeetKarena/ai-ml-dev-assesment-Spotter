"""Custom API exception hierarchy.

Raising domain-specific exceptions lets the global exception handler in
app.py return structured JSON errors rather than raw Python tracebacks.
"""

from __future__ import annotations


class SpotterError(Exception):
    """Base exception for all Spotter API errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class ModelNotReadyError(SpotterError):
    """Raised when the model artifact has not been loaded yet."""

    status_code = 503
    detail = "Model artifact is unavailable. Run training first."


class PredictionError(SpotterError):
    """Raised when inference fails due to bad input or internal error."""

    status_code = 422
    detail = "Prediction failed. Check your input payload."


class ArtifactNotFoundError(SpotterError):
    """Raised when a required artifact file is missing from disk."""

    status_code = 503
    detail = "Required artifact not found on disk."
