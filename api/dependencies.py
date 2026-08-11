"""FastAPI dependency injection providers.

Each function is a FastAPI Depends()-compatible callable that resolves a
shared resource.  Using dependency injection keeps route handlers thin and
makes the service easy to test with overrides.
"""

from __future__ import annotations

from api.services.model_service import load_model
from api.core.exceptions import ModelNotReadyError


def get_model():
    """FastAPI dependency that resolves the singleton model artifact.

    Raises:
        ModelNotReadyError: When the model artifact is not present on disk.

    Returns:
        The deserialized scikit-learn pipeline.
    """
    try:
        return load_model()
    except FileNotFoundError as exc:
        raise ModelNotReadyError() from exc
