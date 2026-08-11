"""Model artifact loading with in-process singleton caching."""

from __future__ import annotations

import joblib

from api.config import MODEL_PATH

# Module-level singleton — loaded once, reused for every request
_model = None


def load_model():
    """Return the cached model, deserializing from disk on first call.

    Returns:
        The fitted scikit-learn pipeline.

    Raises:
        FileNotFoundError: When the artifact is absent from MODEL_PATH.
    """
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    return _model


def is_model_available() -> bool:
    """Return True when the primary model artifact exists on disk."""
    return MODEL_PATH.is_file()


def clear_cache() -> None:
    """Reset the singleton — used only in tests."""
    global _model  # noqa: PLW0603
    _model = None
