"""Prediction-service orchestration for the API."""

from __future__ import annotations

from api.core.exceptions import ModelNotReadyError, PredictionError
from api.services.model_service import load_model
from api.services.preprocessing_service import preprocess


def predict(payload: dict) -> float:
    """End-to-end prediction: preprocess payload → model inference → clip.

    Args:
        payload: Raw request dict matching the FreightLoad schema.

    Returns:
        Positive predicted freight rate in USD, rounded to two decimal places.

    Raises:
        ModelNotReadyError: If the model artifact is not available.
        PredictionError: If inference fails for any reason.
    """
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise ModelNotReadyError() from exc

    features = preprocess(payload)

    try:
        raw = float(model.predict(features)[0])
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Model inference failed: {exc}") from exc

    # Rates must always be positive — clip defensively but never silently log
    rate = max(raw, 0.01)
    return round(rate, 2)
