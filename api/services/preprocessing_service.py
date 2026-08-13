"""Input-feature preprocessing service for the API layer.

The API receives a raw JSON payload.  Before constructing features this
module normalises pickup, delivery, and equipment to their canonical
training-set casing so the model's OrdinalEncoder can always find them.
"""

from __future__ import annotations

import pandas as pd

from api.core.exceptions import PredictionError
from api.services.location_service import normalize_equipment, normalize_location
from src.features.pipeline import make_model_features


def preprocess(payload: dict) -> pd.DataFrame:
    """Normalise location/equipment names and build a feature DataFrame.

    Normalisation steps (applied before feature engineering):
    1. ``pickup``   — case-fold + whitespace-collapse → canonical Title Case
    2. ``delivery`` — same treatment
    3. ``equipment`` — case-fold → canonical form ("dry van" → "Dry Van")

    Args:
        payload: Raw request dict matching the FreightLoad schema.

    Returns:
        Single-row DataFrame ready for ``model.predict()``.

    Raises:
        PredictionError: If location/equipment is unrecognised or the date
            is invalid.
    """
    try:
        normalised = dict(payload)  # shallow copy — do not mutate the original
        normalised["pickup"] = normalize_location(payload["pickup"], field="pickup")
        normalised["delivery"] = normalize_location(payload["delivery"], field="delivery")
        normalised["equipment"] = normalize_equipment(payload["equipment"])
        frame = pd.DataFrame([normalised])
        return make_model_features(frame)
    except ValueError as exc:
        raise PredictionError(str(exc)) from exc
    except KeyError as exc:
        raise PredictionError(f"Missing required field: {exc}") from exc
