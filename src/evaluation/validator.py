"""Evaluation-time input validation helpers."""

from __future__ import annotations

import numpy as np


def validate_shapes(y_true, y_pred) -> None:
    """Assert that ground-truth and prediction arrays have the same length.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values from the model.

    Raises:
        ValueError: If shapes do not match.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {actual.shape} " f"but y_pred has shape {predicted.shape}."
        )


def validate_positive_predictions(y_pred) -> None:
    """Assert that every predicted rate is strictly positive.

    Args:
        y_pred: Predicted values from the model.

    Raises:
        ValueError: If any prediction is zero or negative.
    """
    predicted = np.asarray(y_pred, dtype=float)
    if (predicted <= 0).any():
        bad_count = int((predicted <= 0).sum())
        raise ValueError(
            f"{bad_count} prediction(s) are non-positive. "
            "All freight rates must be strictly greater than zero."
        )
