"""Metrics used for the chronological holdout evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return interpretable regression metrics with JSON-safe values."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mape": round(
            float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100), 4
        ),
        "median_ae": round(float(np.median(np.abs(actual - predicted))), 4),
        "n_samples": int(actual.size),
    }
