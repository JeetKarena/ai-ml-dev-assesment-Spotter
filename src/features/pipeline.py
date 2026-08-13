"""Leakage-safe feature construction and model preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features.encoding import SafeOrdinalEncoder
from src.features.route import route_features
from src.features.temporal import make_temporal_features

TARGET_COLUMN = "posted_rate"
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment", "route", "distance_bucket", "equipment_x_distance"]
NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "is_q4",
    "days_to_christmas",
    "days_to_thanksgiving",
    "is_holiday_week",
    "distance_per_weight",
    "weight_x_distance",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features available at quote time.

    No target-derived aggregates are used, so this function behaves identically
    for chronological holdout, final validation, and December scenarios.
    """
    required = {"pickup", "delivery", "equipment", "date", "distance", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    result = frame.copy()

    # Non-positive truck weights are physically implausible. Mark them missing
    # and let the fitted median imputer learn only from each training fold.
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    result.loc[result["weight"] <= 0, "weight"] = np.nan

    # Delegate temporal feature creation
    result = make_temporal_features(result, date_column="date")

    # Delegate geographic and interaction feature creation
    result = route_features(result)

    # The fixed December scenario deliberately provides fewer operational
    # fields than the regular quote feed. Missing numeric inputs are retained
    # as NaN so the model's fitted median imputer handles them consistently.
    for column in NUMERIC_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    expected = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    return result.loc[:, expected]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing fitted strictly inside the model pipeline."""
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", SafeOrdinalEncoder(unknown_value=-1)),
        ]
    )
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("numeric", numeric, NUMERIC_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
