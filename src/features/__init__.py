"""Feature engineering utilities."""

from __future__ import annotations

from src.features.encoding import SafeOrdinalEncoder
from src.features.pipeline import TARGET_COLUMN, build_preprocessor, make_model_features
from src.features.route import route_features
from src.features.temporal import make_temporal_features

__all__ = [
    "make_model_features",
    "build_preprocessor",
    "TARGET_COLUMN",
    "make_temporal_features",
    "route_features",
    "SafeOrdinalEncoder",
]
