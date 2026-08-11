"""Feature engineering utilities."""

from __future__ import annotations

from src.features.pipeline import make_model_features, build_preprocessor, TARGET_COLUMN
from src.features.temporal import make_temporal_features
from src.features.route import route_features
from src.features.encoding import SafeOrdinalEncoder

__all__ = [
    "make_model_features",
    "build_preprocessor",
    "TARGET_COLUMN",
    "make_temporal_features",
    "route_features",
    "SafeOrdinalEncoder",
]
