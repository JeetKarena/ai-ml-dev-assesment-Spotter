"""HistGradientBoosting model implementation inheriting from BaseModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.models.base import BaseModel
from src.features.pipeline import build_preprocessor
from src.core.settings import settings


class HistGradientModel(BaseModel):
    """Wrapper class around Scikit-Learn's HistGradientBoostingRegressor."""

    def __init__(self, **kwargs) -> None:
        cfg = settings.get("model", default={}) or {}
        # Merge configuration defaults with explicitly passed keyword arguments
        self.params = {
            "learning_rate": float(kwargs.get("learning_rate", cfg.get("learning_rate", 0.04))),
            "max_iter": int(kwargs.get("max_iter", cfg.get("max_iter", 250))),
            "max_leaf_nodes": int(kwargs.get("max_leaf_nodes", cfg.get("max_leaf_nodes", 12))),
            "min_samples_leaf": int(kwargs.get("min_samples_leaf", cfg.get("min_samples_leaf", 80))),
            "l2_regularization": float(kwargs.get("l2_regularization", cfg.get("l2_regularization", 20.0))),
            "early_stopping": kwargs.get("early_stopping", False),
            "random_state": int(kwargs.get("random_state", settings.get("project", "random_seed", default=42))),
        }
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", HistGradientBoostingRegressor(**self.params)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> HistGradientModel:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)
