"""CatBoost model wrapper inheriting from BaseModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.base import BaseModel
from src.features.pipeline import build_preprocessor


class CatBoostModel(BaseModel):
    """Wrapper class around CatBoostRegressor."""

    def __init__(self, **kwargs) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is not installed. Add it to requirements.txt to use CatBoostModel."
            ) from exc

        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", CatBoostRegressor(**kwargs)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> CatBoostModel:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)
