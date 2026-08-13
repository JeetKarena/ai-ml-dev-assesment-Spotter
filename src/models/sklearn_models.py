"""Scikit-learn model implementations inheriting from BaseModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocessor
from src.models.base import BaseModel


class MeanBaseline(BaseModel):
    """Simple baseline model that predicts the historical group mean or median."""

    def __init__(self, group_cols: list[str] | None = None) -> None:
        self.group_cols = group_cols or ["route", "equipment"]
        self.global_median = 0.0
        self.group_rates = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> MeanBaseline:
        df = X.copy()
        df["target"] = y
        self.global_median = float(df["target"].median())

        # Calculate median target for each group
        grouped = df.groupby(self.group_cols)["target"].median()
        self.group_rates = grouped.to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = tuple(row[col] for col in self.group_cols)
            # Handle tuple key vs single string key
            if len(self.group_cols) == 1:
                key = key[0]
            preds.append(self.group_rates.get(key, self.global_median))
        return np.array(preds)


class RidgeRegressionModel(BaseModel):
    """Ridge regression model trained within a preprocessing pipeline."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RidgeRegressionModel:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest regressor model trained within a preprocessing pipeline."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RandomForestModel:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting regressor model trained within a preprocessing pipeline."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=self.n_estimators,
                        learning_rate=self.learning_rate,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> GradientBoostingModel:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)
