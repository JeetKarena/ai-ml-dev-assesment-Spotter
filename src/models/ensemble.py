"""Ensemble model implementation combining multiple BaseModel estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class WeightedEnsembleModel(BaseModel):
    """Ensemble model that averages predictions across multiple estimators."""

    def __init__(self, models: list[BaseModel], weights: list[float] | None = None) -> None:
        """Create a weighted ensemble of sub-models.

        Args:
            models: List of instantiated BaseModel objects.
            weights: Optional list of weights corresponding to each model.
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> WeightedEnsembleModel:
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for model, weight in zip(self.models, self.weights, strict=True):
            preds = model.predict(X)
            predictions.append(preds * weight)
        return np.sum(predictions, axis=0)


def build_ensemble(
    models: list[BaseModel] | None = None, weights: list[float] | None = None
) -> WeightedEnsembleModel:
    """Helper function to build a WeightedEnsembleModel.

    Args:
        models: List of sub-models.
        weights: Optional list of weights.

    Returns:
        Instantiated WeightedEnsembleModel.
    """
    if not models:
        from src.models.hist_gradient_model import HistGradientModel
        from src.models.sklearn_models import RidgeRegressionModel

        models = [HistGradientModel(), RidgeRegressionModel()]
    return WeightedEnsembleModel(models=models, weights=weights)
