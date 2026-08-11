"""Central registry for retrieving model instances by name."""

from __future__ import annotations

from typing import Type
from src.models.base import BaseModel
from src.models.hist_gradient_model import HistGradientModel
from src.models.sklearn_models import (
    MeanBaseline,
    RidgeRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
)
from src.models.ensemble import WeightedEnsembleModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.catboost_model import CatBoostModel


class ModelRegistry:
    """Registry class linking string identifiers to BaseModel implementations."""

    def __init__(self) -> None:
        self._models = {
            "hist_gradient_boosting": HistGradientModel,
            "mean_baseline": MeanBaseline,
            "ridge_regression": RidgeRegressionModel,
            "random_forest": RandomForestModel,
            "gradient_boosting": GradientBoostingModel,
            "weighted_ensemble": WeightedEnsembleModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "catboost": CatBoostModel,
        }

    def register(self, name: str, model_cls: Type[BaseModel]) -> Type[BaseModel]:
        """Register a new model class under a custom name."""
        self._models[name] = model_cls
        return model_cls

    def get(self, name: str, **kwargs) -> BaseModel:
        """Retrieve and instantiate a registered model by name.

        Args:
            name: The registered name of the model class.
            kwargs: Parameters forwarded to the model constructor.

        Returns:
            An instantiated BaseModel.
        """
        model_cls = self._models.get(name)
        if model_cls is None:
            raise KeyError(f"Model '{name}' is not registered. Registered models: {list(self._models.keys())}")
        return model_cls(**kwargs)


# Global singleton registry instance
registry = ModelRegistry()
