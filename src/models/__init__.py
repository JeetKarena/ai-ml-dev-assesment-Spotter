"""Model definitions, wrappers, and registry interfaces."""

from __future__ import annotations

from src.models.base import BaseModel
from src.models.catboost_model import CatBoostModel
from src.models.ensemble import WeightedEnsembleModel, build_ensemble
from src.models.hist_gradient_model import HistGradientModel
from src.models.lightgbm_model import LightGBMModel
from src.models.registry import registry
from src.models.sklearn_models import (
    GradientBoostingModel,
    MeanBaseline,
    RandomForestModel,
    RidgeRegressionModel,
)
from src.models.xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "registry",
    "HistGradientModel",
    "MeanBaseline",
    "RidgeRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "WeightedEnsembleModel",
    "build_ensemble",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
]
