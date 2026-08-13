"""Base model interfaces for model registration and training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract Base Class for all models in the Freight Rate Predictor codebase.

    This interface ensures that all models expose uniform methods for fitting,
    predicting, saving, and loading, satisfying OOP clean architecture guidelines.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BaseModel:
        """Fit the model to the training dataset.

        Args:
            X: Input features DataFrame.
            y: Target values.

        Returns:
            The fitted model instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input features.

        Args:
            X: Input features DataFrame.

        Returns:
            Numpy array of predictions.
        """
        pass
