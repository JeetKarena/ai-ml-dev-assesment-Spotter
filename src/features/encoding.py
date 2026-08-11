"""Categorical encoding utilities and custom estimators."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encoder that handles unseen categories during inference safely."""

    def __init__(self, categories: str = "auto", unknown_value: int = -1) -> None:
        self.categories = categories
        self.unknown_value = unknown_value
        self.encoder = OrdinalEncoder(
            categories=self.categories,
            handle_unknown="use_encoded_value",
            unknown_value=self.unknown_value
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SafeOrdinalEncoder:
        self.encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)
