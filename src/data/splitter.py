"""Small, reusable data-splitting helpers."""

from __future__ import annotations

import pandas as pd


def chronological_split(df: pd.DataFrame, validation_size: float = 0.2, date_column: str | None = None):
    """Create a chronological holdout split without shuffling."""
    if validation_size <= 0 or validation_size >= 1:
        raise ValueError("validation_size must be between 0 and 1")

    ordered = df.copy()
    if date_column and date_column in ordered.columns:
        ordered = ordered.sort_values(by=date_column).reset_index(drop=True)

    split_index = int(len(ordered) * (1 - validation_size))
    train = ordered.iloc[:split_index]
    validation = ordered.iloc[split_index:]

    return train, validation


def holdout_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Convenience wrapper around a standard holdout split."""
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=random_state)


def time_series_split(df: pd.DataFrame, n_splits: int = 5):
    """Return a TimeSeriesSplit-compatible iterator shape."""
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=n_splits)
    return list(splitter.split(df))
