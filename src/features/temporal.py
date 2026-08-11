"""Temporal feature extraction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_temporal_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Create temporal and calendar-related features from a date column.

    Includes weekend flags, holiday proximities (Thanksgiving, Christmas),
    holiday weeks, Q4 identifiers, and cyclical features (sine/cosine transforms)
    to enforce continuity from December to January.
    """
    res = df.copy()
    date = pd.to_datetime(res[date_column], errors="coerce")
    if date.isna().any():
        raise ValueError(f"'{date_column}' column contains invalid or missing dates.")

    res["day_of_week"] = date.dt.dayofweek
    res["month"] = date.dt.month
    res["day_of_month"] = date.dt.day
    res["day_of_year"] = date.dt.dayofyear
    res["week_of_year"] = date.dt.isocalendar().week.astype(int)
    res["quarter"] = date.dt.quarter
    res["is_weekend"] = (res["day_of_week"] >= 5).astype(int)
    res["is_q4"] = (res["quarter"] == 4).astype(int)

    # US Holiday projections
    years = date.dt.year.astype(str)
    christmas = pd.to_datetime(years + "-12-25")
    
    # Thanksgiving is the 4th Thursday in November
    thanksgiving = pd.to_datetime(years + "-11-01")
    thanksgiving += pd.to_timedelta((3 - thanksgiving.dt.dayofweek) % 7 + 21, unit="D")

    res["days_to_christmas"] = (christmas - date).dt.days
    res["days_to_thanksgiving"] = (thanksgiving - date).dt.days
    res["is_holiday_week"] = (
        res["days_to_christmas"].abs().le(3) | res["days_to_thanksgiving"].abs().le(3)
    ).astype(int)

    # Cyclical representations
    res["month_sin"] = np.sin(2 * np.pi * res["month"] / 12)
    res["month_cos"] = np.cos(2 * np.pi * res["month"] / 12)
    res["day_of_week_sin"] = np.sin(2 * np.pi * res["day_of_week"] / 7)
    res["day_of_week_cos"] = np.cos(2 * np.pi * res["day_of_week"] / 7)

    return res
