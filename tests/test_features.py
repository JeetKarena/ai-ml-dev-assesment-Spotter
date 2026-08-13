import numpy as np
import pandas as pd

from src.features.pipeline import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, make_model_features


def test_features_are_deterministic_and_invalid_weight_is_missing():
    frame = pd.DataFrame(
        {
            "pickup": ["Lexington"],
            "delivery": ["Fort Wayne"],
            "equipment": ["Dry Van"],
            "date": ["2025-12-01"],
            "distance": [360.0],
            "weight": [-1.0],
            "pickup_lat": [38.0],
            "pickup_lon": [-84.5],
            "delivery_lat": [41.0],
            "delivery_lon": [-85.1],
            "market_index": [1.0],
            "quote_signal": [2.0],
        }
    )
    result = make_model_features(frame)
    assert list(result.columns) == CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    assert result.loc[0, "route"] == "Lexington -> Fort Wayne"
    assert np.isnan(result.loc[0, "weight"])
    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[0, "distance_bucket"] == "medium"
    assert result.loc[0, "is_q4"] == 1
    assert result.loc[0, "is_holiday_week"] == 0


def test_december_inputs_can_omit_operational_quote_fields():
    frame = pd.DataFrame(
        {
            "pickup": ["Lexington"],
            "delivery": ["Fort Wayne"],
            "equipment": ["Dry Van"],
            "date": ["2025-12-01"],
            "distance": [360.0],
            "weight": [32000.0],
        }
    )
    result = make_model_features(frame)
    assert result["market_index"].isna().all()
    assert result["pickup_lat"].isna().all()


def test_holiday_features_flag_christmas_week():
    frame = pd.DataFrame(
        {
            "pickup": ["Lexington"],
            "delivery": ["Fort Wayne"],
            "equipment": ["Dry Van"],
            "date": ["2025-12-25"],
            "distance": [360.0],
            "weight": [32000.0],
        }
    )
    result = make_model_features(frame)
    assert result.loc[0, "days_to_christmas"] == 0
    assert result.loc[0, "is_holiday_week"] == 1
