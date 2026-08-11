import pandas as pd

from src.data.splitter import chronological_split


def test_chronological_split_keeps_future_rows_in_validation():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]), "value": [3, 1, 2]})
    train, validation = chronological_split(frame, validation_size=1 / 3, date_column="date")
    assert train["date"].max() < validation["date"].min()
    assert len(train) == 2 and len(validation) == 1
