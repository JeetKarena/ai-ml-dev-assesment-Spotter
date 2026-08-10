"""Basic data validation helpers."""

from typing import Any

import pandas as pd


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    """Validate required columns and return a validation report."""
    missing = [column for column in required_columns if column not in df.columns]

    report = {
        "required_columns_present": not missing,
        "missing_columns": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    if missing:
        report["ok"] = False
        raise ValueError(f"Missing required columns: {missing}")

    report["ok"] = True
    return report


def validate_ranges(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> dict[str, Any]:
    """Check that simple numeric ranges are well-formed."""
    numeric_columns = numeric_columns or []
    invalid_ranges: dict[str, dict[str, Any]] = {}

    for column in numeric_columns:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int(series.isna().sum())
            if invalid_count:
                invalid_ranges[column] = {"invalid_count": invalid_count}

    return {"invalid_ranges": invalid_ranges, "ok": not invalid_ranges}
