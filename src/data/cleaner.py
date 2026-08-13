"""Basic cleaning helpers for ML data ingestion."""

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a minimal, safe cleanup pass to a dataset."""
    cleaned = df.copy()

    for column in cleaned.columns:
        if cleaned[column].isna().any():
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].mode(dropna=True).iloc[0]
                    if not cleaned[column].mode(dropna=True).empty
                    else "unknown"
                )

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].astype(str).str.strip()

    for column in cleaned.select_dtypes(include=["datetime64[ns]", "object"]).columns:
        if column in cleaned.columns:
            try:
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
            except Exception:
                pass

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned
