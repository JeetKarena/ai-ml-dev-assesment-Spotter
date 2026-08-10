"""Artifact loading service for the API layer."""

from pathlib import Path


def load_artifacts(base_dir: str | Path = "artifacts"):
    """Return artifact artifact names from a directory."""
    base = Path(base_dir)
    return {
        "model": base / "model.pkl",
        "encoders": base / "encoders.pkl",
        "feature_columns": base / "feature_columns.json",
    }
