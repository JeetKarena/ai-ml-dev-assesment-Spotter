"""Serialization helpers for models and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_pickle(path: str | Path) -> Any:
    """Load a joblib/pickle serialized artifact from the filesystem.

    Args:
        path: Path to the artifact.

    Returns:
        Deserialized Python object.
    """
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Artifact not found at {target_path}")
    return joblib.load(target_path)


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Save an object as a serialized joblib/pickle artifact to disk.

    Args:
        obj: Python object to serialize.
        path: Path where serialization should be written.

    Returns:
        Path object pointing to the written file.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target_path)
    return target_path
