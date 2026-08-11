"""Artifact discovery helpers for the API layer."""

from __future__ import annotations

from pathlib import Path

from api.config import MODEL_PATH


def list_artifacts(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Return a mapping of known artifact names to their resolved paths.

    This is intentionally read-only.  Writing artifacts is the responsibility
    of the training pipeline, not the serving layer.

    Args:
        base_dir: Override the artifacts directory.  Defaults to the path
            configured in api/config.py.

    Returns:
        Dict with keys ``"model"`` and derived entries for any extra
        ``.joblib`` or ``.json`` files found in the directory.
    """
    base = Path(base_dir) if base_dir else MODEL_PATH.parent
    result: dict[str, Path] = {"model": MODEL_PATH}

    if base.is_dir():
        for path in sorted(base.iterdir()):
            if path.suffix in {".joblib", ".json"} and path != MODEL_PATH:
                result[path.stem] = path

    return result


def model_artifact_exists() -> bool:
    """Return True when the primary model artifact is present on disk."""
    return MODEL_PATH.is_file()
