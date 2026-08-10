"""Filesystem helpers for project operations."""

from pathlib import Path


def ensure_directory(path: str | Path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
