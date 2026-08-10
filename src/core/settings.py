from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Settings:
    """Loads and exposes project configuration from config/config.yaml."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve nested configuration values."""
        value: Any = self._config

        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)

            if value is None:
                return default

        return value


settings = Settings()
