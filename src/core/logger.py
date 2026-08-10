from __future__ import annotations

import logging

from src.core.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""

    logging.basicConfig(
        level=getattr(logging, settings.get("logging", "level", default="INFO")),
        format=settings.get("logging", "format"),
    )

    return logging.getLogger(name)
