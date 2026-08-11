"""API configuration — central place for all tunable service constants.

Loaded at import time so every module imports from here rather than
reaching into environment variables directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
API_TITLE = "Freight Rate Predictor"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production inference service for the Spotter freight-rate model. "
    "Accepts a load description and returns a positive predicted rate in USD."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(PROJECT_ROOT / "artifacts" / "freight_rate_model.joblib"),
    )
)
