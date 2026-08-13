from __future__ import annotations

from pathlib import Path

from src.core.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS_DIR = PROJECT_ROOT / settings.get("paths", "artifacts_dir")

OUTPUTS_DIR = PROJECT_ROOT / settings.get("paths", "outputs_dir")

REPORTS_DIR = PROJECT_ROOT / settings.get("paths", "reports_dir")

TRAIN_DATA = PROJECT_ROOT / settings.get("paths", "train_data")

VALIDATION_DATA = PROJECT_ROOT / settings.get("paths", "validation_data")

VALIDATION_TEMPLATE = PROJECT_ROOT / settings.get(
    "paths",
    "validation_template",
)

DECEMBER_DATA = PROJECT_ROOT / settings.get("paths", "december_data")


for directory in (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
