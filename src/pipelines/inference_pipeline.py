"""Batch inference for required assessment deliverables."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.pipeline import make_model_features


class InferencePipeline:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.data_dir = self.root / "data"
        self.artifacts_dir = self.root / "artifacts"
        self.outputs_dir = self.root / "outputs"

    def run(self, payload=None) -> dict[str, str]:
        model_path = self.artifacts_dir / "freight_rate_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Model artifact is missing. Run `python src/train.py` first.")
        model = joblib.load(model_path)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        validation = pd.read_csv(self.data_dir / "validation.csv")
        template = pd.read_csv(self.data_dir / "validation-predictions-template.csv")
        if not validation["load_id"].equals(template["load_id"]):
            raise ValueError("Validation data and prediction template load_id order do not match")
        prediction = np.maximum(model.predict(make_model_features(validation)), 0.01)
        final = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": prediction.round(2)})
        final_path = self.outputs_dir / "validation_predictions.csv"
        final.to_csv(final_path, index=False)

        december = pd.read_csv(self.data_dir / "december-chart-inputs.csv")
        december["predicted_rate"] = np.maximum(model.predict(make_model_features(december)), 0.01).round(2)
        december_path = self.outputs_dir / "december_predictions.csv"
        december.to_csv(december_path, index=False)
        return {"validation_predictions": str(final_path), "december_predictions": str(december_path)}
