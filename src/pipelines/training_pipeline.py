"""Chronological training, evaluation, and artifact creation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.core.settings import settings
from src.evaluation.metrics import compute_metrics
from src.features.pipeline import TARGET_COLUMN, make_model_features
from src.models.base import BaseModel
from src.models.registry import registry


class TrainingPipeline:
    """Train an explainable, leak-safe freight-rate model."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.data_dir = self.root / "data"
        self.artifacts_dir = self.root / "artifacts"
        self.reports_dir = self.root / "reports"

    def _build_model(self) -> BaseModel:
        model_name = settings.get("model", "name", default="hist_gradient_boosting")
        return registry.get(model_name)

    def run(self) -> dict:
        train_path = self.data_dir / "train-test.csv"
        frame = pd.read_csv(train_path)
        if TARGET_COLUMN not in frame:
            raise ValueError(f"{train_path} must contain {TARGET_COLUMN}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)

        unique_dates = np.array(sorted(frame["date"].unique()))
        cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])
        development = frame.loc[frame["date"] < cutoff].copy()
        holdout = frame.loc[frame["date"] >= cutoff].copy()
        if development.empty or holdout.empty:
            raise ValueError("Chronological split produced an empty partition")

        baseline_rate = float(development[TARGET_COLUMN].median())
        baseline_metrics = compute_metrics(holdout[TARGET_COLUMN], np.full(len(holdout), baseline_rate))
        candidate = self._build_model()
        candidate.fit(make_model_features(development), development[TARGET_COLUMN])
        model_metrics = compute_metrics(
            holdout[TARGET_COLUMN], candidate.predict(make_model_features(holdout))
        )
        train_metrics = compute_metrics(
            development[TARGET_COLUMN], candidate.predict(make_model_features(development))
        )
        rolling_metrics = self._rolling_forward_metrics(frame, unique_dates)

        final_model = self._build_model()
        final_model.fit(make_model_features(frame), frame[TARGET_COLUMN])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, self.artifacts_dir / "freight_rate_model.joblib")

        quality = {
            "rows": int(len(frame)),
            "date_min": str(frame["date"].min().date()),
            "date_max": str(frame["date"].max().date()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_load_ids": int(frame["load_id"].duplicated().sum()),
            "missing_weight": int(frame["weight"].isna().sum()),
            "nonpositive_weight": int((frame["weight"] <= 0).sum()),
            "missing_market_index": int(frame["market_index"].isna().sum()),
            "target_iqr_outliers": self._iqr_outlier_count(frame[TARGET_COLUMN]),
        }
        summary = {
            "split": {
                "strategy": "chronological holdout",
                "train_through": str((cutoff - pd.Timedelta(days=1)).date()),
                "holdout_from": str(cutoff.date()),
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
            },
            "baseline_median": baseline_metrics,
            "hist_gradient_boosting": model_metrics,
            "training_fit": train_metrics,
            "generalization_gap_mae": round(model_metrics["mae"] - train_metrics["mae"], 4),
            "rolling_forward_validation": rolling_metrics,
            "data_quality": quality,
        }
        (self.artifacts_dir / "training_metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self._write_report(summary)
        self._write_findings(summary)
        return summary

    @staticmethod
    def _iqr_outlier_count(target: pd.Series) -> int:
        """Count, but do not remove, statistically unusual historical rates."""
        lower, upper = target.quantile([0.25, 0.75])
        iqr = upper - lower
        return int(((target < lower - 1.5 * iqr) | (target > upper + 1.5 * iqr)).sum())

    def _rolling_forward_metrics(self, frame: pd.DataFrame, unique_dates: np.ndarray) -> list[dict]:
        """Measure stability across expanding, strictly forward validation folds."""
        results: list[dict] = []
        for fraction in (0.50, 0.60, 0.70, 0.80):
            cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * fraction)])
            train = frame.loc[frame["date"] < cutoff]
            valid = frame.loc[frame["date"] >= cutoff]
            candidate = self._build_model()
            candidate.fit(make_model_features(train), train[TARGET_COLUMN])
            results.append(
                {
                    "holdout_from": str(cutoff.date()),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    **compute_metrics(valid[TARGET_COLUMN], candidate.predict(make_model_features(valid))),
                }
            )
        return results

    def _write_report(self, summary: dict) -> None:
        quality = summary["data_quality"]
        split = summary["split"]
        baseline = summary["baseline_median"]
        model = summary["hist_gradient_boosting"]
        val_approach = (
            f"The labelled development data covers {quality['date_min']} to "
            f"{quality['date_max']}. Because the final validation set starts "
            f"after this period, I used a chronological holdout rather than a "
            f"random split. The model trains on data through "
            f"{split['train_through']} and evaluates on the subsequent period "
            f"beginning {split['holdout_from']} ({split['holdout_rows']:,} "
            f"loads). This prevents future rate, market, and seasonal "
            f"information from leaking into evaluation."
        )
        data_quality = (
            f"The development file contains {quality['rows']:,} loads, "
            f"{quality['duplicate_rows']:,} duplicate rows, and "
            f"{quality['duplicate_load_ids']:,} duplicate load IDs. "
            f"It contains {quality['missing_weight']:,} missing weights, "
            f"{quality['nonpositive_weight']:,} non-positive weights, and "
            f"{quality['missing_market_index']:,} missing market-index "
            f"values. Non-positive weights are treated as missing because "
            f"they are physically invalid; numeric missing values are "
            f"median-imputed inside the fitted pipeline. Categorical "
            f"missing values are imputed with the training-fold mode. "
            f"Fitting these transformations in the pipeline keeps holdout "
            f"information out of preprocessing."
        )
        features_text = (
            "Features include origin, destination, route, equipment, "
            "coordinates, distance, weight, market index, quote signal, "
            "and calendar seasonality (weekday, month, day-of-year, and "
            "cyclic month/weekday terms). `load_id` is excluded because "
            "it is an identifier, not a pricing signal."
        )
        model_text = (
            f"I compared the selected model against a median-rate baseline "
            f"on the same future holdout. The baseline MAE was "
            f"${baseline['mae']:,.2f}; the regularized "
            f"HistGradientBoosting model reduced it to "
            f"${model['mae']:,.2f} (RMSE ${model['rmse']:,.2f}, "
            f"MAPE {model['mape']:.2f}%, R-squared {model['r2']:.3f}). "
            f"Model capacity is deliberately constrained with shallow "
            f"trees, a minimum of 80 observations per leaf, and L2 "
            f"regularization. I also recorded expanding forward-fold "
            f"results to check that performance is not dependent on a "
            f"single date split. After evaluation, the final model is "
            f"refit on all labelled development data for submission "
            f"predictions."
        )
        repro_text = (
            "Run `docker build -t spotter-assessment .`, then "
            "`docker run --rm spotter-assessment` to train. "
            "Run `docker run --rm spotter-assessment python "
            "src/predict.py` to create "
            "`outputs/validation_predictions.csv` and "
            "`outputs/december_predictions.csv`. Finally run "
            "`score.py` against those files to validate format "
            "and generate the December chart."
        )
        content = f"""# Freight Rate Prediction - Technical Report

## Validation approach

{val_approach}

## Data findings and quality treatment

{data_quality}

## Features and model

{features_text}

{model_text}

## Reproduction

{repro_text}
"""
        (self.reports_dir / "approach.md").write_text(content, encoding="utf-8")

    def _write_findings(self, summary: dict) -> None:
        """Create a concise, reproducible EDA and error-analysis handoff."""
        quality = summary["data_quality"]
        model = summary["hist_gradient_boosting"]
        obs_quality = (
            f"The development data has {quality['rows']:,} labelled loads "
            f"from {quality['date_min']} through {quality['date_max']}. "
            f"There are {quality['missing_weight']:,} missing weights, "
            f"{quality['nonpositive_weight']:,} non-positive weights, and "
            f"{quality['missing_market_index']:,} missing market-index "
            f"values. Non-positive weights are not silently corrected: "
            f"they are converted to missing and imputed only within the "
            f"fitted training pipeline."
        )
        iqr_text = (
            f"The IQR rule flags {quality['target_iqr_outliers']:,} "
            f"unusual posted rates. They remain in training because "
            f"unusual rates can be real operational events rather than "
            f"data errors."
        )
        holdout_text = (
            f"On the strictly later chronological holdout, the chosen "
            f"model recorded MAE ${model['mae']:,.2f}, RMSE "
            f"${model['rmse']:,.2f}, MAPE {model['mape']:.2f}%, and "
            f"R-squared {model['r2']:.3f}. The higher RMSE relative to "
            f"MAE indicates occasional large residuals; likely review "
            f"segments are uncommon routes, equipment types, and "
            f"long-haul loads."
        )
        prod_text = (
            "When actual rates arrive, segment MAE by route, equipment, "
            "distance bucket, and month. Monitor input missingness, "
            "unseen-category frequency, and prediction-distribution "
            "drift. Retraining should only promote a candidate after it "
            "outperforms the deployed model on the most recent "
            "time-based holdout."
        )
        content = f"""# Data Findings and Error Analysis

## Observed data quality

{obs_quality}

{iqr_text}

## Holdout behaviour

{holdout_text}

## Next production checks

{prod_text}
"""
        (self.reports_dir / "findings.md").write_text(content, encoding="utf-8")
