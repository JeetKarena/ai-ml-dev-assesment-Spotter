# Freight Rate Prediction Assessment

This repository trains a leakage-safe freight-rate model, generates the two
required submission CSVs, validates them with the provided scorer, and exposes
the fitted pipeline through a small FastAPI service.

## Why this validation strategy

The labelled data ends on 2025-10-31 and the unseen validation set begins on
2025-11-01. Training therefore uses the earlier 80% of distinct dates and
evaluates on the latest 20% of dates. This is more realistic than random
sampling because it avoids learning from future market conditions.

## Modeling decisions

The selected model is a deliberately regularized `HistGradientBoostingRegressor`.
It is evaluated against a median-rate baseline and across expanding forward
folds. Tree depth, minimum leaf size, and L2 regularization are constrained to
reduce variance; the train-versus-holdout gap is recorded in
`artifacts/training_metrics.json`. A neural network was not added simply for
complexity: on this structured dataset, the simpler model is easier to
reproduce, explain, and monitor. The report documents this trade-off.

## Run with Docker

```bash
docker build -t spotter-assessment .
docker run --rm -v "${PWD}/artifacts:/app/artifacts" -v "${PWD}/outputs:/app/outputs" -v "${PWD}/reports:/app/reports" spotter-assessment
docker run --rm -v "${PWD}/artifacts:/app/artifacts" -v "${PWD}/outputs:/app/outputs" spotter-assessment python src/predict.py
docker run --rm -v "${PWD}/outputs:/app/outputs" spotter-assessment python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir outputs/scorer_results
```

The deliverables are `outputs/validation_predictions.csv`,
`outputs/december_predictions.csv`, `outputs/scorer_results/candidate_december.png`,
and `reports/approach.md`. Convert the report to PDF/DOCX for submission after
reviewing its real run metrics.

## Test and serve

```bash
docker run --rm spotter-assessment pytest -q
docker build -f api/Dockerfile -t freight-rate-api .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" freight-rate-api
```

`GET /health` reports whether the model artifact is available. `POST /predict`
accepts pickup, delivery, distance, equipment, weight, date, and optional
geographic/market fields; it returns a positive predicted rate. The API uses
the same serialized preprocessing and model pipeline as batch inference.

## Operational follow-up

Retrain after new labelled loads arrive, compare each candidate with the
previous model on a recent time-based holdout, and monitor missingness,
prediction distribution, and MAE once actual posted rates arrive. December is
outside the observed training range, so its forecast is an extrapolation rather
than a claimed holiday-demand estimate.
