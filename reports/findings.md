# Data Findings and Error Analysis

## Observed data quality

The development data has 48,000 labelled loads from 2025-01-01 through 2025-10-31. There are 300 missing weights, 292 non-positive weights, and 374 missing market-index values. Non-positive weights are not silently corrected: they are converted to missing and imputed only within the fitted training pipeline.

The IQR rule flags 260 unusual posted rates. They remain in training because unusual rates can be real operational events rather than data errors.

## Holdout behaviour

On the strictly later chronological holdout, the chosen model recorded MAE $114.40, RMSE $633.40, MAPE 5.19%, and R-squared 0.828. The higher RMSE relative to MAE indicates occasional large residuals; likely review segments are uncommon routes, equipment types, and long-haul loads.

## Next production checks

When actual rates arrive, segment MAE by route, equipment, distance bucket, and month. Monitor input missingness, unseen-category frequency, and prediction-distribution drift. Retraining should only promote a candidate after it outperforms the deployed model on the most recent time-based holdout.
