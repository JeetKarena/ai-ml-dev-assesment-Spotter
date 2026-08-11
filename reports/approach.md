# Freight Rate Prediction - Technical Report

## Validation approach

The labelled development data covers 2025-01-01 to 2025-10-31. Because the final validation set starts after this period, I used a chronological holdout rather than a random split. The model trains on data through 2025-08-31 and evaluates on the subsequent period beginning 2025-09-01 (9,523 loads). This prevents future rate, market, and seasonal information from leaking into evaluation.

## Data findings and quality treatment

The development file contains 48,000 loads, 0 duplicate rows, and 0 duplicate load IDs. It contains 300 missing weights, 292 non-positive weights, and 374 missing market-index values. Non-positive weights are treated as missing because they are physically invalid; numeric missing values are median-imputed inside the fitted pipeline. Categorical missing values are imputed with the training-fold mode. Fitting these transformations in the pipeline keeps holdout information out of preprocessing.

## Features and model

Features include origin, destination, route, equipment, coordinates, distance, weight, market index, quote signal, and calendar seasonality (weekday, month, day-of-year, and cyclic month/weekday terms). `load_id` is excluded because it is an identifier, not a pricing signal.

I compared the selected model against a median-rate baseline on the same future holdout. The baseline MAE was $1,148.92; the regularized HistGradientBoosting model reduced it to $113.05 (RMSE $632.80, MAPE 5.15%, R-squared 0.828). Model capacity is deliberately constrained with shallow trees, a minimum of 80 observations per leaf, and L2 regularization. I also recorded expanding forward-fold results to check that performance is not dependent on a single date split. After evaluation, the final model is refit on all labelled development data for submission predictions.

## Reproduction

Run `docker build -t spotter-assessment .`, then `docker run --rm spotter-assessment` to train. Run `docker run --rm spotter-assessment python src/predict.py` to create `outputs/validation_predictions.csv` and `outputs/december_predictions.csv`. Finally run `score.py` against those files to validate format and generate the December chart.
