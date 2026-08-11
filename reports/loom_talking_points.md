# 2-3 minute Loom talking points

## 0:00-0:25 - Objective and data

I built a freight-load rate prediction pipeline that trains on 48,000 labelled
loads and predicts the 12,000 rows in the validation file. The training period
runs from January 1 through October 31, 2025; the validation data begins in
November, so time is central to the modeling strategy.

## 0:25-0:55 - Data findings and fixes

The data has no duplicate rows or duplicate load IDs. I found 300 missing
weights, 292 non-positive weights, and 374 missing market-index values. A
non-positive truck weight is physically invalid, so I turn it into missing
data. Numeric missing values use a median imputer and categorical fields use
the most-frequent value. Those transformations are inside the model pipeline,
so they are learned from training data only.

## 0:55-1:30 - Validation strategy

I chose a chronological holdout instead of random k-fold validation. Randomly
mixing dates would let future market and seasonal behavior influence the model
used to evaluate earlier loads. I trained through August 31 and held out
September 1 onward. That gives a realistic forward-looking test of 9,523
loads, then I retrain on all labelled data before creating submission files.

## 1:30-2:05 - Features and model

The features represent operational, geographic, market, and calendar context:
origin, destination, route, equipment, distance, weight, coordinates, market
index, quote signal, and weekday/month seasonality. I excluded load_id because
it is an identifier, not a pricing driver. I used a single
HistGradientBoosting model. It captures non-linear interactions such as route
and equipment effects without the extra operational complexity of an ensemble.
On the chronological holdout it reduced MAE from $1,148.92 for the median-rate
baseline to $152.35, with an R-squared of 0.823.

## 2:05-2:35 - Code and outputs

The training pipeline saves the fitted model and recorded metrics. The
inference pipeline creates `validation_predictions.csv` and the fixed December
scenario. I then ran the supplied `score.py`; it validated all 12,000 final
predictions, all 31 December rows, and generated the report chart. The project
is fully reproducible in Docker with pinned dependencies.
