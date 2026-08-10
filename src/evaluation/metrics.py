"""Metric helpers for model evaluation."""


def compute_metrics(y_true, y_pred):
    """Return a minimal metric summary placeholder."""
    return {
        "n_samples": len(y_true),
        "n_predictions": len(y_pred),
    }
