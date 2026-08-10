"""Evaluation validation helpers."""


def validate_predictions(y_true, y_pred):
    """Validate prediction shape compatibility."""
    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and predictions must have the same length")
    return True
