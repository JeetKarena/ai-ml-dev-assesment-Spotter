from src.evaluation.metrics import compute_metrics
from src.models.pytorch_model import FreightRateNet
from src.models.registry import registry
from src.models.base import BaseModel
import pandas as pd
import numpy as np


def test_regression_metrics_are_correct_for_perfect_predictions():
    metrics = compute_metrics([100.0, 200.0], [100.0, 200.0])
    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "mape": 0.0, "median_ae": 0.0, "n_samples": 2}


def test_pytorch_tabular_model_returns_one_rate_per_row():
    import torch

    model = FreightRateNet(num_continuous=2, embedding_dims=[(4, 2)])
    prediction = model(torch.zeros((3, 2)), [torch.tensor([0, 1, 2])])
    assert tuple(prediction.shape) == (3,)


def test_model_registry_and_oop_wrappers():
    # Verify we can resolve and instantiate models from registry
    model = registry.get("hist_gradient_boosting")
    assert isinstance(model, BaseModel)
    
    # Test MeanBaseline functionality
    baseline = registry.get("mean_baseline")
    assert isinstance(baseline, BaseModel)
    
    df = pd.DataFrame({
        "route": ["A -> B", "A -> B", "C -> D"],
        "equipment": ["Dry Van", "Dry Van", "Reefer"]
    })
    y = np.array([100.0, 200.0, 300.0])
    
    baseline.fit(df, y)
    preds = baseline.predict(df)
    assert len(preds) == 3
    assert preds[0] == 150.0  # Median of [100, 200]
    assert preds[2] == 300.0
