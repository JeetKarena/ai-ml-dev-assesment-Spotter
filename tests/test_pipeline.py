import pandas as pd

from score import validate_december, validate_predictions
from src.pipelines.training_pipeline import TrainingPipeline


def test_training_pipeline_builds_a_sklearn_pipeline(tmp_path):
    pipeline = TrainingPipeline(project_root=tmp_path)
    model = pipeline._build_model()
    assert list(model.pipeline.named_steps) == ["preprocess", "model"]


def test_submission_outputs_match_scorer_contract():
    validate_predictions(pd.read_csv("outputs/validation_predictions.csv"))
    validate_december(pd.read_csv("outputs/december_predictions.csv"))
