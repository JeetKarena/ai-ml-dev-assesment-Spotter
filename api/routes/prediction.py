"""Prediction route — the primary inference endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core.exceptions import SpotterError
from api.schemas import FreightLoad, PredictionResponse
from api.services.prediction_service import predict as run_prediction

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse, summary="Predict freight rate")
def predict(load: FreightLoad) -> PredictionResponse:
    """Predict the USD rate for a single freight load.

    The request body mirrors the fields available at quote time.
    Optional geographic and market fields may be omitted; the fitted
    pipeline's median imputer handles them consistently.

    Returns a positive ``predicted_rate`` rounded to two decimal places.
    """
    try:
        rate = run_prediction(load.model_dump())
        return PredictionResponse(predicted_rate=rate, model_ready=True)
    except SpotterError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
