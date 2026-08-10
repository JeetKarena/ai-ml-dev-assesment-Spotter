"""Pydantic schemas for request/response models."""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Example request body."""

    data: dict


class PredictionResponse(BaseModel):
    """Example prediction response."""

    prediction: object
