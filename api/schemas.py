"""Canonical Pydantic schemas shared across all API routes.

Defining schemas in a single module prevents drift between the request
body in prediction.py and the FastAPI OpenAPI docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FreightLoad(BaseModel):
    """Input payload for a single freight-rate prediction request.

    **Location normalisation**: ``pickup``, ``delivery``, and ``equipment``
    are case-insensitive.  ``"LEXINGTON"``, ``"lexington"``, and
    ``"Lexington"`` are all accepted and resolved to the canonical form used
    during training.  Call ``GET /locations`` to retrieve the full list of
    valid cities and equipment types.
    """

    pickup: str = Field(
        ..., min_length=1,
        description="Origin city. Case-insensitive — 'LEXINGTON' and 'lexington' both work.",
    )
    delivery: str = Field(
        ..., min_length=1,
        description="Destination city. Case-insensitive.",
    )
    distance: float = Field(..., gt=0, description="Route distance in miles.")
    equipment: str = Field(
        ..., min_length=1,
        description="Equipment type. Case-insensitive — 'dry van', 'DRY VAN', 'Dry Van' are all valid.",
    )
    weight: float = Field(..., gt=0, description="Load weight in pounds.")
    date: str = Field(..., description="Pickup date in YYYY-MM-DD format.")

    # Optional operational context — absent for the fixed December scenario
    pickup_lat: float | None = Field(None, description="Origin latitude.")
    pickup_lon: float | None = Field(None, description="Origin longitude.")
    delivery_lat: float | None = Field(None, description="Destination latitude.")
    delivery_lon: float | None = Field(None, description="Destination longitude.")
    market_index: float | None = Field(None, description="Spot market index signal.")
    quote_signal: float | None = Field(None, description="Historical quote-level signal.")

    model_config = {"json_schema_extra": {
        "example": {
            "pickup": "Lexington",
            "delivery": "Fort Wayne",
            "distance": 360.0,
            "equipment": "Dry Van",
            "weight": 32000.0,
            "date": "2025-12-15",
        }
    }}


class PredictionResponse(BaseModel):
    """Response payload returned by POST /predict."""

    predicted_rate: float = Field(..., description="Predicted freight rate in USD.")
    model_ready: bool = Field(..., description="Whether the model artifact was loaded.")


class HealthResponse(BaseModel):
    """Response payload returned by GET /health."""

    status: str = Field(..., description="'ok' when model artifact is available.")


class MetadataResponse(BaseModel):
    """Response payload returned by GET /metadata."""

    model: str
    version: str
    artifact: str
