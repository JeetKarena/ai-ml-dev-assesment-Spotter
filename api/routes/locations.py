"""Locations reference endpoint — exposes valid pickup/delivery/equipment values."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.services.location_service import list_locations, list_equipment

router = APIRouter(tags=["Reference"])


class LocationsResponse(BaseModel):
    """All valid pickup, delivery, and equipment values known to the model."""

    locations: list[str]
    equipment: list[str]
    location_count: int
    equipment_count: int


@router.get(
    "/locations",
    response_model=LocationsResponse,
    summary="List all valid pickup / delivery locations and equipment types",
)
def get_locations() -> LocationsResponse:
    """Return the canonical city and equipment lists.

    These are extracted directly from the training CSV, so they are always
    consistent with what the model was trained on.  Use them to populate
    autocomplete fields or validate inputs client-side before calling
    ``POST /predict``.

    The API accepts any capitalisation of these names (``LEXINGTON``,
    ``lexington``, ``Lexington`` are all accepted), but this endpoint
    returns the canonical Title Case form used internally.
    """
    locs = list_locations()
    equip = list_equipment()
    return LocationsResponse(
        locations=locs,
        equipment=equip,
        location_count=len(locs),
        equipment_count=len(equip),
    )
