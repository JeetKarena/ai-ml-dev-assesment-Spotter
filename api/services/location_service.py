"""Location and equipment normalizer for the API input layer.

The model was trained on city names in Title Case (e.g. "Fort Wayne").
Users may send "FORT WAYNE", "fort wayne", "fort  wayne" etc.
This module resolves any reasonable variant to the canonical training-set form
or raises a clear 422 error listing valid options.

The canonical location list is extracted directly from the training CSV at
import time, so it is always consistent with what the model was trained on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Canonical values extracted from the training data at import time
# ---------------------------------------------------------------------------
_TRAINING_CSV: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "train-test.csv"


def _load_canonical(column: str) -> frozenset[str]:
    """Read unique column values from the training CSV without pandas."""
    if not _TRAINING_CSV.is_file():
        return frozenset()
    seen: set[str] = set()
    with _TRAINING_CSV.open(encoding="utf-8", newline="") as fh:
        import csv

        reader = csv.DictReader(fh)
        for row in reader:
            val = row.get(column, "").strip()
            if val:
                seen.add(val)
    return frozenset(seen)


# Canonical sets — loaded once at module import
VALID_PICKUPS: Final[frozenset[str]] = _load_canonical("pickup")
VALID_DELIVERIES: Final[frozenset[str]] = _load_canonical("delivery")
VALID_EQUIPMENT: Final[frozenset[str]] = _load_canonical("equipment")

# Build a lower-cased lookup dict for O(1) case-insensitive matching
_LOCATION_LOOKUP: Final[dict[str, str]] = {
    # collapse multiple whitespace and lower-case the key
    re.sub(r"\s+", " ", loc).lower(): loc
    for loc in VALID_PICKUPS | VALID_DELIVERIES
}
_EQUIPMENT_LOOKUP: Final[dict[str, str]] = {equip.lower(): equip for equip in VALID_EQUIPMENT}


# ---------------------------------------------------------------------------
# Public normalizers
# ---------------------------------------------------------------------------


def normalize_location(raw: str, field: str = "location") -> str:
    """Resolve a user-supplied city name to its canonical training-set form.

    Handles case variations (``LEXINGTON`` → ``Lexington``), extra whitespace,
    and surrounding punctuation.

    Args:
        raw: The user-supplied city string.
        field: Human-readable field name used in the error message.

    Returns:
        Canonical city name as it appears in the training data.

    Raises:
        ValueError: When the city cannot be matched.  The message includes
            the full list of valid locations so the caller can surface it.
    """
    cleaned = re.sub(r"\s+", " ", raw.strip())
    key = cleaned.lower()

    # Exact (case-insensitive) match
    if key in _LOCATION_LOOKUP:
        return _LOCATION_LOOKUP[key]

    # Fuzzy fallback: find any canonical city whose lower-case name is a
    # substring of the input or vice-versa (handles "St Louis" vs "St. Louis")
    stripped_key = re.sub(r"[^a-z ]", "", key)  # remove punctuation
    for canonical_key, canonical_name in _LOCATION_LOOKUP.items():
        if re.sub(r"[^a-z ]", "", canonical_key) == stripped_key:
            return canonical_name

    valid = sorted(_LOCATION_LOOKUP.values())
    raise ValueError(f"'{raw}' is not a recognised {field}. " f"Valid locations ({len(valid)}): {valid}")


def normalize_equipment(raw: str) -> str:
    """Resolve a user-supplied equipment type to its canonical training-set form.

    Args:
        raw: The user-supplied equipment string (e.g. ``"dry van"``, ``"DRY VAN"``).

    Returns:
        Canonical equipment name (e.g. ``"Dry Van"``).

    Raises:
        ValueError: When the equipment type cannot be matched.
    """
    key = raw.strip().lower()
    if key in _EQUIPMENT_LOOKUP:
        return _EQUIPMENT_LOOKUP[key]

    valid = sorted(VALID_EQUIPMENT)
    raise ValueError(f"'{raw}' is not a recognised equipment type. " f"Valid equipment types: {valid}")


def list_locations() -> list[str]:
    """Return the sorted canonical location list.

    Useful for powering autocomplete endpoints or API documentation.
    """
    return sorted(_LOCATION_LOOKUP.values())


def list_equipment() -> list[str]:
    """Return the sorted canonical equipment list."""
    return sorted(VALID_EQUIPMENT)
