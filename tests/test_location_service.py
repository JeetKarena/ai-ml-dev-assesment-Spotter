"""Tests for location and equipment normalisation."""

from __future__ import annotations

import pytest

from api.services.location_service import (
    normalize_location,
    normalize_equipment,
    list_locations,
    list_equipment,
)


# ---------------------------------------------------------------------------
# normalize_location
# ---------------------------------------------------------------------------

class TestNormalizeLocation:
    def test_title_case_passes_through(self):
        assert normalize_location("Lexington") == "Lexington"

    def test_upper_case_resolved(self):
        assert normalize_location("LEXINGTON") == "Lexington"

    def test_lower_case_resolved(self):
        assert normalize_location("lexington") == "Lexington"

    def test_mixed_case_resolved(self):
        assert normalize_location("lExInGtOn") == "Lexington"

    def test_leading_trailing_whitespace_stripped(self):
        assert normalize_location("  Fort Wayne  ") == "Fort Wayne"

    def test_extra_internal_whitespace_collapsed(self):
        assert normalize_location("Fort  Wayne") == "Fort Wayne"

    def test_multi_word_city_upper(self):
        assert normalize_location("FORT WAYNE") == "Fort Wayne"

    def test_st_louis_with_period(self):
        # Punctuation stripping fuzzy match: "St Louis" → "St. Louis"
        assert normalize_location("St Louis") == "St. Louis"

    def test_unknown_city_raises_value_error(self):
        with pytest.raises(ValueError, match="not a recognised"):
            normalize_location("Gotham City")

    def test_error_message_lists_valid_locations(self):
        with pytest.raises(ValueError, match="Lexington"):
            normalize_location("XYZ_UNKNOWN_999")

    def test_list_locations_returns_all_canonical(self):
        locs = list_locations()
        assert "Lexington" in locs
        assert "Fort Wayne" in locs
        assert "St. Louis" in locs
        assert len(locs) == len(set(locs)), "No duplicates expected"


# ---------------------------------------------------------------------------
# normalize_equipment
# ---------------------------------------------------------------------------

class TestNormalizeEquipment:
    def test_title_case_passes_through(self):
        assert normalize_equipment("Dry Van") == "Dry Van"

    def test_upper_case_resolved(self):
        assert normalize_equipment("DRY VAN") == "Dry Van"

    def test_lower_case_resolved(self):
        assert normalize_equipment("dry van") == "Dry Van"

    def test_flatbed_lower(self):
        assert normalize_equipment("flatbed") == "Flatbed"

    def test_reefer_upper(self):
        assert normalize_equipment("REEFER") == "Reefer"

    def test_unknown_equipment_raises(self):
        with pytest.raises(ValueError, match="not a recognised equipment"):
            normalize_equipment("Tank Car")

    def test_list_equipment_returns_all_canonical(self):
        equip = list_equipment()
        assert "Dry Van" in equip
        assert "Flatbed" in equip
        assert "Reefer" in equip
        assert len(equip) == 3
