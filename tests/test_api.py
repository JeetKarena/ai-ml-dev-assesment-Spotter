from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint_reports_a_known_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "model_unavailable"}


def test_predict_endpoint_accepts_a_valid_load_when_artifact_exists():
    response = client.post("/predict", json={
        "pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360,
        "equipment": "Dry Van", "weight": 32000, "date": "2025-12-01",
    })
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert response.json()["predicted_rate"] > 0


def test_predict_endpoint_accepts_uppercase_city_names():
    """Location normalisation must fire before feature engineering."""
    response = client.post("/predict", json={
        "pickup": "LEXINGTON", "delivery": "FORT WAYNE", "distance": 360,
        "equipment": "DRY VAN", "weight": 32000, "date": "2025-12-01",
    })
    # 200 if model artifact exists, 503 if not — but never 422 (bad location)
    assert response.status_code in {200, 503}


def test_predict_endpoint_returns_422_for_unknown_city():
    """Unknown city must return a structured 422 / 503 with an explanatory message."""
    response = client.post("/predict", json={
        "pickup": "Gotham City", "delivery": "Fort Wayne", "distance": 360,
        "equipment": "Dry Van", "weight": 32000, "date": "2025-12-01",
    })
    # PredictionError (domain) maps to 422; before model loads it may be 503
    assert response.status_code in {422, 503}


def test_locations_endpoint_returns_all_cities():
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "Lexington" in data["locations"]
    assert "Fort Wayne" in data["locations"]
    assert "Dry Van" in data["equipment"]
    assert data["location_count"] == 64
    assert data["equipment_count"] == 3
