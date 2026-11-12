"""
tests/test_service.py
---------------------
FastAPI service integration tests using TestClient (no real model needed).

The tests mock predictor.is_ready() and predictor.predict_one/predict_batch so
the suite runs without the champion bundle being present on disk — suitable for
CI before dvc pull completes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.service.app import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan disabled (bundle load is mocked per test)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


VALID_PAYLOAD = {
    "city": "Delhi",
    "date": "2026-01-15",
    "aqi": 180.0,
    "pm25": 95.0,
    "pm10": 150.0,
    "no2": 55.0,
    "so2": 12.0,
    "co": 1.4,
    "o3": 30.0,
    "temperature_2m_mean": 14.0,
    "temperature_2m_min": 8.0,
    "temperature_2m_max": 20.0,
    "relative_humidity_2m_mean": 72.0,
    "wind_speed_10m_mean": 3.5,
    "wind_direction_10m_dominant": 210.0,
    "precipitation_sum": 0.0,
    "surface_pressure_mean": 1012.0,
    "fire_count_500km": 25.0,
    "fire_frp_sum_500km": 500.0,
    "aqi_lag_1": 185.0,
    "aqi_lag_3": 175.0,
    "aqi_lag_7": 160.0,
    "aqi_lag_14": 140.0,
}

MOCK_PREDICT_RESULT = {
    "predicted_aqi": 172.5,
    "predicted_category": "Moderate",
    "interval": {"lower": 152.2, "median": 172.5, "upper": 191.9},
    "model_name": "residual_lightgbm",
    "fallback_used": False,
}

MOCK_METADATA = {
    "model_name": "residual_lightgbm",
    "bundle_version": "1.0",
    "features_count": 103,
    "val_metrics": {"rmse": 20.1, "r2": 0.903},
    "test_metrics": {"rmse": 23.8, "r2": 0.878},
    "gate_status": {"G1": "pass", "G2": "pass", "G3": "fail", "G4": "pass", "G5": "pass"},
    "split": {"train_end": "2023-12-31", "val_end": "2024-12-31"},
    "git_commit": "abc1234",
    "created_at": "2026-11-10T22:53:52+05:30",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_always_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------

class TestReady:
    def test_ready_when_bundle_loaded(self, client):
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch(
                "src.service.predictor.get_bundle",
                return_value=_make_mock_bundle(),
            ),
        ):
            resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True
        assert body["model_name"] == "residual_lightgbm"
        assert body["features_count"] == 103

    def test_ready_503_when_bundle_missing(self, client):
        with patch("src.service.predictor.is_ready", return_value=False):
            resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["ready"] is False


# ---------------------------------------------------------------------------
# /metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_returns_model_info(self, client):
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch("src.service.predictor.get_metadata", return_value=MOCK_METADATA),
        ):
            resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_name"] == "residual_lightgbm"
        assert body["features_count"] == 103
        assert "val_metrics" in body
        assert "gate_status" in body

    def test_metadata_503_when_not_ready(self, client):
        with patch("src.service.predictor.is_ready", return_value=False):
            resp = client.get("/metadata")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_payload_returns_200(self, client):
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch("src.service.predictor.predict_one", return_value=MOCK_PREDICT_RESULT),
        ):
            resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["city"] == "Delhi"
        assert body["date"] == "2026-01-15"
        assert 0 <= body["predicted_aqi"] <= 500
        assert body["predicted_category"] in {
            "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
        }
        assert "interval" in body
        assert body["interval"]["lower"] <= body["interval"]["median"] <= body["interval"]["upper"]

    def test_missing_required_fields_returns_422(self, client):
        # city and date are required
        resp = client.post("/predict", json={"aqi": 100.0})
        assert resp.status_code == 422

    def test_invalid_city_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "city": "InvalidCity"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_aqi_out_of_range_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "aqi": -10.0}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_bad_date_format_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "date": "15-01-2026"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_503_when_bundle_not_ready(self, client):
        with patch("src.service.predictor.is_ready", return_value=False):
            resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_all_nine_cities_accepted(self, client):
        cities = [
            "Delhi", "Mumbai", "Bengaluru", "Chennai",
            "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Lucknow",
        ]
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch("src.service.predictor.predict_one", return_value=MOCK_PREDICT_RESULT),
        ):
            for city in cities:
                resp = client.post("/predict", json={**VALID_PAYLOAD, "city": city})
                assert resp.status_code == 200, f"Failed for city: {city}"

    def test_optional_fields_can_be_omitted(self, client):
        minimal = {"city": "Mumbai", "date": "2026-03-01", "aqi": 90.0}
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch("src.service.predictor.predict_one", return_value=MOCK_PREDICT_RESULT),
        ):
            resp = client.post("/predict", json=minimal)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_two_cities(self, client):
        rows = [
            {**VALID_PAYLOAD, "city": "Delhi"},
            {**VALID_PAYLOAD, "city": "Mumbai"},
        ]
        mock_batch = [
            {**MOCK_PREDICT_RESULT, "city": "Delhi", "date": "2026-01-15"},
            {**MOCK_PREDICT_RESULT, "city": "Mumbai", "date": "2026-01-15"},
        ]
        with (
            patch("src.service.predictor.is_ready", return_value=True),
            patch("src.service.predictor.predict_batch", return_value=mock_batch),
        ):
            resp = client.post("/predict/batch", json={"rows": rows})
        assert resp.status_code == 200
        body = resp.json()
        assert body["rows_processed"] == 2
        assert len(body["predictions"]) == 2

    def test_empty_batch_returns_422(self, client):
        resp = client.post("/predict/batch", json={"rows": []})
        assert resp.status_code == 422

    def test_batch_503_when_bundle_not_ready(self, client):
        with patch("src.service.predictor.is_ready", return_value=False):
            resp = client.post("/predict/batch", json={"rows": [VALID_PAYLOAD]})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Mock bundle helper
# ---------------------------------------------------------------------------

def _make_mock_bundle():
    """Return a minimal mock object satisfying predictor.get_bundle() usage."""
    from unittest.mock import MagicMock
    bundle = MagicMock()
    bundle.manifest = {
        "model": {"name": "residual_lightgbm"},
        "bundle_version": "1.0",
        "features_count": 103,
    }
    bundle.feature_count = 103
    return bundle

