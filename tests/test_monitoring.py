"""Unit and integration tests for Phase 6 live drift (PSI) and rolling MAE monitoring."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.service.app import app
from src.service.monitoring import ServiceMonitor, compute_psi, monitor


def test_compute_psi_low_for_matching_distribution() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=100.0, scale=25.0, size=400).tolist()
    psi = compute_psi(samples, mean=100.0, std=25.0)
    assert psi < 0.1


def test_compute_psi_high_for_shifted_distribution() -> None:
    rng = np.random.default_rng(7)
    shifted = rng.normal(loc=220.0, scale=25.0, size=200).tolist()
    psi = compute_psi(shifted, mean=100.0, std=25.0)
    assert psi > 0.2


def test_service_monitor_tracks_psi_and_rolling_mae() -> None:
    m = ServiceMonitor(psi_window=50, mae_window=14, val_mae_baseline=13.23)
    m.configure(
        reference_stats={"features": {"aqi": {"mean": 100.0, "std": 30.0}}},
        val_mae_baseline=13.23,
    )

    # Day 1: predict 150 for 2025-01-02 (no prior prediction yet -> realized_err is None)
    psi1, err1, mae1 = m.observe(
        payload={"aqi": 140.0},
        city="Delhi",
        date="2025-01-01",
        target_date="2025-01-02",
        aqi_pred=150.0,
    )
    assert err1 is None
    assert mae1 is None
    assert psi1 == {}  # fewer than 10 samples

    # Day 2: observed AQI is 162.0 -> realized error |162 - 150| = 12.0
    _, err2, mae2 = m.observe(
        payload={"aqi": 162.0},
        city="Delhi",
        date="2025-01-02",
        target_date="2025-01-03",
        aqi_pred=170.0,
    )
    assert err2 == 12.0
    assert mae2 == 12.0

    # Day 3: observed AQI is 160.0 -> realized error |160 - 170| = 10.0, rolling MAE = 11.0
    _, err3, mae3 = m.observe(
        payload={"aqi": 160.0},
        city="Delhi",
        date="2025-01-03",
        target_date="2025-01-04",
        aqi_pred=165.0,
    )
    assert err3 == 10.0
    assert mae3 == 11.0


def test_metrics_endpoint_exposes_drift_and_error_gauges() -> None:
    monitor.reset()
    with TestClient(app) as client:
        # Send 12 consecutive days for Delhi so both PSI (>=10 samples) and realized MAE fire
        for day in range(1, 13):
            payload = {
                "city": "Delhi",
                "date": f"2025-01-{day:02d}",
                "aqi": 180.0 + day * 2.0,
                "temperature_2m_mean": 15.0,
                "wind_speed_10m_mean": 6.0,
                "relative_humidity_2m_mean": 70.0,
                "fire_count_500km": 20.0,
                "fire_frp_sum_500km": 100.0,
            }
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200

        metrics_text = client.get("/metrics").text
        assert "aqi_feature_drift_psi" in metrics_text
        assert "aqi_realized_abs_error" in metrics_text
        assert "aqi_rolling_14d_mae" in metrics_text


def test_monitoring_feedback_matches_once_and_updates_summary() -> None:
    with TestClient(app) as client:
        initial = client.get("/monitoring/summary").json()
        assert initial["ready"]
        assert len(initial["feature_drift"]) == 6
        assert initial["prediction_drift"]["psi"] is None
        assert initial["mae_alert_threshold"] == 19.85

        forecast = client.post(
            "/predict", json={"city": "Delhi", "date": "2025-01-01", "aqi": 180}
        ).json()
        feedback = {"city": "Delhi", "date": "2025-01-02", "actual_aqi": 220}
        assert client.post("/feedback", json={**feedback, "actual_aqi": -1}).status_code == 422
        assert not client.post("/feedback", json={**feedback, "city": "Mumbai"}).json()["matched"]
        result = client.post("/feedback", json=feedback).json()
        error = round(abs(feedback["actual_aqi"] - forecast["aqi_pred"]), 4)
        assert result == {"matched": True, "realized_abs_error": error, "rolling_14d_mae": error}
        assert not client.post("/feedback", json=feedback).json()["matched"]

        summary = client.get("/monitoring/summary").json()
        assert summary["total_predictions"] == 1
        assert summary["per_city_mae"]["Delhi"] == {"mae": error, "observations": 1}
        assert summary["rolling_14d_mae"] == error
        assert "aqi_global_rolling_14d_mae" in client.get("/metrics").text


def test_prediction_drift_and_optional_inputs_follow_prediction_window() -> None:
    m = ServiceMonitor(psi_window=10)
    reference = {"mean": 100, "std": 20}
    m.configure({"features": {"wind_speed_10m_mean": reference}, "target_aqi": reference})
    for _ in range(10):
        m.observe(
            {"wind_speed_10m_mean": 100},
            "Delhi",
            "2025-01-01",
            "2025-01-02",
            250,
            out_of_range_fields=["aqi", "wind_speed_10m_mean"],
        )
    summary = m.summary()
    assert summary["prediction_drift"]["status"] == "RED"
    assert summary["retrain_recommended"]
    assert summary["feature_drift"]["wind_speed_10m_mean"]["z_score"] == 0
    assert summary["out_of_range_predictions"] == 10
    assert summary["out_of_range_counts"] == {"aqi": 10, "wind_speed_10m_mean": 10}
    m.observe({}, "Delhi", "2025-01-01", "2025-01-02", 250)
    assert m.summary()["feature_drift"]["wind_speed_10m_mean"]["psi"] is None


def test_rolling_mae_expires_by_target_date_and_alerts_on_sustained_errors() -> None:
    m = ServiceMonitor()
    for day in (1, 2, 3):
        m.observe({}, "Delhi", f"2025-01-0{day}", f"2025-01-0{day + 1}", 100)
        assert m.observe_actual("Delhi", f"2025-01-0{day + 1}", 130) == (30, 30)
    assert m.summary()["breached_cities"] == ["Delhi"]
    assert m.summary()["retrain_recommended"]

    m.observe({}, "Mumbai", "2025-01-16", "2025-01-17", 200)
    m.observe_actual("Mumbai", "2025-01-17", 201)
    summary = m.summary()
    assert summary["per_city_mae"] == {
        "Delhi": {"mae": 30, "observations": 1},
        "Mumbai": {"mae": 1, "observations": 1},
    }
    assert summary["rolling_14d_mae"] == 15.5
    assert not summary["retrain_recommended"]
