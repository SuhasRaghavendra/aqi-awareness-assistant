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
