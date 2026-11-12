"""
scripts/smoke_test_deployed.py
------------------------------
Live production deployment smoke test.
Validates that a live URL (Render, Railway, Cloud Run, or localhost)
responds correctly to all core health and inference endpoints.

Usage:
    python scripts/smoke_test_deployed.py --url https://your-service.onrender.com
"""

from __future__ import annotations

import argparse
import sys

import httpx


def test_endpoint(client: httpx.Client, base_url: str) -> bool:
    print(f"=== Smoke Testing Deployed Service: {base_url} ===")

    # 1. Health check
    try:
        r = client.get(f"{base_url}/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert r.json().get("status") == "healthy"
        print("  [PASS] GET /health")
    except Exception as e:
        print(f"  [FAIL] GET /health: {e}")
        return False

    # 2. Ready check
    try:
        r = client.get(f"{base_url}/ready")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        print(f"  [PASS] GET /ready -> {r.json()}")
    except Exception as e:
        print(f"  [FAIL] GET /ready: {e}")
        return False

    # 3. Metadata
    try:
        r = client.get(f"{base_url}/metadata")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        print(
            f"  [PASS] GET /metadata -> Model: {data.get('model_name')}, Features: {data.get('features_count')}"
        )
    except Exception as e:
        print(f"  [FAIL] GET /metadata: {e}")
        return False

    # 4. Single prediction
    try:
        payload = {
            "city": "Delhi",
            "date": "2025-01-02",
            "aqi": 180.0,
            "pm25": 95.0,
            "pm10": 150.0,
            "no2": 55.0,
            "so2": 12.0,
            "co": 1.4,
            "o3": 30.0,
            "temperature_2m_mean": 14.0,
            "wind_speed_10m_mean": 3.5,
            "relative_humidity_2m_mean": 72.0,
        }
        r = client.post(f"{base_url}/predict", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        pred = r.json()
        assert "aqi_pred" in pred and "category" in pred and "aqi_interval_80" in pred
        assert pred["aqi_interval_80"]["lower"] <= pred["aqi_interval_80"]["upper"]
        print(
            f"  [PASS] POST /predict -> City: {pred['city']}, "
            f"Predicted AQI: {pred['aqi_pred']:.1f}, Category: {pred['category']}, "
            f"degraded={pred['degraded']}, history_days={pred['history_days']}"
        )
    except Exception as e:
        print(f"  [FAIL] POST /predict: {e}")
        return False

    # 4b. Schema and semantic errors
    try:
        bad_schema = {"city": "Delhi", "date": "not-a-date", "aqi": 100.0}
        assert client.post(f"{base_url}/predict", json=bad_schema).status_code == 422
        bad_value = {**payload, "aqi": -1.0}
        assert client.post(f"{base_url}/predict", json=bad_value).status_code == 400
        print("  [PASS] POST /predict error contract (422 schema, 400 impossible value)")
    except Exception as e:
        print(f"  [FAIL] POST /predict error contract: {e}")
        return False

    # 5. Prometheus Metrics
    try:
        r = client.get(f"{base_url}/metrics")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        print("  [PASS] GET /metrics (Prometheus exposition active)")
    except Exception as e:
        print(f"  [FAIL] GET /metrics: {e}")
        return False

    print("\nALL LIVE ENDPOINT CHECKS PASSED SUCCESSFULLY! 🚀")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live service smoke tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of service")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    with httpx.Client(timeout=15.0) as client:
        success = test_endpoint(client, base_url)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
