"""
scripts/smoke_test.py
---------------------
Deployment smoke test for the AQI Forecasting Inference Service.
Validates that a live URL (Render, Railway, Cloud Run, or localhost) responds
correctly to health, readiness, inference contract, and documentation endpoints.

Standard library only (no external dependencies).

Usage:
    python scripts/smoke_test.py --url https://your-service.onrender.com
    python scripts/smoke_test.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from typing import Any

MAX_RETRY_SECONDS = 150
POLL_INTERVAL_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 15.0


def make_request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> tuple[int, Any]:
    """Execute an HTTP request using Python standard library urllib."""
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    payload_bytes: bytes | None = None
    if data is not None:
        payload_bytes = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers=req_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = response.read().decode("utf-8", errors="replace")
            if "application/json" in content_type:
                try:
                    return status_code, json.loads(raw)
                except json.JSONDecodeError:
                    return status_code, raw
            return status_code, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, raw


def wait_for_service(base_url: str, max_seconds: int = MAX_RETRY_SECONDS) -> bool:
    """Poll GET /health until it returns 200 or the max_seconds timeout expires.

    Accounts for free-tier PaaS cold starts (e.g. Render spin-up from idle).
    """
    print(f"Polling {base_url}/health (up to {max_seconds}s for cold start)...")
    start = time.monotonic()
    health_url = f"{base_url}/health"

    while True:
        elapsed = time.monotonic() - start
        if elapsed > max_seconds:
            print(
                f"[FAIL] Service at {base_url} did not respond within {max_seconds} seconds.",
                file=sys.stderr,
            )
            return False

        try:
            req = urllib.request.Request(
                health_url,
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    print(f"  [PASS] Service responded 200 OK to /health after {elapsed:.1f}s.")
                    return True
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            pass

        time.sleep(POLL_INTERVAL_SECONDS)


def run_smoke_test(base_url: str, max_seconds: int = MAX_RETRY_SECONDS) -> bool:
    """Execute end-to-end smoke test suite on the deployed target."""
    print("=" * 70)
    print(f"RUNNING SMOKE TEST: {base_url}")
    print("=" * 70)

    # 1. Wait for cold-start wake-up and verify GET /health
    if not wait_for_service(base_url, max_seconds=max_seconds):
        return False

    status_code, body = make_request(f"{base_url}/health", method="GET")
    if status_code != 200:
        print(f"[FAIL] GET /health expected 200, got {status_code}: {body}", file=sys.stderr)
        return False
    print(f"  [PASS] GET /health -> 200 OK: {body}")

    # 2. Verify GET /ready
    ready_start = time.monotonic()
    ready_ok = False
    ready_url = f"{base_url}/ready"
    last_ready_res: Any = None

    while time.monotonic() - ready_start <= 30.0:
        status_code, body = make_request(ready_url, method="GET")
        last_ready_res = body
        if status_code == 200:
            ready_ok = True
            print(f"  [PASS] GET /ready -> 200 OK: {body}")
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if not ready_ok:
        print(
            f"[FAIL] GET /ready did not return 200. Last response: {last_ready_res}",
            file=sys.stderr,
        )
        return False

    # 3. POST /predict with a valid payload built from src/service/schemas.py
    predict_url = f"{base_url}/predict"
    valid_payload = {
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
        "temperature_2m_min": 8.0,
        "temperature_2m_max": 20.0,
        "relative_humidity_2m_mean": 72.0,
        "wind_speed_10m_mean": 3.5,
        "precipitation_sum": 0.0,
        "surface_pressure_mean": 1012.0,
        "fire_count_500km": 15.0,
        "fire_frp_sum_500km": 120.0,
        "fire_dist_min_km": 45.0,
    }

    status_code, body = make_request(predict_url, method="POST", data=valid_payload)
    if status_code != 200:
        print(
            f"[FAIL] POST /predict expected status 200, got {status_code}: {body}",
            file=sys.stderr,
        )
        return False

    if not isinstance(body, dict):
        print(
            f"[FAIL] POST /predict response is not a valid JSON dictionary: {body}",
            file=sys.stderr,
        )
        return False

    if "aqi_pred" not in body:
        print(
            f"[FAIL] POST /predict response missing required field 'aqi_pred': {body}",
            file=sys.stderr,
        )
        return False

    try:
        aqi_pred = float(body["aqi_pred"])
    except (TypeError, ValueError):
        print(
            f"[FAIL] 'aqi_pred' could not be parsed as float: {body.get('aqi_pred')}",
            file=sys.stderr,
        )
        return False

    if not math.isfinite(aqi_pred):
        print(
            f"[FAIL] 'aqi_pred' is not finite: {aqi_pred}",
            file=sys.stderr,
        )
        return False

    if not (0.0 <= aqi_pred <= 500.0):
        print(
            f"[FAIL] 'aqi_pred' must be between 0 and 500, got {aqi_pred}",
            file=sys.stderr,
        )
        return False

    category = body.get("category", "unknown")
    interval = body.get("aqi_interval_80", {})
    print(
        f"  [PASS] POST /predict -> 200 OK: aqi_pred={aqi_pred:.2f} (finite, [0, 500]), "
        f"category='{category}', 80% interval={interval}"
    )

    # 4. POST /predict with missing required field -> expect 422
    missing_payload = {
        "city": "Delhi",
        "date": "2025-01-02",
        # 'aqi' field is intentionally omitted
    }
    status_code, body = make_request(predict_url, method="POST", data=missing_payload)
    if status_code != 422:
        print(
            f"[FAIL] POST /predict with missing field expected status 422, got {status_code}: {body}",
            file=sys.stderr,
        )
        return False
    print("  [PASS] POST /predict with missing field -> 422 Unprocessable Entity (schema enforced)")

    # 5. GET /docs -> expect 200
    docs_url = f"{base_url}/docs"
    status_code, _ = make_request(docs_url, method="GET")
    if status_code != 200:
        print(
            f"[FAIL] GET /docs expected status 200, got {status_code}",
            file=sys.stderr,
        )
        return False
    print("  [PASS] GET /docs -> 200 OK (Swagger UI accessible)")

    print("=" * 70)
    print("ALL SMOKE TEST ENDPOINT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live service smoke tester (Python standard library only)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of service (e.g. https://your-service.onrender.com)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MAX_RETRY_SECONDS,
        help=f"Max retry seconds for cold start (default: {MAX_RETRY_SECONDS}s)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    success = run_smoke_test(base_url, max_seconds=args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
