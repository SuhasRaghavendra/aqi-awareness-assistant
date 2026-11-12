"""Measure live single-row and batch latency against a real ``uvicorn`` process.

The offline risk card measures ``predict`` with a warm in-process model, which
does not prove that the HTTP service meets the Plan's latency budget. This
script starts the real ASGI app on a free port, waits for ``/ready``, sends a
warm-up pass, then records:

- single-row ``POST /predict`` server-reported ``latency_ms``;
- nine-city ``POST /predict/batch`` wall-clock latency per call.

``reports/metrics/latency_probe.json`` records p50/p95/p99 for both, the
process command line and a timestamp. The process exits non-zero when the
p95 budget (single < 50 ms, batch < 100 ms) is missed.

Usage:
    python scripts/latency_probe.py
    python scripts/latency_probe.py --iterations 500 --output reports/metrics/latency_probe.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports/metrics/latency_probe.json"

SINGLE_P95_BUDGET_MS = 50.0
BATCH_P95_BUDGET_MS = 100.0

CITIES = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Pune",
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _observation(city: str, date: str) -> dict[str, Any]:
    """A plausible, in-range observation for one city-day."""
    return {
        "city": city,
        "date": date,
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
        "precipitation_sum": 0.0,
        "fire_count_500km": 12.0,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles (the standard latency-report definition)."""
    ordered = sorted(values)
    count = len(ordered)

    def pick(quantile: float) -> float:
        index = min(count - 1, max(0, math.ceil(quantile * count) - 1))
        return round(float(ordered[index]), 4)

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "min": round(float(ordered[0]), 4),
        "max": round(float(ordered[-1]), 4),
        "mean": round(float(sum(ordered) / count), 4),
    }


def _wait_ready(client: httpx.Client, base_url: str, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            response = client.get(f"{base_url}/ready")
            if response.status_code == 200:
                return
            last_error = f"/ready returned {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:  # server still starting
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise RuntimeError(f"uvicorn did not become ready within {timeout_s:.0f}s ({last_error})")


def _timed_request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> float:
    started = time.perf_counter()
    response = client.request(method, url, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(f"{method} {url} returned {response.status_code}: {response.text[:300]}")
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Live uvicorn latency probe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--iterations", type=int, default=200, help="single-row requests")
    parser.add_argument("--batch-iterations", type=int, default=50, help="nine-row requests")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--date", default="2024-12-30", help="Observation date inside the seed")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--single-p95-ms", type=float, default=SINGLE_P95_BUDGET_MS)
    parser.add_argument("--batch-p95-ms", type=float, default=BATCH_P95_BUDGET_MS)
    args = parser.parse_args()

    port = args.port or _free_port()
    base_url = f"http://{args.host}:{port}"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.service.app:app",
        "--host",
        args.host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")] if env.get("PYTHONPATH") else [str(REPO_ROOT)]
    )

    print(f"Starting: {' '.join(command)}")
    log_path = Path(os.environ.get("TEMP", ".")) / "aqi_latency_probe_uvicorn.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    try:
        with httpx.Client(timeout=30.0) as client:
            _wait_ready(client, base_url, args.startup_timeout)
            print(f"Service ready at {base_url}")

            row = _observation("Delhi", args.date)
            batch = {"rows": [_observation(city, args.date) for city in CITIES]}

            for _ in range(max(1, args.warmup)):
                _timed_request(client, "POST", f"{base_url}/predict", json=row)

            single_server: list[float] = []
            single_wall: list[float] = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                response = client.post(f"{base_url}/predict", json=row)
                wall_ms = (time.perf_counter() - started) * 1000.0
                if response.status_code != 200:
                    raise RuntimeError(
                        f"POST /predict returned {response.status_code}: {response.text[:300]}"
                    )
                single_wall.append(wall_ms)
                single_server.append(float(response.json()["latency_ms"]))

            for _ in range(max(1, args.warmup // 2)):
                _timed_request(client, "POST", f"{base_url}/predict/batch", json=batch)

            batch_wall: list[float] = []
            batch_rows: list[int] = []
            for _ in range(args.batch_iterations):
                wall_ms = _timed_request(client, "POST", f"{base_url}/predict/batch", json=batch)
                batch_wall.append(wall_ms)
                batch_rows.append(len(CITIES))

            metadata = client.get(f"{base_url}/metadata").json()
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)

    single = _percentiles(single_server)
    single_http = _percentiles(single_wall)
    batch = _percentiles(batch_wall)
    passed = single["p95"] < args.single_p95_ms and batch["p95"] < args.batch_p95_ms

    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "method": "live uvicorn subprocess, HTTP requests, post-warmup",
        "transport": base_url,
        "command": " ".join(command),
        "date": args.date,
        "iterations": {
            "single": args.iterations,
            "batch": args.batch_iterations,
            "batch_rows": len(CITIES),
            "warmup": max(1, args.warmup),
        },
        "champion": metadata.get("model_name"),
        "bundle_version": metadata.get("model_version"),
        "single": {
            "server_latency_ms": single,
            "http_roundtrip_ms": single_http,
        },
        "batch_9_cities": {
            "http_roundtrip_ms": batch,
            "per_row_ms": {key: round(value / len(CITIES), 4) for key, value in batch.items()},
        },
        "budget_ms": {
            "single_p95": args.single_p95_ms,
            "batch_p95": args.batch_p95_ms,
        },
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report -> {args.output}")
    if not passed:
        raise SystemExit(
            "latency budget missed: "
            f"single p95 {single['p95']:.2f} ms (budget {args.single_p95_ms}), "
            f"batch p95 {batch['p95']:.2f} ms (budget {args.batch_p95_ms})"
        )


if __name__ == "__main__":
    main()
