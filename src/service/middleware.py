"""
src/service/middleware.py
-------------------------
Request logging, timing, and Prometheus instrumentation middleware.

Prometheus metrics are exposed at GET /metrics (pulled by a Prometheus scraper)
and are also available via the starlette_exporter ASGI middleware that wraps
the app in app.py.

Requires: prometheus-client (listed in requirements-serve.txt).
"""

from __future__ import annotations

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Prometheus — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True

    REQUEST_COUNT = Counter(
        "aqi_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    REQUEST_LATENCY = Histogram(
        "aqi_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    PREDICTION_COUNT = Counter(
        "aqi_predictions_total",
        "Total AQI predictions served",
        ["city", "fallback"],
    )

except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------

class TimingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status = response.status_code

        logger.info(
            "%s %s → %d  (%.1f ms)",
            method,
            endpoint,
            status,
            elapsed * 1000,
        )

        if _PROM_AVAILABLE:
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status_code=str(status)
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)

        return response


# ---------------------------------------------------------------------------
# /metrics handler (registered as a plain route in app.py)
# ---------------------------------------------------------------------------

def metrics_response() -> Response:
    """Return a Prometheus text exposition page."""
    if not _PROM_AVAILABLE:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
            status_code=200,
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Helper: record a prediction event
# ---------------------------------------------------------------------------

def record_prediction(city: str, fallback: bool) -> None:
    if _PROM_AVAILABLE:
        PREDICTION_COUNT.labels(city=city, fallback=str(fallback)).inc()

