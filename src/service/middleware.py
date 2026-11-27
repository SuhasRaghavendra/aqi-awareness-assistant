"""
src/service/middleware.py
-------------------------
Request-id propagation, JSON request logging, timing, and Prometheus
instrumentation middleware.

Prometheus metrics are exposed at GET /metrics (pulled by a Prometheus scraper).
The prediction histogram and out-of-range counter are recorded by the route
handlers through :func:`record_prediction`.

Requires: prometheus-client (listed in requirements-serve.txt).
"""

from __future__ import annotations

import time
import uuid
from typing import get_args

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.service.monitoring import MONITORED_FEATURES
from src.service.schemas import CITIES
from src.utils.logging_utils import get_logger, log_audit_event

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# ---------------------------------------------------------------------------
# Optional Prometheus - degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
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
        ["city", "fallback", "model_used"],
    )
    PREDICTION_HISTOGRAM = Histogram(
        "aqi_prediction_value",
        "Distribution of predicted next-day AQI values",
        ["city"],
        buckets=(0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    )
    OUT_OF_RANGE_COUNT = Counter(
        "aqi_out_of_range_total",
        "Predictions whose inputs fell outside the training range",
        ["field"],
    )
    FEATURE_DRIFT_PSI = Gauge(
        "aqi_feature_drift_psi",
        "Rolling Population Stability Index (PSI) per input feature (warn > 0.2)",
        ["feature"],
    )
    PREDICTION_DRIFT_PSI = Gauge(
        "aqi_prediction_drift_psi",
        "Rolling PSI of forecasts compared with the training target AQI distribution",
    )
    ROLLING_GLOBAL_MAE = Gauge(
        "aqi_global_rolling_14d_mae",
        "Rolling 14-day MAE across all matched city-day observations",
    )
    REALIZED_ABS_ERROR = Gauge(
        "aqi_realized_abs_error",
        "Most recent realized absolute AQI prediction error per city",
        ["city"],
    )
    ROLLING_14D_MAE = Gauge(
        "aqi_rolling_14d_mae",
        "Rolling 14-day realized MAE per city (alert > 1.5x val MAE)",
        ["city"],
    )

except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class TimingMiddleware(BaseHTTPMiddleware):
    """Attach a request id, emit one JSON log line and record metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - log then let FastAPI handle it
            elapsed = time.perf_counter() - start
            log_audit_event(
                "http_request",
                level=40,
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                latency_ms=round(elapsed * 1000.0, 3),
                error=str(exc),
            )
            if _PROM_AVAILABLE:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code="500",
                ).inc()
            raise

        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code

        response.headers[REQUEST_ID_HEADER] = request_id

        log_audit_event(
            "http_request",
            request_id=request_id,
            method=method,
            path=endpoint,
            status_code=status_code,
            latency_ms=round(elapsed * 1000.0, 3),
        )

        if _PROM_AVAILABLE:
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
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


def record_prediction(
    city: str,
    fallback: bool,
    aqi: float | None = None,
    out_of_range_fields: list[str] | tuple[str, ...] = (),
    model_used: str | None = None,
) -> None:
    """Record prediction count, value histogram, and out-of-range inputs."""
    if not _PROM_AVAILABLE:
        return
    PREDICTION_COUNT.labels(
        city=city,
        fallback=str(fallback),
        model_used=model_used or "unknown",
    ).inc()
    if aqi is not None:
        PREDICTION_HISTOGRAM.labels(city=city).observe(max(float(aqi), 0.0))
    for field in out_of_range_fields:
        OUT_OF_RANGE_COUNT.labels(field=field).inc()


def record_monitoring(
    city: str,
    psi_by_feature: dict[str, float] | None = None,
    realized_abs_error: float | None = None,
    rolling_14d_mae: float | None = None,
    prediction_psi: float | None = None,
) -> None:
    """Update Prometheus drift (PSI) and live realized-error gauges."""
    if not _PROM_AVAILABLE:
        return
    if psi_by_feature is not None:
        for feature in MONITORED_FEATURES:
            FEATURE_DRIFT_PSI.labels(feature=feature).set(psi_by_feature.get(feature, float("nan")))
    if prediction_psi is not None:
        PREDICTION_DRIFT_PSI.set(prediction_psi)
    if realized_abs_error is not None:
        REALIZED_ABS_ERROR.labels(city=city).set(float(realized_abs_error))
    if rolling_14d_mae is not None:
        ROLLING_14D_MAE.labels(city=city).set(float(rolling_14d_mae))


def record_accuracy_summary(summary: dict) -> None:
    """Publish the bounded calendar window, including expired city series."""
    if not _PROM_AVAILABLE:
        return
    cities = summary["per_city_mae"]
    for city in get_args(CITIES):
        ROLLING_14D_MAE.labels(city=city).set(cities.get(city, {}).get("mae", float("nan")))
    global_mae = summary["rolling_14d_mae"]
    ROLLING_GLOBAL_MAE.set(global_mae if global_mae is not None else float("nan"))
