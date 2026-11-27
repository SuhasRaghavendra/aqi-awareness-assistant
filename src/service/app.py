"""
src/service/app.py
------------------
FastAPI application for the Indian AQI Awareness Assistant.

Endpoints
---------
GET  /health
GET  /ready
GET  /metadata
POST /predict
POST /predict/batch
GET  /metrics
GET  /monitoring/summary
POST /feedback
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.service import middleware as mw
from src.service import predictor
from src.service.monitoring import monitor
from src.service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    validate_semantics,
)
from src.utils.logging_utils import (
    configure_json_logging,
    get_logger,
    log_audit_event,
)

# --------------------------------------------------
# Logging
# --------------------------------------------------

configure_json_logging()
logger = get_logger(__name__)


# --------------------------------------------------
# App lifecycle
# --------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model bundle at startup."""

    bundle_dir = os.environ.get("BUNDLE_DIR", "artifacts/champion")
    config_dir = os.environ.get("CONFIG_DIR")

    predictor.configure(bundle_dir, config_dir)
    predictor.load()

    monitor.reset()

    if predictor.is_ready():
        bundle = predictor.get_bundle()

        val_mae = bundle.manifest["model"].get("val_metrics", {}).get("mae")

        monitor.configure(
            reference_stats=bundle.reference_stats,
            val_mae_baseline=(float(val_mae) if val_mae is not None else None),
        )

    yield


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AQI Awareness Assistant",
    description=(
        "Production-grade next-day Air Quality Index forecasting for 9 Indian cities. "
        "Champion model: residual_lightgbm_monotone (val R2=0.90, test R2=0.88). "
        "Data sources: XKDR CAAQM, ERA5 weather, NASA FIRMS, MoRTH Vahan."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(mw.TimingMiddleware)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def _require_ready() -> None:
    """Check whether the prediction model is loaded."""

    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded. Try again shortly.",
        )


def _record_prediction(
    payload: dict,
    result: dict,
    request_id: str | None = None,
) -> None:
    """Record prediction metrics, monitoring and audit events."""

    out_of_range = result.pop("out_of_range_fields", [])

    # Record Prometheus prediction metrics
    mw.record_prediction(
        city=result["city"],
        fallback=result["fallback_used"],
        aqi=result["aqi_pred"],
        out_of_range_fields=out_of_range,
        model_used=result["model_used"],
    )

    # Update monitoring statistics
    psi_updates, realized_err, _ = monitor.observe(
        payload=payload,
        city=result["city"],
        date=result["date"],
        target_date=result["target_date"],
        aqi_pred=result["aqi_pred"],
        out_of_range_fields=out_of_range,
    )

    mw.record_monitoring(
        city=result["city"],
        psi_by_feature=psi_updates,
        prediction_psi=monitor.prediction_psi(),
        realized_abs_error=realized_err,
    )

    if realized_err is not None:
        mw.record_accuracy_summary(monitor.summary())

    # Write structured prediction audit log (stdout JSON + JSONL audit file)
    log_audit_event(
        "prediction_served",
        request_id=request_id,
        city=result["city"],
        date=result["date"],
        target_date=result["target_date"],
        input_aqi=payload.get("aqi"),
        aqi_pred=result["aqi_pred"],
        category=result.get("category"),
        interval_80=result.get("aqi_interval_80"),
        model_name=result.get("model_name"),
        model_version=result.get("model_version"),
        alias=result.get("alias"),
        model_used=result["model_used"],
        fallback_used=result["fallback_used"],
        degraded=result.get("degraded"),
        out_of_range_fields=out_of_range,
        latency_ms=result.get("latency_ms"),
    )


def _validate(observation) -> None:
    """Reject physically impossible observation values."""

    problems = validate_semantics(observation)

    if problems:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Impossible observation values",
                "errors": problems,
            },
        )


# --------------------------------------------------
# General route
# --------------------------------------------------


@app.get("/", tags=["General"])
async def root():
    """Welcome page with API links."""

    return {
        "message": "Welcome to the AQI Awareness Assistant API",
        "docs_url": "/docs",
        "health_url": "/health",
        "ready_url": "/ready",
        "metadata_url": "/metadata",
        "monitoring_url": "/monitoring/summary",
    }


# --------------------------------------------------
# Health and readiness routes
# --------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Observability"],
)
async def health() -> HealthResponse:
    """Check whether the application is running."""

    return HealthResponse()


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["Observability"],
)
async def ready() -> ReadyResponse:
    """Check whether the model is ready."""

    if not predictor.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "model_name": None,
                "model_version": None,
                "features_count": None,
                "history_seed_days": None,
            },
        )

    bundle = predictor.get_bundle()

    return ReadyResponse(
        ready=True,
        model_name=bundle.manifest["model"]["name"],
        model_version=str(bundle.manifest.get("bundle_version", "?")),
        features_count=bundle.feature_count,
        history_seed_days=int(bundle.manifest.get("history_seed_days", 0)),
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["Observability"],
)
async def metadata() -> MetadataResponse:
    """Return model information and training metrics."""

    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded yet.",
        )

    return MetadataResponse(**predictor.get_metadata())


# --------------------------------------------------
# Single prediction route
# --------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
)
async def predict(
    request: PredictRequest,
    http_request: Request,
) -> PredictResponse:
    """Predict next-day AQI for a single city."""

    _require_ready()
    _validate(request)

    payload = request.model_dump()

    result = predictor.predict_one(payload)

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    _record_prediction(
        payload,
        result,
        request_id=request_id,
    )

    return PredictResponse(**result)


# --------------------------------------------------
# Batch prediction route
# --------------------------------------------------


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
)
async def predict_batch(
    request: BatchPredictRequest,
    http_request: Request,
) -> BatchPredictResponse:
    """Perform multiple AQI predictions in one request."""

    _require_ready()

    # Validate every input row
    for index, row in enumerate(request.rows):

        problems = validate_semantics(row)

        if problems:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": (f"Impossible observation values in row {index}"),
                    "errors": problems,
                },
            )

    rows = [row.model_dump() for row in request.rows]

    # Generate batch predictions
    results = predictor.predict_batch(rows)

    predictions = []

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    # Record every prediction separately
    for row_payload, result in zip(
        rows,
        results,
        strict=False,
    ):

        _record_prediction(
            row_payload,
            result,
            request_id=request_id,
        )

        predictions.append(PredictResponse(**result))

    return BatchPredictResponse(
        predictions=predictions,
        rows_processed=len(predictions),
    )


# --------------------------------------------------
# Prometheus metrics route
# --------------------------------------------------


@app.get(
    "/metrics",
    tags=["Observability"],
)
async def metrics():
    """Expose application metrics for Prometheus."""

    return mw.metrics_response()


# --------------------------------------------------
# Monitoring summary route
# --------------------------------------------------


@app.get(
    "/monitoring/summary",
    tags=["Observability"],
)
async def monitoring_summary():
    """Return current drift and accuracy statistics."""

    return {
        **monitor.summary(),
        "ready": predictor.is_ready(),
    }


# --------------------------------------------------
# Feedback route
# --------------------------------------------------


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Observability"],
)
async def feedback(
    request: FeedbackRequest,
) -> FeedbackResponse:
    """Submit actual AQI values to evaluate prediction accuracy."""

    _require_ready()

    error, mae = monitor.observe_actual(
        request.city,
        request.date.isoformat(),
        request.actual_aqi,
    )

    if error is not None:

        mw.record_monitoring(
            request.city,
            realized_abs_error=error,
        )

        mw.record_accuracy_summary(monitor.summary())

    return FeedbackResponse(
        matched=error is not None,
        realized_abs_error=error,
        rolling_14d_mae=mae,
    )
