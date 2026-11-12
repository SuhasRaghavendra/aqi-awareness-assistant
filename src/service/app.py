"""
src/service/app.py
------------------
FastAPI application for the Indian AQI Awareness Assistant.

Endpoints
---------
GET  /health          Liveness probe.
GET  /ready           Readiness probe (bundle + scaler + history seed loaded?).
GET  /metadata        Model version, metrics, gates, feature count.
POST /predict         Single city-day prediction.
POST /predict/batch   Vectorised batch (up to 63 rows - 9 cities x 7 days).
GET  /metrics         Prometheus exposition page.

Usage (local dev)
-----------------
    uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

Production (Docker entrypoint)
------------------------------
    uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.service import middleware as mw
from src.service import predictor
from src.service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    validate_semantics,
)
from src.utils.logging_utils import configure_json_logging, get_logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_json_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion bundle at startup; nothing to tear down."""
    bundle_dir = os.environ.get("BUNDLE_DIR", "artifacts/champion")
    config_dir = os.environ.get("CONFIG_DIR")
    predictor.configure(bundle_dir, config_dir)
    predictor.load()
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AQI Awareness Assistant",
    description=(
        "Production-grade next-day Air Quality Index forecasting for 9 Indian cities. "
        "Champion model: residual_lightgbm (val R2=0.90, test R2=0.88). "
        "Data sources: XKDR CAAQM, ERA5 weather, NASA FIRMS, MoRTH Vahan."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach timing + request-id + Prometheus middleware
app.add_middleware(mw.TimingMiddleware)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ready() -> None:
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded. Try again shortly.",
        )


def _validate(observation) -> None:
    """Reject physically impossible values with HTTP 400 (not schema 422)."""
    problems = validate_semantics(observation)
    if problems:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Impossible observation values", "errors": problems},
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["General"])
async def root():
    """Welcome page with quick links to API docs and health check."""
    return {
        "message": "Welcome to the AQI Awareness Assistant API",
        "docs_url": "/docs",
        "health_url": "/health",
        "ready_url": "/ready",
        "metadata_url": "/metadata",
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health() -> HealthResponse:
    """Liveness probe - always returns 200 if the process is alive."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["Observability"])
async def ready() -> ReadyResponse:
    """Readiness probe - returns ready=False (503) until the bundle is loaded."""
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


@app.get("/metadata", response_model=MetadataResponse, tags=["Observability"])
async def metadata() -> MetadataResponse:
    """Exposes model version, training metrics, gate status, and feature count."""
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded yet.",
        )
    return MetadataResponse(**predictor.get_metadata())


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict next-day AQI and CPCB category for a single city-day.

    The request carries today's raw observations only; calendar, city-static,
    vehicle/IIP, GRAP and lag features are rebuilt server-side from the seeded
    history (or the optional stateless ``history`` array).

    Returns:
    - **aqi_pred**: continuous AQI value for tomorrow.
    - **category**: CPCB health-risk category.
    - **aqi_interval_80**: 10th / 50th / 90th percentile prediction interval.
    - **warnings** / **degraded**: out-of-range inputs and short-history flags.
    """
    _require_ready()
    _validate(request)
    result = predictor.predict_one(request.model_dump())
    out_of_range = result.pop("out_of_range_fields", [])
    mw.record_prediction(
        city=result["city"],
        fallback=result["fallback_used"],
        aqi=result["aqi_pred"],
        out_of_range_fields=out_of_range,
        model_used=result["model_used"],
    )
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Vectorised batch inference for multiple city-days in one call.

    Accepts up to 63 rows (9 cities x 7 days). Ideal for dashboard use. Each
    row is inferred in order, so a chronological per-city batch reuses the
    rolling history correctly.
    """
    _require_ready()
    for index, row in enumerate(request.rows):
        problems = validate_semantics(row)
        if problems:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": f"Impossible observation values in row {index}",
                    "errors": problems,
                },
            )
    rows = [r.model_dump() for r in request.rows]
    results = predictor.predict_batch(rows)
    predictions = []
    for result in results:
        out_of_range = result.pop("out_of_range_fields", [])
        mw.record_prediction(
            city=result["city"],
            fallback=result["fallback_used"],
            aqi=result["aqi_pred"],
            out_of_range_fields=out_of_range,
            model_used=result["model_used"],
        )
        predictions.append(PredictResponse(**result))
    return BatchPredictResponse(predictions=predictions, rows_processed=len(predictions))


@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics exposition endpoint."""
    return mw.metrics_response()
