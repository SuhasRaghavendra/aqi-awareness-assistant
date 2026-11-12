"""
src/service/app.py
------------------
FastAPI application for the Indian AQI Awareness Assistant.

Endpoints
---------
GET  /health          Liveness probe.
GET  /ready           Readiness probe (model loaded?).
GET  /metadata        Model version, metrics, gates, feature count.
POST /predict         Single city-day prediction.
POST /predict/batch   Vectorised batch (up to 63 rows — 9 cities × 7 days).
GET  /metrics         Prometheus exposition page.

Usage (local dev)
-----------------
    uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

Production (Docker entrypoint)
------------------------------
    uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import logging
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
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion bundle at startup; nothing to teardown."""
    bundle_dir = os.environ.get("BUNDLE_DIR", "artifacts/champion")
    predictor.configure(bundle_dir)
    predictor.load()
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AQI Awareness Assistant",
    description=(
        "Production-grade next-day Air Quality Index forecasting for 9 Indian cities. "
        "Champion model: residual_lightgbm (val R²=0.90, test R²=0.88). "
        "Data sources: XKDR CAAQM, ERA5 weather, NASA FIRMS, MoRTH Vahan."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach timing + Prometheus middleware
app.add_middleware(mw.TimingMiddleware)


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
    """Liveness probe — always returns 200 if the process is alive."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["Observability"])
async def ready() -> ReadyResponse:
    """Readiness probe — returns ready=False (503) until bundle is loaded."""
    if not predictor.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "model_name": None, "features_count": None, "bundle_version": None},
        )
    bundle = predictor.get_bundle()
    return ReadyResponse(
        ready=True,
        model_name=bundle.manifest["model"]["name"],
        features_count=bundle.feature_count,
        bundle_version=bundle.manifest.get("bundle_version"),
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

    Returns:
    - **predicted_aqi**: continuous AQI value for tomorrow.
    - **predicted_category**: CPCB health risk category.
    - **interval**: 10th / 50th / 90th percentile prediction interval.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded. Try again shortly.",
        )
    result = predictor.predict_one(request.model_dump())
    mw.record_prediction(city=request.city, fallback=result["fallback_used"])
    return PredictResponse(city=request.city, date=request.date, **result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Vectorised batch inference for multiple city-days in one call.

    Accepts up to 63 rows (9 cities × 7 days). Ideal for dashboard use.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded. Try again shortly.",
        )
    rows = [r.model_dump() for r in request.rows]
    results = predictor.predict_batch(rows)
    predictions = [PredictResponse(**r) for r in results]
    for r in results:
        mw.record_prediction(city=r["city"], fallback=r["fallback_used"])
    return BatchPredictResponse(predictions=predictions, rows_processed=len(predictions))


@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics exposition endpoint."""
    return mw.metrics_response()

