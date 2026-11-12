"""
src/service/schemas.py
----------------------
Pydantic v2 request / response models for the AQI FastAPI service.

The request carries today's raw observations only (AQI, weather, fires).  All
calendar, city-static, vehicle/IIP, GRAP and lag features are rebuilt on the
server with the same code path as offline training.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, Field

AQICategory = Literal["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

CITIES = Literal[
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Pune",
    "Ahmedabad",
    "Lucknow",
]


# ---------------------------------------------------------------------------
# Raw observation payload
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """One city-day of raw observations.

    Used both for the request body and for the optional stateless ``history``
    array (previous days needed to rebuild lag/rolling features).
    """

    city: CITIES | None = Field(None, description="Defaults to the request city.")
    date: date_type = Field(..., description="Observation date (YYYY-MM-DD).")

    # --- Current-day pollutant observations (context only; not model features) ---
    pm25: float | None = None
    pm10: float | None = None
    no2: float | None = None
    so2: float | None = None
    co: float | None = None
    o3: float | None = None

    # --- Today's AQI (required model feature) ---
    aqi: float = Field(..., description="Current-day CPCB AQI.")

    # --- Weather (ERA5 / Open-Meteo) ---
    temperature_2m_mean: float | None = None
    temperature_2m_min: float | None = None
    temperature_2m_max: float | None = None
    relative_humidity_2m_mean: float | None = None
    wind_speed_10m_mean: float | None = None
    wind_direction_10m_dominant: float | None = None
    precipitation_sum: float | None = None
    surface_pressure_mean: float | None = None

    # --- Fire events (NASA FIRMS) ---
    fire_count_500km: float | None = None
    fire_frp_sum_500km: float | None = None
    fire_dist_min_km: float | None = None

    # --- Optional policy/server-side overrides ---
    grap_stage: int | None = Field(None, ge=0, le=4, description="GRAP stage 0-4 (Delhi).")
    festival_flag: int | None = Field(None, ge=0, le=1, description="Festival day flag.")

    # --- Slow-moving reference values (history rows may carry these) ---
    diesel_frac: float | None = Field(None, ge=0, le=1)
    cng_frac: float | None = Field(None, ge=0, le=1)
    ev_frac: float | None = Field(None, ge=0, le=1)
    two_wheeler_frac: float | None = Field(None, ge=0, le=1)
    vehicles_per_lakh: float | None = Field(None, ge=0)
    iip_manufacturing_index: float | None = None

    model_config = {"extra": "forbid"}


class PredictRequest(Observation):
    """Single city-day prediction request with optional stateless history."""

    city: CITIES = Field(..., description="One of the 9 supported Indian cities.")
    history: list[Observation] | None = Field(
        None,
        description=(
            "Optional previous observations (oldest first) used to rebuild lag/rolling "
            "features in stateless deployments. When omitted the service uses its "
            "seeded per-city history."
        ),
        max_length=90,
    )


class BatchPredictRequest(BaseModel):
    """Batch of city-day rows for vectorised inference."""

    rows: list[PredictRequest] = Field(..., min_length=1, max_length=9 * 7)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Semantic validation (impossible values -> HTTP 400, unlike schema 422)
# ---------------------------------------------------------------------------


def validate_semantics(observation: Observation) -> list[str]:
    """Return human-readable problems for values that are physically impossible."""
    problems: list[str] = []

    def check_range(name: str, value: float | None, low: float, high: float) -> None:
        if value is not None and not low <= value <= high:
            problems.append(f"{name}={value} is outside the allowed range [{low}, {high}]")

    check_range("aqi", observation.aqi, 0, 500)
    check_range("relative_humidity_2m_mean", observation.relative_humidity_2m_mean, 0, 100)
    check_range("wind_speed_10m_mean", observation.wind_speed_10m_mean, 0, 100)
    check_range("wind_direction_10m_dominant", observation.wind_direction_10m_dominant, 0, 360)
    check_range("precipitation_sum", observation.precipitation_sum, 0, 1000)
    check_range("fire_count_500km", observation.fire_count_500km, 0, 100_000)
    check_range("fire_frp_sum_500km", observation.fire_frp_sum_500km, 0, 10_000_000)
    check_range("fire_dist_min_km", observation.fire_dist_min_km, 0, 20_000)
    check_range("grap_stage", observation.grap_stage, 0, 4)
    check_range("festival_flag", observation.festival_flag, 0, 1)
    for pollutant in ("pm25", "pm10", "no2", "so2", "co", "o3"):
        check_range(pollutant, getattr(observation, pollutant), 0, 1_000_000)
    return problems


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PredictionInterval(BaseModel):
    lower: float = Field(..., description="10th-percentile AQI from residual quantiles.")
    median: float = Field(..., description="50th-percentile AQI from residual quantiles.")
    upper: float = Field(..., description="90th-percentile AQI from residual quantiles.")


class PredictResponse(BaseModel):
    city: str
    date: str = Field(..., description="Observation date echoed from the request.")
    target_date: str = Field(..., description="Date the prediction refers to (date + 1).")
    aqi_pred: float = Field(..., description="Predicted next-day AQI.")
    category: AQICategory = Field(..., description="Predicted next-day CPCB category.")
    category_probs: dict[str, float] = Field(
        ..., description="Approximate category probabilities (interval mass over CPCB bands)."
    )
    aqi_interval_80: PredictionInterval
    model_name: str
    model_version: str
    alias: str
    model_used: str
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float
    degraded: bool
    history_days: int


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    rows_processed: int


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str | None = None
    model_version: str | None = None
    features_count: int | None = None
    history_seed_days: int | None = None


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    features_count: int
    residual_mode: bool
    val_metrics: dict
    test_metrics: dict
    gate_status: dict
    split: dict
    git_commit: str
    created_at: str
    history_seed_days: int
    history_seed_rows: int
