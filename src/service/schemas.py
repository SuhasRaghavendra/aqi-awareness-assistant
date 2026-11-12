"""
src/service/schemas.py
----------------------
Pydantic v2 request / response models for the AQI FastAPI service.

All field names match the feature names in feature_list.json so the
predictor can pass the validated dict straight to the bundle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CPCB category literal
# ---------------------------------------------------------------------------
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
# Single-row prediction request
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Raw feature payload for one city-day prediction."""

    # --- Identity ---
    city: CITIES = Field(..., description="One of the 9 supported Indian cities.")
    date: str = Field(
        ...,
        description="Date string ISO-8601 (YYYY-MM-DD). Used for calendar features.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    # --- Current-day pollutants (µg/m³, CO in mg/m³) ---
    aqi: float = Field(..., ge=0, le=500, description="Current-day CPCB AQI.")
    pm25: float | None = Field(None, ge=0)
    pm10: float | None = Field(None, ge=0)
    no2: float | None = Field(None, ge=0)
    so2: float | None = Field(None, ge=0)
    co: float | None = Field(None, ge=0)
    o3: float | None = Field(None, ge=0)

    # --- Weather (ERA5 / Open-Meteo) ---
    temperature_2m_mean: float | None = Field(None, description="°C")
    temperature_2m_min: float | None = Field(None, description="°C")
    temperature_2m_max: float | None = Field(None, description="°C")
    relative_humidity_2m_mean: float | None = Field(None, ge=0, le=100)
    wind_speed_10m_mean: float | None = Field(None, ge=0)
    wind_direction_10m_dominant: float | None = Field(None, ge=0, le=360)
    precipitation_sum: float | None = Field(None, ge=0)
    surface_pressure_mean: float | None = Field(None)

    # --- Fire events (NASA FIRMS) ---
    fire_count_500km: float | None = Field(None, ge=0)
    fire_frp_sum_500km: float | None = Field(None, ge=0)

    # --- Lag features (optional — if omitted, service uses history_seed) ---
    aqi_lag_1: float | None = Field(None, ge=0)
    aqi_lag_3: float | None = Field(None, ge=0)
    aqi_lag_7: float | None = Field(None, ge=0)
    aqi_lag_14: float | None = Field(None, ge=0)

    model_config = {"extra": "allow"}  # forward-compat: ignore unknown fields


# ---------------------------------------------------------------------------
# Batch request
# ---------------------------------------------------------------------------
class BatchPredictRequest(BaseModel):
    """Batch of city-day rows for vectorised inference."""

    rows: list[PredictRequest] = Field(..., min_length=1, max_length=9 * 7)


# ---------------------------------------------------------------------------
# Prediction interval
# ---------------------------------------------------------------------------
class PredictionInterval(BaseModel):
    lower: float = Field(..., description="10th-percentile AQI (residual quantile).")
    median: float = Field(..., description="50th-percentile AQI.")
    upper: float = Field(..., description="90th-percentile AQI.")


# ---------------------------------------------------------------------------
# Single prediction response
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    city: str
    date: str
    predicted_aqi: float = Field(..., description="Next-day AQI (continuous).")
    predicted_category: AQICategory = Field(..., description="Next-day CPCB category.")
    interval: PredictionInterval
    model_name: str
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Batch response
# ---------------------------------------------------------------------------
class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    rows_processed: int


# ---------------------------------------------------------------------------
# Health / ready / metadata responses
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str | None = None
    features_count: int | None = None
    bundle_version: str | None = None


class MetadataResponse(BaseModel):
    model_name: str
    bundle_version: str
    features_count: int
    val_metrics: dict
    test_metrics: dict
    gate_status: dict
    split: dict
    git_commit: str
    created_at: str

