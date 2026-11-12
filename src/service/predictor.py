"""
src/service/predictor.py
------------------------
Loads the champion AqiBundle at startup and exposes a simple inference API
used by the FastAPI route handlers.

Design principles
-----------------
* Single load at startup — the bundle is loaded once and held in a module-level
  singleton so that every request reuses the same in-memory object.
* Graceful degradation — if the bundle directory is missing the service still
  starts (ready=False); /health stays up, /predict returns 503.
* Thread-safe — joblib/sklearn/lightgbm prediction is GIL-releasing; pandas
  DataFrame construction is per-request, so there is no shared mutable state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.aqi_utils import aqi_to_category
from src.serving.bundle import AqiBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default bundle path (overridable via env var BUNDLE_DIR)
# ---------------------------------------------------------------------------
_DEFAULT_BUNDLE_DIR = Path("artifacts/champion")

# Module-level singleton
_bundle: AqiBundle | None = None
_bundle_dir: Path = _DEFAULT_BUNDLE_DIR


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def configure(bundle_dir: str | Path) -> None:
    """Override the bundle directory before load() is called."""
    global _bundle_dir
    _bundle_dir = Path(bundle_dir)


def load() -> None:
    """Load the champion bundle into memory. Called once at app startup."""
    global _bundle
    try:
        _bundle = AqiBundle.load(_bundle_dir)
        logger.info(
            "Champion bundle loaded: model=%s features=%d version=%s",
            _bundle.manifest["model"]["name"],
            _bundle.feature_count,
            _bundle.manifest.get("bundle_version", "?"),
        )
    except FileNotFoundError:
        logger.warning(
            "Bundle not found at %s — service will start in degraded mode (ready=False).",
            _bundle_dir,
        )
        _bundle = None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load bundle: %s", exc)
        _bundle = None


def is_ready() -> bool:
    return _bundle is not None


def get_bundle() -> AqiBundle:
    if _bundle is None:
        raise RuntimeError("Champion bundle is not loaded.")
    return _bundle


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _build_frame(row: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a validated PredictRequest dict into a single-row DataFrame whose
    columns include at least the bundle's required feature_names.

    Missing optional fields default to NaN; the bundle's scaler will impute
    them through the same pipeline used during training.
    """
    bundle = get_bundle()

    # Start with everything the caller provided
    record: dict[str, Any] = dict(row)

    # Fill required feature columns that weren't provided with NaN
    for col in bundle.feature_names:
        if col not in record:
            record[col] = np.nan

    frame = pd.DataFrame([record])

    # Keep only the columns the bundle knows about (plus city for quantiles)
    keep = list(bundle.feature_names) + (["city"] if "city" not in bundle.feature_names else [])
    frame = frame[[c for c in keep if c in frame.columns]]

    return frame


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict_one(row: dict[str, Any]) -> dict[str, Any]:
    """
    Run inference for a single city-day row.

    Returns a dict with keys: predicted_aqi, predicted_category, interval,
    model_name, fallback_used.
    """
    bundle = get_bundle()
    fallback_used = False

    try:
        frame = _build_frame(row)
        lower_arr, median_arr, upper_arr = bundle.predict_interval(frame)
        predicted_aqi = float(median_arr[0])
        lower = float(lower_arr[0])
        upper = float(upper_arr[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion prediction failed (%s), falling back to persistence.", exc)
        frame = _build_frame(row)
        preds = bundle.predict_fallback(frame)
        predicted_aqi = float(preds[0])
        lower = upper = predicted_aqi
        fallback_used = True

    predicted_category = aqi_to_category(predicted_aqi) or "Moderate"

    return {
        "predicted_aqi": round(predicted_aqi, 2),
        "predicted_category": predicted_category,
        "interval": {
            "lower": round(lower, 2),
            "median": round(predicted_aqi, 2),
            "upper": round(upper, 2),
        },
        "model_name": bundle.manifest["model"]["name"],
        "fallback_used": fallback_used,
    }


def predict_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Vectorised batch inference — builds one DataFrame, runs one predict call."""
    bundle = get_bundle()
    fallback_used = False

    frames = [_build_frame(r) for r in rows]
    frame = pd.concat(frames, ignore_index=True)

    try:
        lower_arr, median_arr, upper_arr = bundle.predict_interval(frame)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch champion prediction failed (%s), falling back.", exc)
        preds = bundle.predict_fallback(frame)
        median_arr = preds
        lower_arr = upper_arr = preds
        fallback_used = True

    results = []
    model_name = bundle.manifest["model"]["name"]
    for i, row in enumerate(rows):
        aqi_val = round(float(median_arr[i]), 2)
        results.append(
            {
                "city": row["city"],
                "date": row["date"],
                "predicted_aqi": aqi_val,
                "predicted_category": aqi_to_category(aqi_val) or "Moderate",
                "interval": {
                    "lower": round(float(lower_arr[i]), 2),
                    "median": aqi_val,
                    "upper": round(float(upper_arr[i]), 2),
                },
                "model_name": model_name,
                "fallback_used": fallback_used,
            }
        )
    return results


def get_metadata() -> dict[str, Any]:
    """Return a metadata snapshot from the loaded bundle's manifest."""
    bundle = get_bundle()
    m = bundle.manifest
    return {
        "model_name": m["model"]["name"],
        "bundle_version": m.get("bundle_version", "?"),
        "features_count": m.get("features_count", bundle.feature_count),
        "val_metrics": m["model"].get("val_metrics", {}),
        "test_metrics": m["model"].get("test_metrics", {}),
        "gate_status": m.get("gate_status", {}),
        "split": m.get("split", {}),
        "git_commit": m.get("git_commit", ""),
        "created_at": m.get("created_at", ""),
    }

