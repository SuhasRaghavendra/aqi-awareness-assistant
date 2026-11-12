"""
src/service/predictor.py
------------------------
Loads the champion AqiBundle at startup and exposes the inference API used by
the FastAPI route handlers.

The service rebuilds the full 103-feature panel from raw observations plus the
bundle's 45-day history seed.  Feature construction lives in
``src.features.serving_features`` and is shared with offline training, so the
parity test can hold the service to the offline pipeline within 1e-6.
"""

from __future__ import annotations

import logging
import time
from datetime import date as date_type
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.aqi_utils import aqi_to_category
from src.features.serving_features import (
    CATEGORY_ORDER,
    OBSERVATION_COLUMNS,
    STATIC_COLUMNS,
    build_feature_record,
    build_grap_lookup,
    build_observation_record,
    category_probabilities,
    load_serving_configs,
    residual_sigma,
)
from src.service.history import HistoryStore
from src.serving.bundle import AqiBundle

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUNDLE_DIR = Path("artifacts/champion")
_DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"

# Module-level singleton state
_bundle: AqiBundle | None = None
_bundle_dir: Path = _DEFAULT_BUNDLE_DIR
_config_dir: Path = _DEFAULT_CONFIG_DIR
_params: dict[str, Any] = {}
_train_medians: dict[str, float] = {}
_grap_calendar: pd.DataFrame = pd.DataFrame()
_grap_lookup: dict[str, tuple[int, int]] = {}
_festival_days: set[str] = set()
_history = HistoryStore()


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def configure(bundle_dir: str | Path | None = None, config_dir: str | Path | None = None) -> None:
    """Override the bundle/config directories before ``load()`` is called."""
    global _bundle_dir, _config_dir
    if bundle_dir is not None:
        _bundle_dir = Path(bundle_dir)
    if config_dir is not None:
        _config_dir = Path(config_dir)


def load() -> None:
    """Load the champion bundle, serving configs and the history seed."""
    global _bundle, _params, _train_medians, _grap_calendar, _grap_lookup, _festival_days
    try:
        _bundle = AqiBundle.load(_bundle_dir)
        _params = dict(_bundle.feature_config or {})
        _train_medians = dict(_bundle.reference_stats.get("train_medians", {}))
        _grap_calendar, _festival_days = load_serving_configs(_config_dir)
        _grap_lookup = build_grap_lookup(_grap_calendar)
        _history.reset(_bundle.history_seed)
        logger.info(
            "Champion bundle loaded: model=%s features=%d version=%s history_days=%d",
            _bundle.manifest["model"]["name"],
            _bundle.feature_count,
            _bundle.manifest.get("bundle_version", "?"),
            _bundle.manifest.get("history_seed_days", 0),
        )
    except FileNotFoundError:
        logger.warning(
            "Bundle not found at %s - service will start in degraded mode (ready=False).",
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


def load_history(frame: pd.DataFrame | None) -> None:
    """Replace the rolling history (used by tests and stateless deployments)."""
    _history.reset(frame)


def reset_history() -> None:
    """Reset the rolling history to the bundle's seed."""
    _history.reset(_bundle.history_seed if _bundle is not None else None)


def history_snapshot() -> dict[str, int]:
    return _history.snapshot()


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    if isinstance(value, str) and len(value) >= 10:
        return value[:10]
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and np.isnan(value)


def _observation_record(row: dict[str, Any], city: str, date: str) -> dict[str, Any]:
    record = {
        column: row[column]
        for column in OBSERVATION_COLUMNS
        if column in row and row[column] is not None and not _is_nan(row[column])
    }
    record["city"] = city
    record["date"] = date
    return record


def _static_reference(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    last = frame.iloc[-1]
    return {
        column: float(last[column])
        for column in STATIC_COLUMNS
        if column in frame.columns and pd.notna(last[column])
    }


def _seed_static_reference(city: str, date: str) -> dict[str, Any]:
    frame = _history.history_before(city, date)
    if frame.empty:
        frame = _history.history_before(city, "9999-12-31")
    return _static_reference(frame)


def _last_static_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the most recent record carrying the slow-moving reference fields."""
    for record in reversed(records):
        reference = {
            column: record[column]
            for column in STATIC_COLUMNS
            if column in record and pd.notna(record[column])
        }
        if reference:
            return reference
    return {}


def _build_history_frame(records: list[dict[str, Any]], city: str) -> pd.DataFrame:
    """Normalise a stateless ``history`` payload into an engineered panel."""
    if not records:
        return pd.DataFrame()
    normalised = []
    for record in records:
        entry = dict(record)
        entry["city"] = entry.get("city") or city
        normalised.append(entry)

    static_reference = _last_static_reference(normalised)
    rows = [
        build_observation_record(
            record,
            grap_lookup=_grap_lookup,
            festival_days=_festival_days,
            static_reference=static_reference,
            train_medians=_train_medians,
        )
        for record in normalised
    ]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values("date").reset_index(drop=True)


def _resolve_context(
    row: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the pre-request history window and derived metadata for a row."""
    city = str(row["city"])
    date = _iso_date(row["date"])
    override = history if history is not None else row.get("history")

    if override is not None:
        records = [dict(record) for record in override]
        records = [record for record in records if _iso_date(record["date"]) < date]
        history_frame = _build_history_frame(records, city)
        if not history_frame.empty:
            history_frame = history_frame[history_frame["city"] == city]
    else:
        history_frame = _history.history_before(city, date)

    history_days = int(history_frame["date"].nunique()) if not history_frame.empty else 0
    last_history_date = str(history_frame["date"].max()) if not history_frame.empty else None
    static_reference = _static_reference(history_frame) or _seed_static_reference(city, date)
    return {
        "city": city,
        "date": date,
        "history": history_frame,
        "history_days": history_days,
        "last_history_date": last_history_date,
        "static_reference": static_reference,
        "used_store": override is None,
    }


def _build_observation(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    record = _observation_record(row, context["city"], context["date"])
    return build_observation_record(
        record,
        grap_lookup=_grap_lookup,
        festival_days=_festival_days,
        static_reference=context["static_reference"],
        train_medians=_train_medians,
    )


def _build_features(observation_row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return build_feature_record(
        context["history"],
        observation_row,
        get_bundle().feature_names,
        _params,
        _train_medians,
    )


def _prepare(row: dict[str, Any], *, update_history: bool = False) -> dict[str, Any]:
    """Build everything needed to score one request row."""
    started = time.perf_counter()
    context = _resolve_context(row)
    observation_row = _build_observation(row, context)
    record = _build_features(observation_row, context)
    if update_history and context["used_store"]:
        _history.append(observation_row)
    return {
        "row": row,
        "context": context,
        "observation_row": observation_row,
        "record": record,
        "started": started,
    }


def prepare_features(
    row: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], int, str | None, bool]:
    """Build the feature row for one request (public, side-effect free).

    Returns ``(feature_row, observation_row, history_days, last_history_date,
    used_store_history)``.  When a request supplies its own history the service
    does not mutate the rolling store (stateless mode).
    """
    context = _resolve_context(row, history)
    observation_row = _build_observation(row, context)
    record = _build_features(observation_row, context)
    return (
        pd.DataFrame([record]),
        observation_row,
        context["history_days"],
        context["last_history_date"],
        context["used_store"],
    )


def _reference_warnings(
    row: dict[str, Any],
    history_days: int,
    last_history_date: str | None,
    date: str,
    degraded_floor: int = 30,
) -> tuple[list[str], list[str]]:
    """Flag out-of-range inputs, short history and calendar gaps."""
    warnings: list[str] = []
    fields: list[str] = []
    reference = _bundle.reference_stats if _bundle is not None else {}
    feature_stats = reference.get("features", {})
    tolerance = float(reference.get("tolerance", {}).get("relative", 0.0))

    for column in OBSERVATION_COLUMNS:
        value = row.get(column)
        if value is None or _is_nan(value) or column not in feature_stats:
            continue
        stats = feature_stats[column]
        low, high = float(stats["min"]), float(stats["max"])
        span = max(high - low, 1e-9)
        margin = tolerance * span
        if value < low - margin or value > high + margin:
            warnings.append(
                f"{column}={value} is outside the training range [{low:.3f}, {high:.3f}]"
            )
            fields.append(column)

    if history_days < degraded_floor:
        warnings.append(
            f"history has {history_days} day(s) (< {degraded_floor}); lag features may be degraded"
        )

    if last_history_date is not None:
        gap = (
            datetime.fromisoformat(date) - datetime.fromisoformat(str(last_history_date)[:10])
        ).days
        if gap > 1:
            warnings.append(
                f"history ends {gap - 1} day(s) before the requested date; "
                "gaps are median-filled"
            )

    return warnings[:20], fields


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------


def _finish(
    prepared: list[dict[str, Any]],
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Assemble response payloads from prepared rows and model predictions."""
    bundle = get_bundle()
    lower, median, upper = predictions if predictions is not None else ([], [], [])
    results: list[dict[str, Any]] = []

    for index, item in enumerate(prepared):
        row = item["row"]
        context = item["context"]
        observation_row = item["observation_row"]
        city = context["city"]
        date = context["date"]
        target_date = (datetime.fromisoformat(date) + timedelta(days=1)).strftime("%Y-%m-%d")

        warnings, out_of_range_fields = _reference_warnings(
            row, context["history_days"], context["last_history_date"], date
        )
        fallback_used = False
        model_used = bundle.manifest["model"]["name"]

        if predictions is not None:
            predicted_aqi = float(median[index])
            lower_bound = float(lower[index])
            upper_bound = float(upper[index])
            quantiles = bundle.residual_quantiles.get(city, bundle.residual_quantiles["__global__"])
            probabilities = category_probabilities(predicted_aqi, residual_sigma(quantiles))
        else:
            predicted_aqi = float(observation_row["aqi"])
            lower_bound = upper_bound = predicted_aqi
            fallback_used = True
            model_used = "fallback_persistence"
            current_category = aqi_to_category(predicted_aqi) or "Moderate"
            probabilities = {
                label: (1.0 if label == current_category else 0.0) for label in CATEGORY_ORDER
            }
            warnings.append(f"champion prediction failed; persistence used ({error})")

        category = aqi_to_category(predicted_aqi) or "Moderate"

        gap_days = 0
        if context["last_history_date"] is not None:
            gap_days = (
                datetime.fromisoformat(date)
                - datetime.fromisoformat(str(context["last_history_date"])[:10])
            ).days
        degraded = fallback_used or context["history_days"] < 30 or gap_days > 1
        latency_ms = (time.perf_counter() - item["started"]) * 1000.0

        results.append(
            {
                "city": city,
                "date": date,
                "target_date": target_date,
                "aqi_pred": predicted_aqi,
                "category": category,
                "category_probs": probabilities,
                "aqi_interval_80": {
                    "lower": lower_bound,
                    "median": predicted_aqi,
                    "upper": upper_bound,
                },
                "model_name": bundle.manifest["model"]["name"],
                "model_version": str(bundle.manifest.get("bundle_version", "?")),
                "alias": "champion",
                "model_used": model_used,
                "fallback_used": fallback_used,
                "warnings": warnings[:20],
                "latency_ms": round(latency_ms, 3),
                "degraded": degraded,
                "history_days": context["history_days"],
                "out_of_range_fields": out_of_range_fields,
            }
        )
    return results


def predict_one(row: dict[str, Any]) -> dict[str, Any]:
    """Run inference for a single city-day and update the rolling history."""
    prepared = [_prepare(row, update_history=True)]
    try:
        frame = pd.DataFrame([prepared[0]["record"]])
        predictions = get_bundle().predict_interval(frame)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion prediction failed (%s), falling back to persistence.", exc)
        predictions = None
        error = exc
    return _finish(prepared, predictions, error)[0]


def predict_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run inference for a batch, scoring all feature rows in one model call."""
    if not rows:
        return []
    prepared = [_prepare(row, update_history=True) for row in rows]
    try:
        frame = pd.DataFrame([item["record"] for item in prepared])
        predictions = get_bundle().predict_interval(frame)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch prediction failed (%s), falling back to persistence.", exc)
        predictions = None
        error = exc
    return _finish(prepared, predictions, error)


def get_metadata() -> dict[str, Any]:
    """Return a metadata snapshot from the loaded bundle's manifest."""
    bundle = get_bundle()
    manifest = bundle.manifest
    return {
        "model_name": manifest["model"]["name"],
        "model_version": str(manifest.get("bundle_version", "?")),
        "alias": "champion",
        "features_count": manifest.get("features_count", bundle.feature_count),
        "residual_mode": bool(bundle.residual_mode),
        "val_metrics": manifest["model"].get("val_metrics", {}),
        "test_metrics": manifest["model"].get("test_metrics", {}),
        "gate_status": manifest.get("gate_status", {}),
        "split": manifest.get("split", {}),
        "git_commit": manifest.get("git_commit", ""),
        "created_at": manifest.get("created_at", ""),
        "history_seed_days": int(manifest.get("history_seed_days", 0)),
        "history_seed_rows": int(manifest.get("history_seed_rows", 0)),
    }
