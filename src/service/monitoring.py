"""
src/service/monitoring.py
-------------------------
In-memory input drift (PSI) and live realized accuracy (rolling 14-day MAE)
monitor for the AQI Forecasting Inference Service (Plan Phase 6).

Container filesystems on PaaS hosts are ephemeral, so rolling windows live in
memory and alerts are emitted as structured JSON logs plus Prometheus gauges.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from datetime import date as calendar_date
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Standard-normal quintile cut-offs (20% expected mass per bin)
_QUINTILE_Z = (-0.84162123, -0.25334710, 0.25334710, 0.84162123)
_EXPECTED_SHARE = 0.2
_EPS = 1e-4

MONITORED_FEATURES: tuple[str, ...] = (
    "aqi",
    "temperature_2m_mean",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
    "fire_count_500km",
    "fire_frp_sum_500km",
)

DEFAULT_PSI_WINDOW = 500
DEFAULT_PSI_AMBER_THRESHOLD = 0.10
DEFAULT_PSI_WARN_THRESHOLD = 0.20
DEFAULT_MAE_WINDOW = 14
DEFAULT_MAE_ALERT_MULTIPLIER = 1.5
DEFAULT_VAL_MAE = 13.233


def compute_psi(values: list[float] | deque[float], mean: float, std: float) -> float:
    """Compute 5-bin Population Stability Index against a normal(mean, std) reference."""
    n = len(values)
    if n == 0 or std <= 0 or not math.isfinite(mean) or not math.isfinite(std):
        return 0.0

    edges = [mean + z * std for z in _QUINTILE_Z]
    counts = [0, 0, 0, 0, 0]
    for v in values:
        if v <= edges[0]:
            counts[0] += 1
        elif v <= edges[1]:
            counts[1] += 1
        elif v <= edges[2]:
            counts[2] += 1
        elif v <= edges[3]:
            counts[3] += 1
        else:
            counts[4] += 1

    psi = 0.0
    for count in counts:
        actual = max(count / n, _EPS)
        psi += (actual - _EXPECTED_SHARE) * math.log(actual / _EXPECTED_SHARE)
    return round(max(psi, 0.0), 6)


class ServiceMonitor:
    """Thread-safe rolling PSI drift and realized-error monitor."""

    def __init__(
        self,
        psi_window: int = DEFAULT_PSI_WINDOW,
        psi_warn_threshold: float = DEFAULT_PSI_WARN_THRESHOLD,
        mae_window: int = DEFAULT_MAE_WINDOW,
        val_mae_baseline: float = DEFAULT_VAL_MAE,
    ) -> None:
        self._lock = threading.Lock()
        self._psi_window = psi_window
        self._psi_warn = psi_warn_threshold
        self._mae_window = mae_window
        self._val_mae = val_mae_baseline
        self._ref_features: dict[str, dict[str, float]] = {}
        self._feature_buffers: dict[str, deque[float | None]] = {
            f: deque(maxlen=psi_window) for f in MONITORED_FEATURES
        }
        self._feature_drift: dict[str, dict[str, Any]] = {}
        self._ref_target: dict[str, float] = {}
        self._outputs: deque[float] = deque(maxlen=psi_window)
        self._prediction_psi: float | None = None
        self._total_predictions = 0
        self._out_of_range_predictions = 0
        self._out_of_range_fields: Counter[str] = Counter()
        self._predictions: dict[tuple[str, str], float] = {}
        self._abs_errors: dict[str, dict[calendar_date, float]] = {}
        self._latest_actual_date: calendar_date | None = None

    def configure(
        self,
        reference_stats: dict[str, Any] | None = None,
        val_mae_baseline: float | None = None,
    ) -> None:
        """Load reference feature statistics and baseline validation MAE."""
        with self._lock:
            if reference_stats is not None:
                self._ref_features = dict(reference_stats.get("features", {}))
                self._ref_target = dict(reference_stats.get("target_aqi", {}))
            if val_mae_baseline is not None and val_mae_baseline > 0:
                self._val_mae = float(val_mae_baseline)

    def reset(self) -> None:
        """Clear rolling windows (used by tests)."""
        with self._lock:
            for buf in self._feature_buffers.values():
                buf.clear()
            self._feature_drift.clear()
            self._outputs.clear()
            self._prediction_psi = None
            self._out_of_range_predictions = 0
            self._out_of_range_fields.clear()
            self._latest_actual_date = None
            self._total_predictions = 0
            self._predictions.clear()
            self._abs_errors.clear()

    def observe(
        self,
        payload: dict[str, Any],
        city: str,
        date: str,
        target_date: str,
        aqi_pred: float,
        *,
        out_of_range_fields: list[str] | tuple[str, ...] = (),
    ) -> tuple[dict[str, float], float | None, float | None]:
        """Record one inference request and return (psi_by_feature, abs_error, rolling_mae)."""
        psi_updates: dict[str, float] = {}
        realized_err: float | None = None
        rolling_mae: float | None = None

        with self._lock:
            self._total_predictions += 1
            if out_of_range_fields:
                self._out_of_range_predictions += 1
                self._out_of_range_fields.update(set(out_of_range_fields))

            self._outputs.append(float(aqi_pred))
            if self._ref_target and len(self._outputs) >= 10:
                self._prediction_psi = compute_psi(
                    self._outputs, self._ref_target["mean"], self._ref_target["std"]
                )
            for feature, buffer in self._feature_buffers.items():
                raw = payload.get(feature)
                try:
                    value = float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    value = None
                buffer.append(value if value is not None and math.isfinite(value) else None)
                values = [value for value in buffer if value is not None]
                ref = self._ref_features.get(feature)
                psi, z_score = None, None
                if ref and len(values) >= 10 and ref["std"] > 0:
                    psi = compute_psi(values, ref["mean"], ref["std"])
                    z_score = round((sum(values) / len(values) - ref["mean"]) / ref["std"], 6)
                    psi_updates[feature] = psi
                    severity = self._psi_status(psi)
                    if severity in ("AMBER", "RED"):
                        logger.warning(
                            "input_drift_detected",
                            extra={
                                "event": "input_drift",
                                "feature": feature,
                                "psi": psi,
                                "severity": severity,
                                "threshold": (
                                    self._psi_warn
                                    if severity == "RED"
                                    else DEFAULT_PSI_AMBER_THRESHOLD
                                ),
                                "window_size": len(values),
                            },
                        )
                self._feature_drift[feature] = {
                    "psi": psi,
                    "status": self._psi_status(psi),
                    "z_score": z_score,
                    "window_size": len(values),
                }

            obs_aqi = payload.get("aqi")
            if obs_aqi is not None:
                realized_err, rolling_mae = self._record_actual(city, date, float(obs_aqi))

            # Store today's prediction for tomorrow's realized-error check
            if len(self._predictions) >= 2000:
                oldest_key = next(iter(self._predictions))
                self._predictions.pop(oldest_key, None)
            if calendar_date.fromisoformat(target_date) not in self._abs_errors.get(city, {}):
                self._predictions[(city, target_date)] = float(aqi_pred)

        return psi_updates, realized_err, rolling_mae

    def observe_actual(
        self, city: str, date: str, actual_aqi: float
    ) -> tuple[float | None, float | None]:
        """Pair an independently supplied actual with a pending forecast."""
        with self._lock:
            return self._record_actual(city, date, actual_aqi)

    def _record_actual(
        self, city: str, date: str, actual_aqi: float
    ) -> tuple[float | None, float | None]:
        prior_pred = self._predictions.pop((city, date), None)
        if prior_pred is None:
            return None, None
        error = round(abs(actual_aqi - prior_pred), 4)
        observed = calendar_date.fromisoformat(date)
        self._latest_actual_date = max(self._latest_actual_date or observed, observed)
        cutoff = calendar_date.fromordinal(
            max(1, self._latest_actual_date.toordinal() - self._mae_window + 1)
        )
        if observed >= cutoff:
            self._abs_errors.setdefault(city, {}).setdefault(observed, error)
        for name, city_errors in list(self._abs_errors.items()):
            self._abs_errors[name] = {day: err for day, err in city_errors.items() if day >= cutoff}
        errors = self._abs_errors.get(city, {})
        rolling_mae = round(sum(errors.values()) / len(errors), 4) if errors else None
        if len(errors) >= 3 and rolling_mae > self.mae_threshold:
            logger.warning(
                "live_mae_degradation",
                extra={
                    "event": "mae_alert",
                    "city": city,
                    "realized_abs_error": error,
                    "rolling_14d_mae": rolling_mae,
                    "alert_threshold": self.mae_threshold,
                    "observations": len(errors),
                },
            )
        return error, rolling_mae

    @property
    def mae_threshold(self) -> float:
        return round(self._val_mae * DEFAULT_MAE_ALERT_MULTIPLIER, 2)

    def _psi_status(self, psi: float | None) -> str | None:
        if psi is None:
            return None
        if psi > self._psi_warn:
            return "RED"
        return "AMBER" if psi >= DEFAULT_PSI_AMBER_THRESHOLD else "GREEN"

    def prediction_psi(self) -> float | None:
        with self._lock:
            return self._prediction_psi

    def summary(self) -> dict[str, Any]:
        """Return the per-worker view of drift, realized accuracy, and counts."""
        with self._lock:
            features = {
                name: self._feature_drift.get(
                    name, {"psi": None, "status": None, "z_score": None, "window_size": 0}
                ).copy()
                for name in MONITORED_FEATURES
            }
            cities = {
                city: {
                    "mae": round(sum(errors.values()) / len(errors), 4),
                    "observations": len(errors),
                }
                for city, errors in self._abs_errors.items()
                if errors
            }
            breached_features = [name for name, row in features.items() if row["status"] == "RED"]
            breached_cities = [
                city
                for city, row in cities.items()
                if row["observations"] >= 3 and row["mae"] > self.mae_threshold
            ]
            prediction_status = self._psi_status(self._prediction_psi)
            alerts = [
                {"type": "input_drift", "feature": name, "severity": row["status"]}
                for name, row in features.items()
                if row["status"] in ("AMBER", "RED")
            ]
            alerts.extend(
                {"type": "mae_alert", "city": city, "severity": "RED"} for city in breached_cities
            )
            if prediction_status in ("AMBER", "RED"):
                alerts.append({"type": "prediction_drift", "severity": prediction_status})
            errors = [
                error for city_errors in self._abs_errors.values() for error in city_errors.values()
            ]
            global_mae = round(sum(errors) / len(errors), 4) if errors else None
            if len(errors) >= 3 and global_mae > self.mae_threshold:
                alerts.append({"type": "mae_alert", "scope": "global", "severity": "RED"})
            return {
                "total_predictions": self._total_predictions,
                "rolling_14d_mae": global_mae,
                "mae_alert_threshold": self.mae_threshold,
                "val_mae_baseline": self._val_mae,
                "max_feature_psi": max(
                    (row["psi"] for row in features.values() if row["psi"] is not None),
                    default=None,
                ),
                "psi_warning_threshold": DEFAULT_PSI_AMBER_THRESHOLD,
                "psi_alert_threshold": self._psi_warn,
                "feature_drift": features,
                "prediction_drift": {
                    "psi": self._prediction_psi,
                    "status": prediction_status,
                    "window_size": len(self._outputs),
                },
                "per_city_mae": cities,
                "out_of_range_predictions": self._out_of_range_predictions,
                "out_of_range_counts": dict(self._out_of_range_fields),
                "breached_features": breached_features,
                "breached_cities": breached_cities,
                "active_alerts": alerts,
                "retrain_recommended": any(alert["severity"] == "RED" for alert in alerts),
            }


monitor = ServiceMonitor()
