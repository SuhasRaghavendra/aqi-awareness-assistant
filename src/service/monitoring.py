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
from collections import deque
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
DEFAULT_PSI_WARN_THRESHOLD = 0.2
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
        self._feature_buffers: dict[str, deque[float]] = {
            f: deque(maxlen=psi_window) for f in MONITORED_FEATURES
        }
        # Stored predictions keyed by (city, target_date) -> predicted AQI
        self._predictions: dict[tuple[str, str], float] = {}
        # Rolling realized absolute errors per city (last 14 observations)
        self._abs_errors: dict[str, deque[float]] = {}

    def configure(
        self,
        reference_stats: dict[str, Any] | None = None,
        val_mae_baseline: float | None = None,
    ) -> None:
        """Load reference feature statistics and baseline validation MAE."""
        with self._lock:
            if reference_stats:
                self._ref_features = dict(reference_stats.get("features", {}))
            if val_mae_baseline is not None and val_mae_baseline > 0:
                self._val_mae = float(val_mae_baseline)

    def reset(self) -> None:
        """Clear rolling windows (used by tests)."""
        with self._lock:
            for buf in self._feature_buffers.values():
                buf.clear()
            self._predictions.clear()
            self._abs_errors.clear()

    def observe(
        self,
        payload: dict[str, Any],
        city: str,
        date: str,
        target_date: str,
        aqi_pred: float,
    ) -> tuple[dict[str, float], float | None, float | None]:
        """Record one inference request and return (psi_by_feature, abs_error, rolling_mae)."""
        psi_updates: dict[str, float] = {}
        realized_err: float | None = None
        rolling_mae: float | None = None

        with self._lock:
            # 1. Input drift (PSI) over rolling 500-request window
            for feat in MONITORED_FEATURES:
                raw_val = payload.get(feat)
                if raw_val is None:
                    continue
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                buf = self._feature_buffers[feat]
                buf.append(val)
                ref = self._ref_features.get(feat)
                if ref and len(buf) >= 10:
                    psi = compute_psi(buf, float(ref["mean"]), float(ref["std"]))
                    psi_updates[feat] = psi
                    if psi > self._psi_warn:
                        logger.warning(
                            "input_drift_detected",
                            extra={
                                "event": "input_drift",
                                "feature": feat,
                                "psi": psi,
                                "threshold": self._psi_warn,
                                "window_size": len(buf),
                            },
                        )

            # 2. Live realized accuracy: match today's observed AQI against prior prediction
            obs_aqi = payload.get("aqi")
            if obs_aqi is not None:
                key = (city, date)
                prior_pred = self._predictions.pop(key, None)
                if prior_pred is not None:
                    realized_err = round(abs(float(obs_aqi) - prior_pred), 4)
                    err_buf = self._abs_errors.setdefault(city, deque(maxlen=self._mae_window))
                    err_buf.append(realized_err)
                    rolling_mae = round(sum(err_buf) / len(err_buf), 4)
                    mae_limit = self._val_mae * DEFAULT_MAE_ALERT_MULTIPLIER
                    if len(err_buf) >= 3 and rolling_mae > mae_limit:
                        logger.warning(
                            "live_mae_degradation",
                            extra={
                                "event": "mae_alert",
                                "city": city,
                                "rolling_14d_mae": rolling_mae,
                                "alert_threshold": round(mae_limit, 3),
                                "observations": len(err_buf),
                            },
                        )

            # Store today's prediction for tomorrow's realized-error check
            if len(self._predictions) > 2000:
                oldest_key = next(iter(self._predictions))
                self._predictions.pop(oldest_key, None)
            self._predictions[(city, target_date)] = float(aqi_pred)

        return psi_updates, realized_err, rolling_mae


# Module-level singleton used by FastAPI routes
monitor = ServiceMonitor()
