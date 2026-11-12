"""Gate 3: uniform per-city calibration (ADR D-031)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    CityCalibratedRegressor,
    forward_calibrated_predictions,
    per_city_offsets,
)


class _OffsetModel:
    """Tiny regressor whose prediction equals ``aqi`` plus a fixed bias."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def fit(self, X, y, aqi=None):  # noqa: ANN001, ANN201 - sklearn contract
        return self

    def predict(self, X, aqi=None):  # noqa: ANN001, ANN201 - sklearn contract
        base = np.asarray(aqi if aqi is not None else X["aqi"], dtype=float)
        return base + self.bias


def test_per_city_offsets_is_the_median_residual_per_city() -> None:
    y_true = np.array([10.0, 12.0, 20.0, 70.0, 72.0, 66.0])
    y_pred = np.array([8.0, 8.0, 8.0, 80.0, 80.0, 80.0])
    cities = np.array(["Delhi", "Delhi", "Delhi", "Mumbai", "Mumbai", "Mumbai"])

    offsets = per_city_offsets(y_true, y_pred, cities)

    assert offsets == pytest.approx({"Delhi": 4.0, "Mumbai": -10.0})


def test_forward_calibration_uses_only_strictly_earlier_rows() -> None:
    """Changing late outcomes must not move earlier calibrated predictions."""
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    cities = np.array(["Delhi", "Mumbai"] * (n // 2))
    raw = rng.normal(100.0, 5.0, size=n)
    truth = raw + rng.normal(0.0, 3.0, size=n)

    baseline, records = forward_calibrated_predictions(truth, raw, cities, dates, n_splits=4)

    corrupted = truth.copy()
    corrupted[-10:] += 500.0
    after, _ = forward_calibrated_predictions(corrupted, raw, cities, dates, n_splits=4)

    # Only the final fold may see the corrupted outcomes, and even it only
    # through the offsets estimated from earlier folds inside that block.
    assert np.allclose(baseline[:-30], after[:-30])
    assert len(records) == 4
    # The earliest block has no history to calibrate from, so it is not covered.
    expected_applied = n - n // (len(records) + 1)
    assert sum(record["apply_rows"] for record in records) == expected_applied


def test_calibrated_wrapper_applies_the_offset_and_defaults_to_zero() -> None:
    frame = pd.DataFrame({"aqi": [50.0, 60.0, 70.0]})
    model = CityCalibratedRegressor(
        base_model=_OffsetModel(bias=2.0),
        offsets={"Delhi": -5.0, "Mumbai": 3.0},
    )

    preds = model.predict(
        frame, frame["aqi"].to_numpy(), city=np.array(["Delhi", "Mumbai", "Pune"])
    )

    assert preds == pytest.approx([47.0, 65.0, 72.0])


def test_calibration_report_is_serialisable_and_complete() -> None:
    model = CityCalibratedRegressor(base_model=_OffsetModel(), offsets={"Delhi": -1.5, "Pune": 2.5})
    report = model.calibration_report()

    assert report["method"] == "per_city_median_residual"
    assert report["fitted_on"] == "validation"
    assert report["n_cities"] == 2
    assert report["offsets"] == {"Delhi": -1.5, "Pune": 2.5}
    assert report["offset_abs_max"] == 2.5
    assert report["offset_mean"] == pytest.approx(0.5)
