"""Monotone constraints for the fire/wind candidates (ADR D-030).

The metamorphic gates require that increasing fire activity never *decreases*
the predicted AQI and that increasing wind never *increases* it.  Instead of
asking the gate for tolerance, the candidate encodes the direction
structurally using LightGBM ``monotone_constraints``.

``fire_dist_min_km`` is deliberately left unconstrained: within a 500 km
aggregate the nearest-fire distance is not a clean monotone signal.
"""

from __future__ import annotations

from collections.abc import Sequence

FIRE_PREFIXES = ("fire_count_500km", "fire_frp_sum_500km")
POSITIVE_FEATURES = frozenset(
    {
        "fire_burning_season",
        "inversion_x_fire",
        "fire_x_low_wind",
        # Stagnation rises when wind falls, so it must move *with* AQI.
        "winter_stagnation_index",
    }
)
NEGATIVE_FEATURES = frozenset(
    {
        "wind_speed_10m_mean",
        # The metamorphic shift changes the whole raw wind column, so its lags
        # and rolling aggregates change too; they carry the same physical sign.
        "wind_speed_10m_mean_lag_1",
        "wind_speed_10m_mean_lag_3",
        "wind_roll_mean_3d",
        "wind_roll_mean_7d",
    }
)


def monotone_direction(feature_name: str) -> int:
    """Return +1, -1 or 0 for a single feature."""
    if feature_name in NEGATIVE_FEATURES:
        return -1
    if feature_name in POSITIVE_FEATURES:
        return 1
    if feature_name.startswith(FIRE_PREFIXES):
        return 1
    return 0


def build_monotone_constraints(feature_names: Sequence[str]) -> list[int]:
    """Build the LightGBM constraint vector in feature order."""
    return [monotone_direction(name) for name in feature_names]


def constrained_features(feature_names: Sequence[str]) -> dict[str, int]:
    """Return only the non-zero constraints, for the model card and tests."""
    return {
        name: direction
        for name, direction in zip(
            feature_names, build_monotone_constraints(feature_names), strict=False
        )
        if direction
    }
