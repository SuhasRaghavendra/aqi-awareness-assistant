"""
src/features/build_features.py
-------------------------------
Phase 4 – Feature Engineering & Chronological Split.

Constructs leakage-free root-cause features from the cleaned city × day panel.
All features use information available at day t to predict day t+1 AQI.

Feature groups:
  [A] Fire/Burning lags & rolling windows (fire_count, frp)
  [B] Meteorological drivers lags & derived indices (wind, humidity, temp, precip)
  [C] Vehicle emission proxy (annual -> interpolated; fractions)
  [D] Industrial activity (IIP monthly index, power consumption)
  [E] Policy/Governance (GRAP stage, BS-VI, odd-even)
  [F] AQI history lags & rolling means (as context; NOT same-day AQI as feature)
  [G] Calendar & seasonal features
  [H] City one-hot encoding + static geographic features
  [I] Interaction features (e.g. fire × wind_speed, burning_season × fire_count)

Outputs:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
  data/processed/feature_list.json
  models/scaler.joblib
  reports/metrics/features_stats.json
  reports/figures/features/feature_correlations.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_params
from src.features.aqi_utils import aqi_to_category
from src.features.serving_features import (
    add_aqi_history_features,
    add_city_onehot,
    add_fire_features,
    add_grap_features,
    add_interaction_features,
    add_season_onehot,
    add_weather_lag_features,
)
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("build_features")

# ── Columns that are never ML features ───────────────────────────────────────
# NOTE: 'aqi' (day-t observation) IS now included as a feature (ADR D-019).
# aqi_lag_1 and aqi_diff_1d cannot reconstruct aqi because of the reindex bug:
# after grp.reindex(full_date_range), the index is DatetimeIndex but grp_idx
# holds the original integer positions, so .reindex(grp_idx) returns all-NaN
# which the median-fill then sets to 0.  Adding raw aqi sidesteps this entirely.
NON_FEATURE_COLS = {
    "city",
    "date",
    "aqi_category",
    "aqi_median",
    "station_count",  # raw AQI source cols
    "year",  # encoded via calendar
}


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create next-day AQI target per city.
    Shift within a continuous calendar (gap-safe: use date index per city).
    """
    df = df.sort_values(["city", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    targets = []
    for city, grp in df.groupby("city"):
        grp = grp.set_index("date").sort_index()
        # Reindex to full calendar to avoid shifting across date gaps
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)
        grp["target_aqi"] = grp["aqi"].shift(-1)
        grp["target_category"] = grp["target_aqi"].apply(
            lambda v: aqi_to_category(v) if pd.notna(v) else None
        )
        grp["city"] = city
        grp.index.name = "date"
        grp = grp.reset_index()
        targets.append(grp)

    return pd.concat(targets, ignore_index=True)


def split_chronological(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test by date — no data leakage."""
    df["date"] = pd.to_datetime(df["date"])
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    df_train = df[df["date"] <= train_end_dt].copy()
    df_val = df[(df["date"] > train_end_dt) & (df["date"] <= val_end_dt)].copy()
    df_test = df[df["date"] > val_end_dt].copy()

    for part in [df_train, df_val, df_test]:
        part["date"] = part["date"].dt.strftime("%Y-%m-%d")

    return df_train, df_val, df_test


def main() -> None:
    setup_logging(stage="features")
    parser = argparse.ArgumentParser(description="Phase 4: Build root-cause features")
    parser.add_argument(
        "--input-file", type=Path, default=Path("data/interim/city_day_clean.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]
    feat_cfg = params.get("features", {})
    split_cfg = params.get("split", {})

    aqi_lags: list[int] = feat_cfg.get("aqi_lags", [1, 3, 7, 14])
    fire_lags: list[int] = feat_cfg.get("fire_lags", [1, 3, 7])
    rolling_w: list[int] = feat_cfg.get("rolling_windows", [7, 14, 30])
    train_end: str = split_cfg.get("train_end", "2022-12-31")
    val_end: str = split_cfg.get("val_end", "2025-12-31")

    df = pd.read_parquet(args.input_file)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    logger.info(f"Loaded clean panel: {len(df):,} rows, {len(df.columns)} cols")

    # ── Build features ────────────────────────────────────────────────────────
    logger.info("Creating next-day targets …")
    df = make_target(df)

    logger.info("Adding fire lag/rolling features …")
    df = add_fire_features(df, fire_lags, rolling_w)

    logger.info("Adding AQI history features …")
    df = add_aqi_history_features(df, aqi_lags, rolling_w)

    logger.info("Adding weather lag features …")
    df = add_weather_lag_features(df, aqi_lags[:2])

    logger.info("Adding interaction features …")
    df = add_interaction_features(df)

    logger.info("Adding city one-hot …")
    df = add_city_onehot(df, cities)

    logger.info("Adding season one-hot …")
    df = add_season_onehot(df)

    logger.info("Adding GRAP binary features …")
    df = add_grap_features(df)

    # Drop rows with no target (last row per city)
    df = df.dropna(subset=["target_aqi"]).reset_index(drop=True)
    logger.info(f"After target creation: {len(df):,} rows")

    # ── Split ─────────────────────────────────────────────────────────────────
    df_train, df_val, df_test = split_chronological(df, train_end, val_end)
    logger.info(f"Split: train={len(df_train):,} | val={len(df_val):,} | test={len(df_test):,}")

    # ── Feature list ──────────────────────────────────────────────────────────
    # Identify feature columns: numeric, not in NON_FEATURE_COLS
    all_non_feat = NON_FEATURE_COLS | {
        "target_aqi",
        "target_category",
        "season",  # replaced by season_* one-hots
    }
    feature_cols = [
        c for c in df.columns if c not in all_non_feat and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info(f"Total engineered features: {len(feature_cols)}")

    # Fill remaining NaN with train medians
    train_medians = df_train[feature_cols].median().fillna(0.0).to_dict()
    for part in [df_train, df_val, df_test]:
        part[feature_cols] = part[feature_cols].fillna(train_medians)

    # ── Scale ─────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    scaler_path = args.output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Fitted StandardScaler on train split -> {scaler_path}")

    # ── Save splits ───────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(args.output_dir / "train.parquet", index=False)
    df_val.to_parquet(args.output_dir / "val.parquet", index=False)
    df_test.to_parquet(args.output_dir / "test.parquet", index=False)

    feat_list: dict[str, Any] = {
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "target_regression": "target_aqi",
        "target_classification": "target_category",
        "feature_groups": {
            "fire_burning": [c for c in feature_cols if "fire" in c or "burning" in c],
            "meteorological": [
                c
                for c in feature_cols
                if any(
                    k in c
                    for k in [
                        "temperature",
                        "wind",
                        "humidity",
                        "precip",
                        "pressure",
                        "inversion",
                        "fog",
                        "rain_washout",
                    ]
                )
            ],
            "vehicle_emissions": [
                c
                for c in feature_cols
                if any(
                    k in c
                    for k in [
                        "diesel",
                        "cng",
                        "ev_",
                        "two_wheel",
                        "vehicles_per",
                        "bsvi",
                        "odd_even",
                    ]
                )
            ],
            "industrial": [
                c for c in feature_cols if any(k in c for k in ["iip", "energy_", "stagnation"])
            ],
            "policy_governance": [
                c for c in feature_cols if any(k in c for k in ["grap", "bsvi", "odd_even"])
            ],
            "aqi_history": [c for c in feature_cols if c.startswith("aqi_")],
            "calendar_seasonal": [
                c
                for c in feature_cols
                if any(
                    k in c
                    for k in [
                        "month",
                        "season_",
                        "day_of",
                        "weekend",
                        "festival",
                        "dow_",
                        "month_sin",
                        "month_cos",
                    ]
                )
            ],
            "spatial": [
                c
                for c in feature_cols
                if any(k in c for k in ["city_", "elevation", "population", "lat", "lon"])
            ],
            "interaction": [
                c
                for c in feature_cols
                if any(k in c for k in ["_x_", "intensity", "overlap", "stagnation_index"])
            ],
        },
        "generated_at": datetime.now().isoformat(),
    }
    with open(args.output_dir / "feature_list.json", "w") as f:
        json.dump(feat_list, f, indent=2)

    # ── Correlation plot ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt  # training-only dependency (not in the serve image)

    corr = df_train[feature_cols].apply(lambda s: s.corr(df_train["target_aqi"])).dropna()
    top15 = corr.abs().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    top15.sort_values().plot(kind="barh", ax=ax, color="#1a6faf")
    ax.set_title(
        "Top 15 Root-Cause Features by |Correlation| with Next-Day AQI (Train Split)",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Absolute Pearson |r|")
    fig.tight_layout()
    fig_dir = Path("reports/figures/features")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "feature_correlations.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature correlation figure")

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "target_aqi_train": {
            "min": float(df_train["target_aqi"].min()),
            "max": float(df_train["target_aqi"].max()),
            "mean": float(df_train["target_aqi"].mean()),
        },
        "feature_group_sizes": {k: len(v) for k, v in feat_list["feature_groups"].items()},
    }
    out_path = Path("reports/metrics/features_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    with track_stage(
        "features",
        inputs=[str(args.input_file)],
        outputs=[
            str(args.output_dir / "train.parquet"),
            str(args.output_dir / "val.parquet"),
            str(args.output_dir / "test.parquet"),
        ],
    ) as st:
        st.record_input_frame(df)
        st.record_output_frame(df_train)

    print("\n" + "=" * 65)
    print("ROOT-CAUSE FEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"Total Features:     {len(feature_cols)}")
    for grp, cols in feat_list["feature_groups"].items():
        print(f"  {grp:<22} {len(cols):>3} features")
    print(f"\nTrain: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")
    print(
        f"Target AQI range (train): {stats['target_aqi_train']['min']:.0f}–{stats['target_aqi_train']['max']:.0f}"
    )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
