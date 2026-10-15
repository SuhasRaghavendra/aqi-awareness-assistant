"""
src/models/ablations.py
-----------------------
Phase 7: Feature Ablation Studies (Section 11, item 4 of Implementation Plan).
Evaluates the impact and incremental value of:
  (a) Lag & Rolling features (temporal memory)
  (b) Weather features (Open-Meteo ERA5 enrichment)
  (c) Spatial context features (coordinates & city one-hot encodings)

Logs each ablation experiment to MLflow and saves:
  - reports/metrics/ablation_results.json
  - reports/figures/evaluation/ablation_study.png
"""

from __future__ import annotations

# On Windows, lightgbm MUST be imported before pandas/pyarrow to avoid OpenMP C-runtime conflict
import lightgbm as lgb

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import get_logger, setup_logging
from src.utils.mlflow_utils import setup_experiment, start_run
from src.utils.stage_stats import track_stage

logger = get_logger(__name__)


def evaluate_feature_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[dict[str, float], Any]:
    """Train Ridge regression with TimeSeriesSplit CV and evaluate on validation."""
    tscv = TimeSeriesSplit(n_splits=4)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    model = RidgeCV(alphas=alphas, cv=tscv)
    model.fit(X_train, y_train)

    preds_val = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds_val))
    rmse = float(root_mean_squared_error(y_val, preds_val))
    r2 = float(r2_score(y_val, preds_val))

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "best_alpha": float(model.alpha_),
    }
    return metrics, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Feature Ablation Studies")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--val-data", type=Path, default=Path("data/processed/val.parquet"))
    parser.add_argument("--feature-list", type=Path, default=Path("data/processed/feature_list.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/metrics/ablation_results.json"))
    parser.add_argument("--output-fig", type=Path, default=Path("reports/figures/evaluation/ablation_study.png"))
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    args = parser.parse_args()

    setup_logging(stage="ablations")
    logger.info("Starting Phase 7: Feature Ablation Studies...")

    df_train = pd.read_parquet(args.train_data)
    df_val = pd.read_parquet(args.val_data)

    with open(args.feature_list, "r", encoding="utf-8") as f:
        feature_manifest = json.load(f)
    all_features = feature_manifest["feature_names"]

    y_train = df_train["target_aqi"].to_numpy(dtype=float)
    y_val = df_val["target_aqi"].to_numpy(dtype=float)

    # Categorize feature subsets
    weather_cols = [c for c in all_features if any(w in c for w in ["temp", "humidity", "pressure", "precipitation", "wind"])]
    lag_cols = [c for c in all_features if any(l in c for l in ["lag", "rolling", "diff1"])]
    spatial_cols = [c for c in all_features if any(s in c for s in ["city", "latitude", "longitude"])]

    ablation_definitions = {
        "Full Feature Set (All)": all_features,
        "No Weather Features": [c for c in all_features if c not in weather_cols],
        "No Lag Features": [c for c in all_features if c not in lag_cols],
        "No Spatial Features": [c for c in all_features if c not in spatial_cols],
    }

    exp_id = setup_experiment("aqi-ablations", tracking_uri=args.tracking_uri)
    results = []

    # Run baseline (Full) first
    full_feats = ablation_definitions["Full Feature Set (All)"]
    scaler = StandardScaler()
    X_tr_full = scaler.fit_transform(df_train[full_feats])
    X_val_full = scaler.transform(df_val[full_feats])
    full_metrics, _ = evaluate_feature_subset(X_tr_full, y_train, X_val_full, y_val)
    full_rmse = full_metrics["rmse"]

    for name, subset in ablation_definitions.items():
        t0 = time.time()
        logger.info(f"Running ablation: '{name}' ({len(subset)} features)...")

        sc = StandardScaler()
        X_tr = sc.fit_transform(df_train[subset])
        X_va = sc.transform(df_val[subset])

        metrics, model = evaluate_feature_subset(X_tr, y_train, X_va, y_val)
        train_time = round(time.time() - t0, 3)

        delta_rmse = round(metrics["rmse"] - full_rmse, 3)
        pct_degradation = round((metrics["rmse"] - full_rmse) / full_rmse * 100, 2)

        record = {
            "ablation": name,
            "n_features": len(subset),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "delta_rmse_vs_full": delta_rmse,
            "pct_degradation": pct_degradation,
            "train_time_sec": train_time,
        }
        results.append(record)

        # Log run to MLflow
        with start_run(
            step_id="009",
            logbook_entry="009-feature-ablations.md",
            run_name=f"ablation_{name.lower().replace(' ', '_')}",
            experiment_id=exp_id,
            tags={
                "task": "regression",
                "stage": "ablation",
                "ablation_name": name,
                "author": "Srijan Kumar G S",
            },
        ):
            mlflow.log_params({
                "ablation_name": name,
                "n_features": len(subset),
                "best_alpha": metrics["best_alpha"],
            })
            mlflow.log_metrics({
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "delta_rmse_vs_full": delta_rmse,
                "pct_degradation": pct_degradation,
                "train_time_sec": train_time,
            })

    # Save results JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot ablation comparison chart
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    names = [r["ablation"] for r in results]
    rmses = [r["rmse"] for r in results]
    degrads = [r["pct_degradation"] for r in results]
    colors = ["#2563eb", "#d97706", "#dc2626", "#4b5563"]

    # RMSE Plot
    bars1 = ax1.barh(names, rmses, color=colors, height=0.55)
    ax1.set_xlabel("Validation RMSE (lower is better)", fontsize=11)
    ax1.set_title("Ablation Study: Validation RMSE Impact", fontsize=12, pad=10)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=10)

    # Degradation Plot
    bars2 = ax2.barh(names[1:], degrads[1:], color=colors[1:], height=0.55)
    ax2.set_xlabel("% Error Increase vs Full Model (higher = more essential)", fontsize=11)
    ax2.set_title("Feature Group Importance (% Performance Loss When Dropped)", fontsize=12, pad=10)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"+{w:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=200)
    plt.close(fig)
    logger.info(f"Saved ablation chart -> {args.output_fig}")

    # Track stage stats
    inputs = [str(args.train_data), str(args.val_data), str(args.feature_list)]
    outputs = [str(args.output_json), str(args.output_fig)]
    with track_stage("ablations", inputs=inputs, outputs=outputs) as st:
        st.record_input_frame(df_train)

    print("\n" + "=" * 90)
    print("PHASE 7: FEATURE ABLATION STUDY RESULTS")
    print("=" * 90)
    df_res = pd.DataFrame(results)
    print(df_res[["ablation", "n_features", "rmse", "r2", "delta_rmse_vs_full", "pct_degradation"]].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
