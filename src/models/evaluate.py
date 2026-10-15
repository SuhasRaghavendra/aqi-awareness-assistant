"""
src/models/evaluate.py
----------------------
Phase 8: Out-of-sample Test Set Evaluation and Diagnostics.
Touches the holdout test set (data/processed/test.parquet, Jan-Sep 2026) exactly once.
Computes test metrics, error breakdowns by city, and exports diagnostic figures to
reports/figures/evaluation/.
"""

from __future__ import annotations

# On Windows, lightgbm MUST be imported before pandas/pyarrow to avoid OpenMP C-runtime conflict
import lightgbm as lgb

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from src.features.aqi_utils import aqi_to_category
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger(__name__)

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}


def calc_test_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray, persist_rmse: float) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    within_20 = float(np.mean(np.abs(y_true - y_pred) <= 20.0))
    skill_score = 1.0 - (rmse / persist_rmse) if persist_rmse > 0 else 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "median_ae": round(medae, 3),
        "within_20_rate": round(within_20, 4),
        "skill_score_vs_persistence": round(float(skill_score), 4),
    }


def calc_test_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    true_indices = [CAT_TO_IDX.get(cat, -99) for cat in y_true]
    pred_indices = [CAT_TO_IDX.get(cat, -99) for cat in y_pred]
    valid_mask = [(t != -99 and p != -99) for t, p in zip(true_indices, pred_indices)]
    if any(valid_mask):
        t_arr = np.array([t for t, v in zip(true_indices, valid_mask) if v])
        p_arr = np.array([p for p, v in zip(pred_indices, valid_mask) if v])
        within_one = float(np.mean(np.abs(t_arr - p_arr) <= 1))
    else:
        within_one = acc

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "within_one_category_rate": round(within_one, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: Out-of-Sample Test Evaluation")
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--feature-list", type=Path, default=Path("data/processed/feature_list.json"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures/evaluation"))
    parser.add_argument("--output-metrics", type=Path, default=Path("reports/metrics/test_metrics.json"))
    args = parser.parse_args()

    setup_logging(stage="evaluate")
    logger.info("Starting Phase 8: Out-of-Sample Test Set Evaluation...")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_test = pd.read_parquet(args.test_data)
    df_train = pd.read_parquet(args.train_data)
    with open(args.feature_list, "r", encoding="utf-8") as f:
        feature_manifest = json.load(f)
    feature_names = feature_manifest["feature_names"]

    logger.info(f"Loaded test holdout: {len(df_test):,} rows across {len(feature_names)} features")

    # Fit scaler on train only (ensuring identical feature alignment)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_names])
    X_test_scaled = scaler.transform(df_test[feature_names])
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    y_test_reg = df_test["target_aqi"].to_numpy(dtype=float)
    y_test_clf = df_test["target_category"].to_numpy()

    # Persistence benchmark
    persist_model = joblib.load(args.models_dir / "persistence.joblib")
    preds_test_persist = persist_model.predict(df_test)
    m_persist = calc_test_reg_metrics(y_test_reg, preds_test_persist, persist_rmse=1.0)
    persist_rmse = m_persist["rmse"]
    m_persist["skill_score_vs_persistence"] = 0.0

    leaderboard: list[dict[str, Any]] = [
        {"task": "regression", "model": "persistence", **m_persist}
    ]

    # Evaluate Regression Models
    reg_models = [
        ("seasonal_naive", False, False),
        ("station_month_mean", False, False),
        ("ridge", True, True),
        ("random_forest_reg", False, True),
        ("xgboost_reg", False, True),
        ("lightgbm_reg", False, True),
    ]

    test_predictions: dict[str, np.ndarray] = {"persistence": preds_test_persist}

    for name, is_scaled, uses_df in reg_models:
        model_path = args.models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if uses_df:
            preds = model.predict(X_test_df if is_scaled else df_test[feature_names])
        elif is_scaled:
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(df_test)

        test_predictions[name] = preds
        metrics = calc_test_reg_metrics(y_test_reg, preds, persist_rmse=persist_rmse)
        leaderboard.append({"task": "regression", "model": name, **metrics})

    # Evaluate Classification Models
    clf_models = [
        ("majority_class", False),
        ("regress_then_bucket", False),
        ("logistic_regression", True),
        ("random_forest_clf", False),
        ("lightgbm_clf", False),
    ]

    clf_predictions: dict[str, np.ndarray] = {}

    for name, is_scaled in clf_models:
        model_path = args.models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if is_scaled:
            preds = model.predict(X_test_scaled)
        elif name == "regress_then_bucket":
            preds = model.predict(X_test_df)
        else:
            preds = model.predict(X_test_df)

        clf_predictions[name] = preds
        metrics = calc_test_clf_metrics(y_test_clf, preds)
        leaderboard.append({"task": "classification", "model": name, **metrics})

    # Save metrics JSON
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    # =========================================================================
    # DIAGNOSTIC FIGURES
    # =========================================================================
    # 1. Predicted vs Actual (Ridge / Best Regressor vs Persistence)
    best_reg = "ridge" if "ridge" in test_predictions else "persistence"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, m_name, title in zip(axes, ["persistence", best_reg], ["Persistence Baseline", f"Best Model ({best_reg.title()})"]):
        ax.scatter(y_test_reg, test_predictions[m_name], alpha=0.25, color="#2563eb", s=15)
        ax.plot([0, 500], [0, 500], "r--", linewidth=1.5, label="Ideal (y = x)")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("True Next-Day AQI", fontsize=10)
        ax.set_ylabel("Predicted Next-Day AQI", fontsize=10)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    fig1_path = args.figures_dir / "predicted_vs_actual.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close(fig)

    # 2. Residual Histogram
    residuals = y_test_reg - test_predictions[best_reg]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True, ax=ax, color="#059669")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Distribution ({best_reg.title()} on 2026 Holdout Test Set)", fontsize=12, pad=10)
    ax.set_xlabel("Residual (True AQI - Predicted AQI)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig2_path = args.figures_dir / "residual_histogram.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close(fig)

    # 3. Normalized Confusion Matrix (Random Forest / Best Classifier)
    best_clf = "random_forest_clf" if "random_forest_clf" in clf_predictions else "regress_then_bucket"
    cm = confusion_matrix(y_test_clf, clf_predictions[best_clf], labels=CATEGORY_ORDER, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CATEGORY_ORDER, yticklabels=CATEGORY_ORDER, ax=ax)
    ax.set_title(f"Normalized Confusion Matrix ({best_clf.replace('_', ' ').title()})", fontsize=12, pad=10)
    ax.set_xlabel("Predicted CPCB Category")
    ax.set_ylabel("True CPCB Category")
    plt.tight_layout()
    fig3_path = args.figures_dir / "confusion_matrix.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close(fig)

    # 4. Error Breakdown by City
    df_eval = df_test.copy()
    df_eval["pred_aqi"] = test_predictions[best_reg]
    df_eval["abs_error"] = np.abs(df_eval["target_aqi"] - df_eval["pred_aqi"])
    city_mae = df_eval.groupby("city")["abs_error"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    city_mae.plot(kind="bar", ax=ax, color="#4f46e5")
    ax.set_title(f"Mean Absolute Error by City ({best_reg.title()} - 2026 Test Set)", fontsize=12, pad=10)
    ax.set_ylabel("MAE (AQI points)")
    ax.set_xlabel("City")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig4_path = args.figures_dir / "city_error_breakdown.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close(fig)

    # Stage Tracking
    inputs = [str(args.test_data), str(args.models_dir)]
    outputs = [str(args.output_metrics), str(fig1_path), str(fig2_path), str(fig3_path), str(fig4_path)]
    with track_stage("evaluate", inputs=inputs, outputs=outputs) as st:
        st.record_input_frame(df_test)

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("PHASE 8: HOLDOUT TEST SET LEADERBOARD (2026 FUTURE DATA)")
    print("=" * 80)
    reg_df = pd.DataFrame([e for e in leaderboard if e["task"] == "regression"])
    print(reg_df[["model", "mae", "rmse", "r2", "within_20_rate", "skill_score_vs_persistence"]].to_string(index=False))

    print("\n" + "=" * 80)
    clf_df = pd.DataFrame([e for e in leaderboard if e["task"] == "classification"])
    print(clf_df[["model", "accuracy", "macro_f1", "weighted_f1", "within_one_category_rate"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
