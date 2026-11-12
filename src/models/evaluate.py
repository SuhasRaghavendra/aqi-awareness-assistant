"""
src/models/evaluate.py
----------------------
Phase 8: Out-of-sample Test Set Evaluation and Diagnostics.
Touches the holdout test set (data/processed/test.parquet, Jan-Sep 2026) exactly once.
Computes test metrics, error breakdowns by city, and exports diagnostic figures to
reports/figures/evaluation/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

# On Windows, lightgbm MUST be imported before pandas/pyarrow to avoid OpenMP C-runtime conflict
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

from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger(__name__)

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}


def calc_test_reg_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, persist_rmse: float
) -> dict[str, float]:
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
    valid_mask = [(t != -99 and p != -99) for t, p in zip(true_indices, pred_indices, strict=False)]
    if any(valid_mask):
        t_arr = np.array([t for t, v in zip(true_indices, valid_mask, strict=False) if v])
        p_arr = np.array([p for p, v in zip(pred_indices, valid_mask, strict=False) if v])
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
    parser.add_argument(
        "--feature-list", type=Path, default=Path("data/processed/feature_list.json")
    )
    parser.add_argument("--scaler", type=Path, default=Path("data/processed/scaler.joblib"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--val-metrics", type=Path, default=Path("reports/metrics/val_metrics.json")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures/evaluation"))
    parser.add_argument(
        "--output-metrics", type=Path, default=Path("reports/metrics/test_metrics.json")
    )
    args = parser.parse_args()

    setup_logging(stage="evaluate")
    logger.info("Starting Phase 8: Out-of-Sample Test Set Evaluation...")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_test = pd.read_parquet(args.test_data)
    with open(args.feature_list, encoding="utf-8") as f:
        feature_manifest = json.load(f)
    feature_names = feature_manifest["feature_names"]

    logger.info(f"Loaded test holdout: {len(df_test):,} rows across {len(feature_names)} features")

    # Load the single scaler fitted in build_features.py (Step 0.3: one scaler only)
    scaler: StandardScaler = joblib.load(args.scaler)
    logger.info(f"Loaded scaler from {args.scaler} (mean_[0]={scaler.mean_[0]:.4f})")
    X_test_scaled = scaler.transform(df_test[feature_names])
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    y_test_reg = df_test["target_aqi"].to_numpy(dtype=float)
    y_test_clf = df_test["target_category"].to_numpy()
    aqi_test = df_test["aqi"].to_numpy(dtype=float)

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
    # Input type mapping:
    # - "df": baseline heuristics needing raw date/city/metadata columns
    # - "scaled_array": linear models fitted on ndarray (Ridge)
    # - "scaled_df": tree-based regressors fitted on scaled DataFrame with feature names
    # - "residual_*": persistence-anchored models needing today's raw AQI as well
    # - "calibrated": residual model wrapped with per-city offsets (ADR D-031)
    reg_models = [
        ("seasonal_naive", "df"),
        ("station_month_mean", "df"),
        ("ridge", "scaled_array"),
        ("random_forest_reg", "scaled_df"),
        ("xgboost_reg", "scaled_df"),
        ("lightgbm_reg", "scaled_df"),
        ("residual_ridge", "residual_scaled_array"),
        ("residual_lightgbm", "residual_scaled_df"),
        ("residual_lightgbm_monotone", "calibrated"),
        ("residual_lightgbm_monotone_huber", "calibrated"),
    ]

    test_predictions: dict[str, np.ndarray] = {"persistence": preds_test_persist}

    for name, input_type in reg_models:
        model_path = args.models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if input_type == "scaled_df":
            preds = model.predict(X_test_df)
        elif input_type == "scaled_array":
            preds = model.predict(X_test_scaled)
        elif input_type == "residual_scaled_df":
            preds = model.predict(X_test_df, aqi_test)
        elif input_type == "residual_scaled_array":
            preds = model.predict(X_test_scaled, aqi_test)
        elif input_type == "calibrated":
            preds = model.predict(X_test_df, aqi_test, city=df_test["city"].to_numpy())
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
        ("xgboost_clf", False),
    ]

    clf_predictions: dict[str, np.ndarray] = {}

    for name, is_scaled in clf_models:
        model_path = args.models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        bundle = joblib.load(model_path)
        # xgboost_clf is saved as {"model": ..., "label_encoder": ...}
        if isinstance(bundle, dict) and "model" in bundle and "label_encoder" in bundle:
            model = bundle["model"]
            le = bundle["label_encoder"]
            raw_preds = model.predict(X_test_df)
            preds = le.inverse_transform(raw_preds)
        elif is_scaled:
            preds = bundle.predict(X_test_scaled)
        else:
            preds = bundle.predict(X_test_df)

        clf_predictions[name] = preds
        metrics = calc_test_clf_metrics(y_test_clf, preds)
        leaderboard.append({"task": "classification", "model": name, **metrics})

    # =========================================================================
    # PER-CITY METRICS + BOOTSTRAP CI ON SKILL (Step 0.5)
    # =========================================================================
    rng = np.random.default_rng(42)

    def bootstrap_skill_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_persist: np.ndarray,
        n_boot: int = 2000,
        ci: float = 0.95,
        dates: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Bootstrap 95% CI on skill vs persistence, resampled by week."""
        n = len(y_true)
        if dates is not None:
            # Build week-id array then resample whole weeks
            weeks = (
                pd.to_datetime(dates).isocalendar().week.values
                + pd.to_datetime(dates).isocalendar().year.values * 100
            )
            unique_weeks = np.unique(weeks)
            skills = []
            for _ in range(n_boot):
                sampled_weeks = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
                idx = np.concatenate([np.where(weeks == w)[0] for w in sampled_weeks])
                rmse_m = float(root_mean_squared_error(y_true[idx], y_pred[idx]))
                rmse_p = float(root_mean_squared_error(y_true[idx], y_persist[idx]))
                skills.append(1.0 - rmse_m / rmse_p if rmse_p > 0 else 0.0)
        else:
            # Row-level resample fallback
            skills = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                rmse_m = float(root_mean_squared_error(y_true[idx], y_pred[idx]))
                rmse_p = float(root_mean_squared_error(y_true[idx], y_persist[idx]))
                skills.append(1.0 - rmse_m / rmse_p if rmse_p > 0 else 0.0)
        alpha = (1 - ci) / 2
        return float(np.quantile(skills, alpha)), float(np.quantile(skills, 1 - alpha))

    # The per-city analysis follows the frozen, validation-selected champion.
    # Choosing the best model on this holdout would leak test information into
    # model selection (Plan A1), so the decision is read from val_metrics.json.
    with open(args.val_metrics, encoding="utf-8") as f:
        val_metrics = json.load(f)
    selected = [entry for entry in val_metrics if entry.get("selected")]
    if len(selected) == 1:
        best_model_name = str(selected[0]["model"])
    else:
        eligible = [
            e
            for e in val_metrics
            if e.get("task") == "regression"
            and e.get("model") not in {"persistence", "seasonal_naive", "station_month_mean"}
        ]
        best_model_name = (
            min(eligible, key=lambda x: float(x["rmse"]))["model"] if eligible else "persistence"
        )
    if best_model_name not in test_predictions:
        raise RuntimeError(
            f"Validation-selected champion {best_model_name!r} has no test predictions; "
            "refusing to fall back to a test-ranked model"
        )
    logger.info(f"Validation-selected champion for per-city analysis: {best_model_name}")

    df_eval = df_test.copy()
    df_eval["pred_best"] = test_predictions.get(best_model_name, test_predictions["persistence"])
    df_eval["pred_persist"] = test_predictions["persistence"]
    df_eval["date_dt"] = pd.to_datetime(df_eval["date"])

    per_city_records = []
    for city, grp in df_eval.groupby("city"):
        yt = grp["target_aqi"].to_numpy(dtype=float)
        yp = grp["pred_best"].to_numpy(dtype=float)
        ypers = grp["pred_persist"].to_numpy(dtype=float)
        dates_city = grp["date_dt"].to_numpy()

        mae_m = float(mean_absolute_error(yt, yp))
        rmse_m = float(root_mean_squared_error(yt, yp))
        mae_p = float(mean_absolute_error(yt, ypers))
        rmse_p = float(root_mean_squared_error(yt, ypers))
        skill = 1.0 - rmse_m / rmse_p if rmse_p > 0 else 0.0

        ci_lo, ci_hi = bootstrap_skill_ci(yt, yp, ypers, dates=dates_city)
        per_city_records.append(
            {
                "city": city,
                "model": best_model_name,
                "n_rows": len(grp),
                "mae": round(mae_m, 3),
                "rmse": round(rmse_m, 3),
                "mae_persistence": round(mae_p, 3),
                "rmse_persistence": round(rmse_p, 3),
                "skill_vs_persistence": round(skill, 4),
                "skill_ci_lo_95": round(ci_lo, 4),
                "skill_ci_hi_95": round(ci_hi, 4),
            }
        )

    per_city_path = args.output_metrics.parent / "test_per_city_metrics.json"
    with open(per_city_path, "w", encoding="utf-8") as f:
        json.dump(per_city_records, f, indent=2)
    logger.info(f"Per-city metrics saved to {per_city_path}")

    # Save metrics JSON
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    # =========================================================================
    # DIAGNOSTIC FIGURES
    # =========================================================================
    # 1. Predicted vs Actual (Best Regressor vs Persistence)
    best_reg = best_model_name if best_model_name in test_predictions else "persistence"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, m_name, title in zip(
        axes,
        ["persistence", best_reg],
        ["Persistence Baseline", f"Best Model ({best_reg.title()})"],
        strict=False,
    ):
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
    ax.set_title(
        f"Residual Distribution ({best_reg.title()} on 2026 Holdout Test Set)", fontsize=12, pad=10
    )
    ax.set_xlabel("Residual (True AQI - Predicted AQI)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig2_path = args.figures_dir / "residual_histogram.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close(fig)

    # 3. Normalized Confusion Matrix (Random Forest / Best Classifier)
    best_clf = (
        "random_forest_clf" if "random_forest_clf" in clf_predictions else "regress_then_bucket"
    )
    cm = confusion_matrix(
        y_test_clf, clf_predictions[best_clf], labels=CATEGORY_ORDER, normalize="true"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CATEGORY_ORDER,
        yticklabels=CATEGORY_ORDER,
        ax=ax,
    )
    ax.set_title(
        f"Normalized Confusion Matrix ({best_clf.replace('_', ' ').title()})", fontsize=12, pad=10
    )
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
    ax.set_title(
        f"Mean Absolute Error by City ({best_reg.title()} - 2026 Test Set)", fontsize=12, pad=10
    )
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
    outputs = [
        str(args.output_metrics),
        str(fig1_path),
        str(fig2_path),
        str(fig3_path),
        str(fig4_path),
    ]
    with track_stage("evaluate", inputs=inputs, outputs=outputs) as st:
        st.record_input_frame(df_test)

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("PHASE 8: HOLDOUT TEST SET LEADERBOARD (2026 FUTURE DATA)")
    print("=" * 80)
    reg_df = pd.DataFrame([e for e in leaderboard if e["task"] == "regression"])
    print(
        reg_df[
            ["model", "mae", "rmse", "r2", "within_20_rate", "skill_score_vs_persistence"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    clf_df = pd.DataFrame([e for e in leaderboard if e["task"] == "classification"])
    print(
        clf_df[
            ["model", "accuracy", "macro_f1", "weighted_f1", "within_one_category_rate"]
        ].to_string(index=False)
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
