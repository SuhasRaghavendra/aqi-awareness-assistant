"""
src/models/train.py
-------------------
Phase 6 / Phase 7: Train baseline regression and classification models,
log parameters, metrics, and model artifacts to MLflow (SQLite backend),
and export validation leaderboard to reports/metrics/val_metrics.json.

Supported Models:
- Regression (Task A: Next-Day AQI):
  1. persistence
  2. seasonal_naive
  3. city_month_mean
  4. ridge (TimeSeriesSplit alpha selection)
  5. random_forest
  6. xgboost
  7. lightgbm
- Classification (Task B: Next-Day CPCB AQI Category):
  1. majority_class
  2. regress_then_bucket
  3. logistic_regression
  4. random_forest_clf
  5. xgboost_clf
  6. lightgbm_clf

Usage:
    python -m src.models.train
"""

from __future__ import annotations

# On Windows, lightgbm MUST be imported before pandas/pyarrow to avoid OpenMP C-runtime conflict
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import load_params
from src.features.aqi_utils import aqi_to_category
from src.models.baselines import (
    MajorityClassBaseline,
    PersistenceBaseline,
    RegressThenBucketClassifier,
    StationMonthMeanBaseline,
    WeeklySeasonalNaiveBaseline,
)
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.mlflow_utils import log_params_yaml, setup_experiment, start_run
from src.utils.stage_stats import track_stage

logger = get_logger(__name__)

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}


def calc_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, persistence_rmse: float | None = None
) -> dict[str, float]:
    """Calculate standard AQI regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    within_20 = float(np.mean(np.abs(y_true - y_pred) <= 20.0))

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "median_ae": round(medae, 3),
        "within_20_rate": round(within_20, 4),
    }

    if persistence_rmse is not None and persistence_rmse > 0:
        skill_score = 1.0 - (rmse / persistence_rmse)
        metrics["skill_score_vs_persistence"] = round(float(skill_score), 4)

    return metrics


def calc_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate multi-class classification metrics for AQI categories."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Within-one-category accuracy: |idx_pred - idx_true| <= 1
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
    parser = argparse.ArgumentParser(description="Phase 6: Train Baseline Models & MLflow Tracking")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--val-data", type=Path, default=Path("data/processed/val.parquet"))
    parser.add_argument("--feature-list", type=Path, default=Path("data/processed/feature_list.json"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-metrics", type=Path, default=Path("reports/metrics/val_metrics.json"))
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    args = parser.parse_args()

    setup_logging(stage="train")
    logger.info("Starting Phase 6: Baseline Model Training and MLflow Tracking...")

    params = load_params()
    seed = int(params.get("seed", 42))
    np.random.seed(seed)

    # 1. Load Data
    df_train = pd.read_parquet(args.train_data)
    df_val = pd.read_parquet(args.val_data)
    with open(args.feature_list, "r", encoding="utf-8") as f:
        feature_manifest = json.load(f)
    feature_names = feature_manifest["feature_names"]

    logger.info(
        f"Loaded splits: Train={len(df_train):,} rows, Val={len(df_val):,} rows across {len(feature_names)} features"
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    # Targets
    y_train_reg = df_train["target_aqi"].to_numpy(dtype=float)
    y_val_reg = df_val["target_aqi"].to_numpy(dtype=float)

    y_train_clf = df_train["target_category"].to_numpy()
    y_val_clf = df_val["target_category"].to_numpy()

    # Pre-scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feature_names])
    X_val_scaled = scaler.transform(df_val[feature_names])

    # Scaled DataFrame with columns preserved for tree models/tabular indexing
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)

    # Setup MLflow experiments
    reg_exp_id = setup_experiment("aqi-regression-baselines", tracking_uri=args.tracking_uri)
    clf_exp_id = setup_experiment("aqi-classification-baselines", tracking_uri=args.tracking_uri)

    leaderboard: list[dict[str, Any]] = []
    saved_model_paths: list[str] = []

    # =========================================================================
    # TASK A: REGRESSION MODELS
    # =========================================================================
    logger.info("\n" + "=" * 60 + "\nTRAINING TASK A: REGRESSION MODELS\n" + "=" * 60)

    # 1. Persistence Baseline
    t0 = time.time()
    persistence = PersistenceBaseline(aqi_col="aqi")
    persistence.fit(df_train, y_train_reg)
    preds_val_persist = persistence.predict(df_val)
    time_persist = time.time() - t0
    m_persist = calc_regression_metrics(y_val_reg, preds_val_persist)
    persist_rmse = m_persist["rmse"]
    m_persist["skill_score_vs_persistence"] = 0.0

    joblib.dump(persistence, args.models_dir / "persistence.joblib")
    saved_model_paths.append(str(args.models_dir / "persistence.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="persistence",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "heuristic", "stage": "baseline", "author": "Srijan Kumar G S"},
    ) as run:
        mlflow.log_params({"model_type": "PersistenceBaseline", "seed": seed, "n_features": len(feature_names)})
        mlflow.log_metrics({**m_persist, "train_time_sec": round(time_persist, 3)})
        mlflow.log_artifact(str(args.models_dir / "persistence.joblib"))
        log_params_yaml()

    leaderboard.append({"task": "regression", "model": "persistence", **m_persist, "time_sec": round(time_persist, 2)})

    # 2. Weekly Seasonal Naive
    t0 = time.time()
    seasonal_naive = WeeklySeasonalNaiveBaseline(lag7_col="aqi_lag_7", fallback_col="aqi")
    seasonal_naive.fit(df_train, y_train_reg)
    preds_val_sn = seasonal_naive.predict(df_val)
    time_sn = time.time() - t0
    m_sn = calc_regression_metrics(y_val_reg, preds_val_sn, persistence_rmse=persist_rmse)

    joblib.dump(seasonal_naive, args.models_dir / "seasonal_naive.joblib")
    saved_model_paths.append(str(args.models_dir / "seasonal_naive.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="seasonal_naive",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "heuristic", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "WeeklySeasonalNaiveBaseline", "seed": seed})
        mlflow.log_metrics({**m_sn, "train_time_sec": round(time_sn, 3)})
        mlflow.log_artifact(str(args.models_dir / "seasonal_naive.joblib"))

    leaderboard.append({"task": "regression", "model": "seasonal_naive", **m_sn, "time_sec": round(time_sn, 2)})

    # 3. Station-Month Mean Baseline
    t0 = time.time()
    st_month_mean = StationMonthMeanBaseline(station_col="location_id", month_col="month", date_col="date")
    st_month_mean.fit(df_train, y_train_reg)
    preds_val_smm = st_month_mean.predict(df_val)
    time_smm = time.time() - t0
    m_smm = calc_regression_metrics(y_val_reg, preds_val_smm, persistence_rmse=persist_rmse)

    joblib.dump(st_month_mean, args.models_dir / "station_month_mean.joblib")
    saved_model_paths.append(str(args.models_dir / "station_month_mean.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="station_month_mean",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "heuristic", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "StationMonthMeanBaseline", "seed": seed})
        mlflow.log_metrics({**m_smm, "train_time_sec": round(time_smm, 3)})
        mlflow.log_artifact(str(args.models_dir / "station_month_mean.joblib"))

    leaderboard.append({"task": "regression", "model": "station_month_mean", **m_smm, "time_sec": round(time_smm, 2)})

    # 4. Ridge Regression (TimeSeriesSplit Cross-Validation)
    t0 = time.time()
    tscv = TimeSeriesSplit(n_splits=4)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeCV(alphas=alphas, cv=tscv)
    ridge.fit(X_train_scaled, y_train_reg)
    preds_val_ridge = ridge.predict(X_val_scaled)
    time_ridge = time.time() - t0
    m_ridge = calc_regression_metrics(y_val_reg, preds_val_ridge, persistence_rmse=persist_rmse)

    joblib.dump(ridge, args.models_dir / "ridge.joblib")
    saved_model_paths.append(str(args.models_dir / "ridge.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="ridge",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "linear", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "RidgeCV", "best_alpha": float(ridge.alpha_), "cv_folds": 4, "seed": seed})
        mlflow.log_metrics({**m_ridge, "train_time_sec": round(time_ridge, 3)})
        mlflow.log_artifact(str(args.models_dir / "ridge.joblib"))

    leaderboard.append({"task": "regression", "model": "ridge", **m_ridge, "time_sec": round(time_ridge, 2)})

    # 5. Random Forest Regressor
    t0 = time.time()
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    rf_reg.fit(X_train_df, y_train_reg)
    preds_val_rf = rf_reg.predict(X_val_df)
    time_rf = time.time() - t0
    m_rf = calc_regression_metrics(y_val_reg, preds_val_rf, persistence_rmse=persist_rmse)

    joblib.dump(rf_reg, args.models_dir / "random_forest_reg.joblib")
    saved_model_paths.append(str(args.models_dir / "random_forest_reg.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="random_forest_reg",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "tree_ensemble", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100, "max_depth": 12, "seed": seed})
        mlflow.log_metrics({**m_rf, "train_time_sec": round(time_rf, 3)})
        mlflow.log_artifact(str(args.models_dir / "random_forest_reg.joblib"))

    leaderboard.append({"task": "regression", "model": "random_forest_reg", **m_rf, "time_sec": round(time_rf, 2)})

    # 6. XGBoost Regressor
    t0 = time.time()
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=-1)
    xgb_reg.fit(X_train_df, y_train_reg)
    preds_val_xgb = xgb_reg.predict(X_val_df)
    time_xgb = time.time() - t0
    m_xgb = calc_regression_metrics(y_val_reg, preds_val_xgb, persistence_rmse=persist_rmse)

    joblib.dump(xgb_reg, args.models_dir / "xgboost_reg.joblib")
    saved_model_paths.append(str(args.models_dir / "xgboost_reg.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="xgboost_reg",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "gradient_boosting", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "XGBRegressor", "n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "seed": seed})
        mlflow.log_metrics({**m_xgb, "train_time_sec": round(time_xgb, 3)})
        mlflow.log_artifact(str(args.models_dir / "xgboost_reg.joblib"))

    leaderboard.append({"task": "regression", "model": "xgboost_reg", **m_xgb, "time_sec": round(time_xgb, 2)})

    # 7. LightGBM Regressor
    t0 = time.time()
    lgbm_reg = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=seed, verbosity=-1)
    lgbm_reg.fit(X_train_df, y_train_reg)
    preds_val_lgbm = lgbm_reg.predict(X_val_df)
    time_lgbm = time.time() - t0
    m_lgbm = calc_regression_metrics(y_val_reg, preds_val_lgbm, persistence_rmse=persist_rmse)

    joblib.dump(lgbm_reg, args.models_dir / "lightgbm_reg.joblib")
    saved_model_paths.append(str(args.models_dir / "lightgbm_reg.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="lightgbm_reg",
        experiment_id=reg_exp_id,
        tags={"task": "regression", "model_family": "gradient_boosting", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "LGBMRegressor", "n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "seed": seed})
        mlflow.log_metrics({**m_lgbm, "train_time_sec": round(time_lgbm, 3)})
        mlflow.log_artifact(str(args.models_dir / "lightgbm_reg.joblib"))

    leaderboard.append({"task": "regression", "model": "lightgbm_reg", **m_lgbm, "time_sec": round(time_lgbm, 2)})

    # Identify best regressor for downstream "regress-then-bucket"
    best_reg_entry = min(
        [entry for entry in leaderboard if entry["task"] == "regression"],
        key=lambda x: x["rmse"],
    )
    best_reg_name = best_reg_entry["model"]
    best_reg_model = joblib.load(args.models_dir / f"{best_reg_name}.joblib")
    logger.info(f"Best validation regressor: {best_reg_name} (RMSE={best_reg_entry['rmse']})")

    # =========================================================================
    # TASK B: CLASSIFICATION MODELS
    # =========================================================================
    logger.info("\n" + "=" * 60 + "\nTRAINING TASK B: CLASSIFICATION MODELS\n" + "=" * 60)

    # 1. Majority Class Baseline
    t0 = time.time()
    majority = MajorityClassBaseline()
    majority.fit(X_train_df, y_train_clf)
    preds_val_maj = majority.predict(X_val_df)
    time_maj = time.time() - t0
    m_maj = calc_classification_metrics(y_val_clf, preds_val_maj)

    joblib.dump(majority, args.models_dir / "majority_class.joblib")
    saved_model_paths.append(str(args.models_dir / "majority_class.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="majority_class",
        experiment_id=clf_exp_id,
        tags={"task": "classification", "model_family": "heuristic", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "MajorityClassBaseline", "majority_class": majority.majority_class_})
        mlflow.log_metrics({**m_maj, "train_time_sec": round(time_maj, 3)})
        mlflow.log_artifact(str(args.models_dir / "majority_class.joblib"))

    leaderboard.append({"task": "classification", "model": "majority_class", **m_maj, "time_sec": round(time_maj, 2)})

    # 2. Regress-Then-Bucket (from best regressor)
    t0 = time.time()
    reg_bucket_clf = RegressThenBucketClassifier(regressor=best_reg_model)
    preds_val_rb = reg_bucket_clf.predict(X_val_df)
    time_rb = time.time() - t0
    m_rb = calc_classification_metrics(y_val_clf, preds_val_rb)

    joblib.dump(reg_bucket_clf, args.models_dir / "regress_then_bucket.joblib")
    saved_model_paths.append(str(args.models_dir / "regress_then_bucket.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="regress_then_bucket",
        experiment_id=clf_exp_id,
        tags={"task": "classification", "model_family": "hybrid", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "RegressThenBucketClassifier", "base_regressor": best_reg_name})
        mlflow.log_metrics({**m_rb, "train_time_sec": round(time_rb, 3)})
        mlflow.log_artifact(str(args.models_dir / "regress_then_bucket.joblib"))

    leaderboard.append({"task": "classification", "model": f"regress_then_bucket_{best_reg_name}", **m_rb, "time_sec": round(time_rb, 2)})

    # 3. Logistic Regression (balanced)
    t0 = time.time()
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    log_reg.fit(X_train_scaled, y_train_clf)
    preds_val_lr = log_reg.predict(X_val_scaled)
    time_lr = time.time() - t0
    m_lr = calc_classification_metrics(y_val_clf, preds_val_lr)

    joblib.dump(log_reg, args.models_dir / "logistic_regression.joblib")
    saved_model_paths.append(str(args.models_dir / "logistic_regression.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="logistic_regression",
        experiment_id=clf_exp_id,
        tags={"task": "classification", "model_family": "linear", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000, "seed": seed})
        mlflow.log_metrics({**m_lr, "train_time_sec": round(time_lr, 3)})
        mlflow.log_artifact(str(args.models_dir / "logistic_regression.joblib"))

    leaderboard.append({"task": "classification", "model": "logistic_regression", **m_lr, "time_sec": round(time_lr, 2)})

    # 4. Random Forest Classifier
    t0 = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", max_depth=12, random_state=seed, n_jobs=-1)
    rf_clf.fit(X_train_df, y_train_clf)
    preds_val_rfc = rf_clf.predict(X_val_df)
    time_rfc = time.time() - t0
    m_rfc = calc_classification_metrics(y_val_clf, preds_val_rfc)

    joblib.dump(rf_clf, args.models_dir / "random_forest_clf.joblib")
    saved_model_paths.append(str(args.models_dir / "random_forest_clf.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="random_forest_clf",
        experiment_id=clf_exp_id,
        tags={"task": "classification", "model_family": "tree_ensemble", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100, "class_weight": "balanced_subsample", "seed": seed})
        mlflow.log_metrics({**m_rfc, "train_time_sec": round(time_rfc, 3)})
        mlflow.log_artifact(str(args.models_dir / "random_forest_clf.joblib"))

    leaderboard.append({"task": "classification", "model": "random_forest_clf", **m_rfc, "time_sec": round(time_rfc, 2)})

    # 5. LightGBM Classifier
    t0 = time.time()
    lgbm_clf = LGBMClassifier(n_estimators=100, class_weight="balanced", max_depth=6, random_state=seed, verbosity=-1)
    lgbm_clf.fit(X_train_df, y_train_clf)
    preds_val_lgbmc = lgbm_clf.predict(X_val_df)
    time_lgbmc = time.time() - t0
    m_lgbmc = calc_classification_metrics(y_val_clf, preds_val_lgbmc)

    joblib.dump(lgbm_clf, args.models_dir / "lightgbm_clf.joblib")
    saved_model_paths.append(str(args.models_dir / "lightgbm_clf.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="lightgbm_clf",
        experiment_id=clf_exp_id,
        tags={"task": "classification", "model_family": "gradient_boosting", "stage": "baseline", "author": "Srijan Kumar G S"},
    ):
        mlflow.log_params({"model_type": "LGBMClassifier", "n_estimators": 100, "class_weight": "balanced", "seed": seed})
        mlflow.log_metrics({**m_lgbmc, "train_time_sec": round(time_lgbmc, 3)})
        mlflow.log_artifact(str(args.models_dir / "lightgbm_clf.joblib"))

    leaderboard.append({"task": "classification", "model": "lightgbm_clf", **m_lgbmc, "time_sec": round(time_lgbmc, 2)})

    # Write output metrics JSON
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    # Track stage stats
    inputs = [str(args.train_data), str(args.val_data), str(args.feature_list)]
    outputs = [str(args.output_metrics)] + saved_model_paths
    with track_stage("train", inputs=inputs, outputs=outputs) as st:
        st.record_input_frame(df_train)
        st.record_output_frame(df_val)

    # Print Leaderboards
    print("\n" + "=" * 80)
    print("PHASE 6: VALIDATION LEADERBOARD (TASK A: REGRESSION)")
    print("=" * 80)
    reg_df = pd.DataFrame([e for e in leaderboard if e["task"] == "regression"])
    print(reg_df[["model", "mae", "rmse", "r2", "within_20_rate", "skill_score_vs_persistence", "time_sec"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("PHASE 6: VALIDATION LEADERBOARD (TASK B: CLASSIFICATION)")
    print("=" * 80)
    clf_df = pd.DataFrame([e for e in leaderboard if e["task"] == "classification"])
    print(clf_df[["model", "accuracy", "macro_f1", "weighted_f1", "within_one_category_rate", "time_sec"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
