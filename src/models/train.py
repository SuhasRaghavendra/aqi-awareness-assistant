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

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib

# On Windows, lightgbm MUST be imported before pandas/pyarrow to avoid OpenMP C-runtime conflict
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
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
from src.features.serving_features import build_feature_frame
from src.models.baselines import (
    MajorityClassBaseline,
    PersistenceBaseline,
    RegressThenBucketClassifier,
    StationMonthMeanBaseline,
    WeeklySeasonalNaiveBaseline,
)
from src.models.calibration import (
    CityCalibratedRegressor,
    forward_calibrated_predictions,
    per_city_offsets,
)
from src.models.monotone import build_monotone_constraints, constrained_features
from src.models.residual import ResidualRegressor
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


def _residual_cv_scores(
    config: dict[str, float | int | list[int]],
    X: pd.DataFrame,
    y: np.ndarray,
    aqi: np.ndarray,
    seed: int,
    n_splits: int = 4,
) -> list[float]:
    """Per-fold TimeSeriesSplit RMSE for a residual LightGBM configuration."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        model = ResidualRegressor(
            LGBMRegressor(**config, random_state=seed, verbosity=-1, n_jobs=1)
        )
        model.fit(X.iloc[train_idx], y[train_idx], aqi[train_idx])
        preds = model.predict(X.iloc[test_idx], aqi[test_idx])
        scores.append(float(root_mean_squared_error(y[test_idx], preds)))
    return scores


def _residual_cv_rmse(
    config: dict[str, float | int | list[int]],
    X: pd.DataFrame,
    y: np.ndarray,
    aqi: np.ndarray,
    seed: int,
    n_splits: int = 4,
) -> float:
    """Mean TimeSeriesSplit RMSE for a residual LightGBM configuration (train only)."""
    return float(np.mean(_residual_cv_scores(config, X, y, aqi, seed, n_splits)))


def _scaled_frame(
    scaler: StandardScaler, frame: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    """Scale a frame while keeping the training feature order and names."""
    scaled = scaler.transform(frame[feature_names])
    return pd.DataFrame(scaled, columns=feature_names)


def _predict_artifact(
    model: Any,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: list[str],
    aqi_max: float,
) -> np.ndarray:
    """Predict with the residual/calibrated contract, clipped to the AQI scale."""
    scaled_df = _scaled_frame(scaler, frame, feature_names)
    aqi = frame["aqi"].to_numpy(dtype=float)
    if isinstance(model, CityCalibratedRegressor):
        preds = model.predict(scaled_df, aqi, city=frame["city"].to_numpy())
    else:
        preds = model.predict(scaled_df, aqi)
    return np.clip(np.asarray(preds, dtype=float), 0.0, float(aqi_max))


def _val_max_city_ratio(frame: pd.DataFrame, preds: np.ndarray, persistence: np.ndarray) -> float:
    """Largest per-city MAE ratio against persistence on the validation split."""
    ratios = []
    for _, group in frame.assign(_pred=preds, _persist=persistence).groupby("city"):
        truth = group["target_aqi"].to_numpy(dtype=float)
        mae_model = float(np.mean(np.abs(truth - group["_pred"].to_numpy(dtype=float))))
        mae_persist = float(np.mean(np.abs(truth - group["_persist"].to_numpy(dtype=float))))
        ratios.append(mae_model / mae_persist if mae_persist > 0 else 0.0)
    return float(max(ratios)) if ratios else 0.0


def _train_medians(train_df: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    """Recompute the train medians identically to the model-risk reference stats."""
    medians = train_df[feature_names].median().fillna(0.0)
    return {name: float(medians[name]) for name in feature_names}


def _metamorphic_acceptance(
    model: Any,
    val_df: pd.DataFrame,
    feature_names: list[str],
    params: dict[str, Any],
    train_medians: dict[str, float],
    scaler: StandardScaler,
    aqi_max: float,
) -> dict[str, Any]:
    """Pre-registered metamorphic check on the last 60 validation days per city.

    This mirrors the fixture tests but runs on the real validation split, so the
    promotion decision never touches the holdout.
    """
    frame = val_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    last = frame.groupby("city")["date"].transform("max")
    sample = frame[frame["date"] >= last - pd.Timedelta(days=59)].copy()
    baseline = _predict_artifact(
        model,
        build_feature_frame(sample, feature_names, params, train_medians),
        scaler,
        feature_names,
        aqi_max,
    )

    def fire_mutation(panel: pd.DataFrame) -> None:
        panel["fire_count_500km"] = panel["fire_count_500km"] * 5.0
        panel["fire_frp_sum_500km"] = panel["fire_frp_sum_500km"] * 5.0

    def wind_mutation(panel: pd.DataFrame) -> None:
        panel["wind_speed_10m_mean"] = panel["wind_speed_10m_mean"] / 3.0

    checks: dict[str, dict[str, float]] = {}
    for label, mutate in (("fire_x5", fire_mutation), ("wind_div3", wind_mutation)):
        treated = sample.copy()
        mutate(treated)
        delta = (
            _predict_artifact(
                model,
                build_feature_frame(treated, feature_names, params, train_medians),
                scaler,
                feature_names,
                aqi_max,
            )
            - baseline
        )
        checks[label] = {
            "mean_delta": round(float(delta.mean()), 4),
            "non_monotone_rate": round(float(np.mean(delta < -2.0)), 4),
        }

    max_rate = max(check["non_monotone_rate"] for check in checks.values())
    min_mean = min(check["mean_delta"] for check in checks.values())
    return {
        "passed": bool(max_rate <= 0.05 and min_mean >= 0.0),
        "tolerance": 0.05,
        "rows": int(len(sample)),
        "checks": checks,
        "max_non_monotone_rate": max_rate,
        "min_mean_delta": min_mean,
    }


def _mark_selected(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered selection rule and flag one champion.

    D-030: a monotone candidate that passes its pre-registered acceptance is
    preferred over the unconstrained incumbent even when its validation RMSE is
    up to 2% worse, because it restores the 5% metamorphic gate structurally.
    D-031: within the preferred pool, minimise validation RMSE subject to
    validation per-city MAE ratio <= 1.05 when any candidate satisfies it.
    """
    eligible = [
        entry
        for entry in leaderboard
        if entry.get("task") == "regression" and entry.get("model") not in model_risk_exclusions()
    ]
    accepted = [
        entry for entry in eligible if entry.get("pre_registered_acceptance", {}).get("passed")
    ]
    pool = accepted or eligible
    conforming = [entry for entry in pool if float(entry.get("val_max_city_ratio", np.inf)) <= 1.05]
    chosen = min(conforming or pool, key=lambda entry: float(entry["rmse"]))
    for entry in eligible:
        entry["selected"] = False
    chosen["selected"] = True
    return chosen


def model_risk_exclusions() -> set[str]:
    """Champion exclusions shared with ``model_risk.CHAMPION_EXCLUDED``."""
    return {"persistence", "seasonal_naive", "station_month_mean", "random_forest_reg"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Train Baseline Models & MLflow Tracking")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--val-data", type=Path, default=Path("data/processed/val.parquet"))
    parser.add_argument(
        "--feature-list", type=Path, default=Path("data/processed/feature_list.json")
    )
    parser.add_argument("--scaler", type=Path, default=Path("data/processed/scaler.joblib"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output-metrics", type=Path, default=Path("reports/metrics/val_metrics.json")
    )
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
    with open(args.feature_list, encoding="utf-8") as f:
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
    # Load the single scaler fitted in build_features.py (Step 0.3: one scaler only)
    scaler: StandardScaler = joblib.load(args.scaler)
    logger.info(f"Loaded scaler from {args.scaler} (mean_[0]={scaler.mean_[0]:.4f})")
    X_train_scaled = scaler.transform(df_train[feature_names])
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
        tags={
            "task": "regression",
            "model_family": "heuristic",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {"model_type": "PersistenceBaseline", "seed": seed, "n_features": len(feature_names)}
        )
        mlflow.log_metrics({**m_persist, "train_time_sec": round(time_persist, 3)})
        mlflow.log_artifact(str(args.models_dir / "persistence.joblib"))
        log_params_yaml()

    leaderboard.append(
        {
            "task": "regression",
            "model": "persistence",
            **m_persist,
            "time_sec": round(time_persist, 2),
        }
    )

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
        tags={
            "task": "regression",
            "model_family": "heuristic",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params({"model_type": "WeeklySeasonalNaiveBaseline", "seed": seed})
        mlflow.log_metrics({**m_sn, "train_time_sec": round(time_sn, 3)})
        mlflow.log_artifact(str(args.models_dir / "seasonal_naive.joblib"))

    leaderboard.append(
        {"task": "regression", "model": "seasonal_naive", **m_sn, "time_sec": round(time_sn, 2)}
    )

    # 3. Station-Month Mean Baseline
    t0 = time.time()
    st_col = "city" if "city" in df_train.columns else "location_id"
    st_month_mean = StationMonthMeanBaseline(station_col=st_col, month_col="month", date_col="date")
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
        tags={
            "task": "regression",
            "model_family": "heuristic",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params({"model_type": "StationMonthMeanBaseline", "seed": seed})
        mlflow.log_metrics({**m_smm, "train_time_sec": round(time_smm, 3)})
        mlflow.log_artifact(str(args.models_dir / "station_month_mean.joblib"))

    leaderboard.append(
        {
            "task": "regression",
            "model": "station_month_mean",
            **m_smm,
            "time_sec": round(time_smm, 2),
        }
    )

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
        tags={
            "task": "regression",
            "model_family": "linear",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "RidgeCV",
                "best_alpha": float(ridge.alpha_),
                "cv_folds": 4,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_ridge, "train_time_sec": round(time_ridge, 3)})
        mlflow.log_artifact(str(args.models_dir / "ridge.joblib"))

    leaderboard.append(
        {"task": "regression", "model": "ridge", **m_ridge, "time_sec": round(time_ridge, 2)}
    )

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
        tags={
            "task": "regression",
            "model_family": "tree_ensemble",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 12,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_rf, "train_time_sec": round(time_rf, 3)})
        mlflow.log_artifact(str(args.models_dir / "random_forest_reg.joblib"))

    leaderboard.append(
        {"task": "regression", "model": "random_forest_reg", **m_rf, "time_sec": round(time_rf, 2)}
    )

    # 6. XGBoost Regressor
    t0 = time.time()
    xgb_reg = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=-1
    )
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
        tags={
            "task": "regression",
            "model_family": "gradient_boosting",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "XGBRegressor",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_xgb, "train_time_sec": round(time_xgb, 3)})
        mlflow.log_artifact(str(args.models_dir / "xgboost_reg.joblib"))

    leaderboard.append(
        {"task": "regression", "model": "xgboost_reg", **m_xgb, "time_sec": round(time_xgb, 2)}
    )

    # 7. LightGBM Regressor
    t0 = time.time()
    lgbm_reg = LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=6, random_state=seed, verbosity=-1
    )
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
        tags={
            "task": "regression",
            "model_family": "gradient_boosting",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "LGBMRegressor",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_lgbm, "train_time_sec": round(time_lgbm, 3)})
        mlflow.log_artifact(str(args.models_dir / "lightgbm_reg.joblib"))

    leaderboard.append(
        {"task": "regression", "model": "lightgbm_reg", **m_lgbm, "time_sec": round(time_lgbm, 2)}
    )

    # =========================================================================
    # PHASE 0.4: RESIDUAL (PERSISTENCE-ANCHORED) CANDIDATES
    # target = target_aqi - aqi_t ; prediction = aqi_t + model output
    # Selection uses TimeSeriesSplit CV on train and a single val check only.
    # =========================================================================
    aqi_train = df_train["aqi"].to_numpy(dtype=float)
    aqi_val = df_val["aqi"].to_numpy(dtype=float)
    time_order = np.argsort(pd.to_datetime(df_train["date"]).to_numpy(), kind="stable")

    # 8. Residual Ridge (alpha selected by TimeSeriesSplit CV on train only)
    t0 = time.time()
    ridge_res = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=4))
    residual_ridge = ResidualRegressor(ridge_res)
    residual_ridge.fit(
        X_train_scaled[time_order],
        y_train_reg[time_order],
        aqi_train[time_order],
    )
    preds_val_res_ridge = residual_ridge.predict(X_val_scaled, aqi_val)
    time_res_ridge = time.time() - t0
    m_res_ridge = calc_regression_metrics(
        y_val_reg, preds_val_res_ridge, persistence_rmse=persist_rmse
    )

    joblib.dump(residual_ridge, args.models_dir / "residual_ridge.joblib")
    saved_model_paths.append(str(args.models_dir / "residual_ridge.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="residual_ridge",
        experiment_id=reg_exp_id,
        tags={
            "task": "regression",
            "model_family": "linear_residual",
            "stage": "candidate",
            "residual": "true",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "ResidualRegressor(RidgeCV)",
                "best_alpha": float(ridge_res.alpha_),
                "cv_folds": 4,
                "residual": True,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_res_ridge, "train_time_sec": round(time_res_ridge, 3)})
        mlflow.log_artifact(str(args.models_dir / "residual_ridge.joblib"))

    leaderboard.append(
        {
            "task": "regression",
            "model": "residual_ridge",
            **m_res_ridge,
            "time_sec": round(time_res_ridge, 2),
        }
    )

    # 9. Residual LightGBM (config selected by TimeSeriesSplit CV on train only)
    lgbm_res_configs: list[dict[str, float | int]] = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 6},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
    ]
    cv_scores = []
    cv_folds_by_config: list[list[float]] = []
    for config in lgbm_res_configs:
        folds = _residual_cv_scores(
            config,
            X_train_df.iloc[time_order],
            y_train_reg[time_order],
            aqi_train[time_order],
            seed=seed,
        )
        cv_folds_by_config.append(folds)
        score = float(np.mean(folds))
        cv_scores.append(score)
        logger.info(f"Residual LightGBM CV RMSE {config}: {score:.3f}")
    best_lgbm_res_idx = int(np.argmin(cv_scores))
    best_lgbm_res_cfg = lgbm_res_configs[best_lgbm_res_idx]
    best_lgbm_res_cv = cv_scores[best_lgbm_res_idx]
    best_lgbm_res_folds = cv_folds_by_config[best_lgbm_res_idx]

    t0 = time.time()
    residual_lgbm = ResidualRegressor(
        LGBMRegressor(**best_lgbm_res_cfg, random_state=seed, verbosity=-1, n_jobs=-1)
    )
    residual_lgbm.fit(X_train_df, y_train_reg, aqi_train)
    preds_val_res_lgbm = residual_lgbm.predict(X_val_df, aqi_val)
    time_res_lgbm = time.time() - t0
    m_res_lgbm = calc_regression_metrics(
        y_val_reg, preds_val_res_lgbm, persistence_rmse=persist_rmse
    )

    joblib.dump(residual_lgbm, args.models_dir / "residual_lightgbm.joblib")
    saved_model_paths.append(str(args.models_dir / "residual_lightgbm.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="residual_lightgbm",
        experiment_id=reg_exp_id,
        tags={
            "task": "regression",
            "model_family": "gradient_boosting_residual",
            "stage": "candidate",
            "residual": "true",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "ResidualRegressor(LGBMRegressor)",
                "n_estimators": int(best_lgbm_res_cfg["n_estimators"]),
                "learning_rate": float(best_lgbm_res_cfg["learning_rate"]),
                "max_depth": int(best_lgbm_res_cfg["max_depth"]),
                "cv_rmse": round(best_lgbm_res_cv, 3),
                "cv_folds": 4,
                "residual": True,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_res_lgbm, "train_time_sec": round(time_res_lgbm, 3)})
        mlflow.log_artifact(str(args.models_dir / "residual_lightgbm.joblib"))

    leaderboard.append(
        {
            "task": "regression",
            "model": "residual_lightgbm",
            **m_res_lgbm,
            "val_max_city_ratio": round(
                _val_max_city_ratio(df_val, preds_val_res_lgbm, preds_val_persist), 4
            ),
            "cv_rmse_mean": round(best_lgbm_res_cv, 3),
            "cv_rmse_std": round(float(np.std(best_lgbm_res_folds)), 3),
            "cv_folds": len(best_lgbm_res_folds),
            "time_sec": round(time_res_lgbm, 2),
        }
    )

    # =========================================================================
    # GATE 3 (ADR D-030 / D-031): monotone + uniformly calibrated candidates
    # Selection is validation-only; the holdout is touched once by
    # scripts/final_test_eval.py after the champion is frozen.
    # =========================================================================
    monotone_cfg = params.get("train", {}).get("monotone", {})
    monotone_enabled = bool(monotone_cfg.get("enabled", True))
    calibration_folds = int(monotone_cfg.get("calibration_folds", 4))
    objectives = list(monotone_cfg.get("objectives", ["l2"]))
    aqi_max = float(df_train["target_aqi"].max())
    train_medians = _train_medians(df_train, feature_names)
    constraints = build_monotone_constraints(feature_names)
    if monotone_enabled:
        logger.info(
            "Monotone constraints: %d features (%s positive, %s negative)",
            sum(1 for value in constraints if value),
            sum(1 for value in constraints if value > 0),
            sum(1 for value in constraints if value < 0),
        )

    monotone_entries: list[dict[str, Any]] = []
    for objective in objectives:
        suffix = "" if objective == "l2" else f"_{objective}"
        name = f"residual_lightgbm_monotone{suffix}"
        t0 = time.time()

        # Reuse the unconstrained family's three pre-specified configurations.
        # The constrained winner is chosen by validation OOF RMSE only; the
        # holdout is not consulted here.
        best: dict[str, Any] | None = None
        for config_index, base_cfg in enumerate(lgbm_res_configs):
            candidate_config: dict[str, Any] = {
                **base_cfg,
                "monotone_constraints": constraints,
                "monotone_constraints_method": str(
                    monotone_cfg.get("constraints_method", "advanced")
                ),
            }
            if objective != "l2":
                candidate_config["objective"] = objective
                if objective == "huber":
                    candidate_config["alpha"] = float(monotone_cfg.get("huber_alpha", 0.9))

            raw_model = ResidualRegressor(
                LGBMRegressor(**candidate_config, random_state=seed, verbosity=-1, n_jobs=-1)
            )
            raw_model.fit(X_train_df, y_train_reg, aqi_train)
            raw_val = raw_model.predict(X_val_df, aqi_val)
            raw_metrics = calc_regression_metrics(y_val_reg, raw_val, persistence_rmse=persist_rmse)

            val_cities = df_val["city"].to_numpy()
            oof_pred, fold_records = forward_calibrated_predictions(
                y_val_reg,
                raw_val,
                val_cities,
                df_val["date"].to_numpy(),
                n_splits=calibration_folds,
            )
            offsets = per_city_offsets(y_val_reg, raw_val, val_cities)
            calibrated_model = CityCalibratedRegressor(base_model=raw_model, offsets=offsets)
            m_calibrated = calc_regression_metrics(
                y_val_reg, oof_pred, persistence_rmse=persist_rmse
            )
            logger.info(
                "Constrained %s config %d (n_estimators=%s, lr=%s, depth=%s): val RMSE %.3f",
                objective,
                config_index,
                base_cfg["n_estimators"],
                base_cfg["learning_rate"],
                base_cfg["max_depth"],
                m_calibrated["rmse"],
            )
            if best is None or m_calibrated["rmse"] < best["metrics"]["rmse"]:
                best = {
                    "config": candidate_config,
                    "config_index": config_index,
                    "model": calibrated_model,
                    "metrics": m_calibrated,
                    "raw_metrics": raw_metrics,
                    "predictions": oof_pred,
                    "folds": fold_records,
                }

        assert best is not None  # lgbm_res_configs is never empty
        calibrated_model = best["model"]
        m_calibrated = best["metrics"]
        raw_metrics = best["raw_metrics"]
        val_ratio = _val_max_city_ratio(df_val, best["predictions"], preds_val_persist)
        metamorphic = _metamorphic_acceptance(
            calibrated_model,
            df_val,
            feature_names,
            params,
            train_medians,
            scaler,
            aqi_max,
        )
        cv_folds_scores = _residual_cv_scores(
            best["config"],
            X_train_df.iloc[time_order],
            y_train_reg[time_order],
            aqi_train[time_order],
            seed=seed,
        )
        relative_change = raw_metrics["rmse"] / m_res_lgbm["rmse"] - 1.0
        calibrated_change = m_calibrated["rmse"] / m_res_lgbm["rmse"] - 1.0
        acceptance = {
            # A2 judges the *constrained model* against the unconstrained
            # incumbent.  The per-city calibration is a separate, uniformly
            # applied A3 step; its OOF cost is reported but is not part of the
            # monotonicity acceptance.
            "passed": bool(relative_change <= 0.02 and metamorphic["passed"]),
            "basis": "uncalibrated RMSE vs uncalibrated champion",
            "val_rmse_relative_change": round(float(relative_change), 4),
            "val_rmse_within_2pct": bool(relative_change <= 0.02),
            "calibrated_val_rmse_relative_change": round(float(calibrated_change), 4),
            "metamorphic_5pct": bool(metamorphic["passed"]),
        }
        time_candidate = time.time() - t0

        joblib.dump(calibrated_model, args.models_dir / f"{name}.joblib")
        saved_model_paths.append(str(args.models_dir / f"{name}.joblib"))

        selected_config = {
            key: value for key, value in best["config"].items() if key != "monotone_constraints"
        }
        with start_run(
            step_id="018",
            logbook_entry="018-gate3-monotone-and-calibration.md",
            run_name=name,
            experiment_id=reg_exp_id,
            tags={
                "task": "regression",
                "model_family": "gradient_boosting_residual_monotone",
                "stage": "gate3_candidate",
                "residual": "true",
                "calibrated": "true",
                "author": "Suhas Raghavendra",
            },
        ):
            mlflow.log_params(
                {
                    "model_type": "CityCalibratedRegressor(ResidualRegressor(LGBMRegressor))",
                    "objective": objective,
                    "monotone_features": int(sum(1 for value in constraints if value)),
                    "calibration_folds": calibration_folds,
                    "n_estimators": int(best["config"]["n_estimators"]),
                    "learning_rate": float(best["config"]["learning_rate"]),
                    "max_depth": int(best["config"]["max_depth"]),
                    "config_index": int(best["config_index"]),
                    "cv_rmse": round(float(np.mean(cv_folds_scores)), 3),
                    "seed": seed,
                }
            )
            mlflow.log_metrics(
                {**m_calibrated, "raw_rmse": raw_metrics["rmse"], "train_time_sec": time_candidate}
            )
            mlflow.log_artifact(str(args.models_dir / f"{name}.joblib"))

        entry = {
            "task": "regression",
            "model": name,
            **m_calibrated,
            "raw_rmse": raw_metrics["rmse"],
            "val_max_city_ratio": round(val_ratio, 4),
            "cv_rmse_mean": round(float(np.mean(cv_folds_scores)), 3),
            "cv_rmse_std": round(float(np.std(cv_folds_scores)), 3),
            "cv_folds": len(cv_folds_scores),
            "config": selected_config,
            "config_index": int(best["config_index"]),
            "calibration": {
                **calibrated_model.calibration_report(),
                "folds": best["folds"],
                "raw_rmse": raw_metrics["rmse"],
                "calibrated_rmse": m_calibrated["rmse"],
            },
            "monotone_constraints": constrained_features(feature_names),
            "metamorphic": metamorphic,
            "pre_registered_acceptance": acceptance,
            "time_sec": round(time_candidate, 2),
        }
        leaderboard.append(entry)
        monotone_entries.append(entry)

    if monotone_entries:
        best_monotone = min(monotone_entries, key=lambda item: float(item["rmse"]))
        logger.info(
            "Best monotone candidate: %s (val RMSE %.3f, accepted=%s)",
            best_monotone["model"],
            best_monotone["rmse"],
            best_monotone["pre_registered_acceptance"]["passed"],
        )

    # Identify best *direct* learned regressor for downstream "regress-then-bucket".
    # Residual models are level candidates and need today's AQI, so they are not
    # used as the base of the categorical wrapper.
    direct_reg_names = {"ridge", "random_forest_reg", "xgboost_reg", "lightgbm_reg"}
    best_reg_entry = min(
        [
            entry
            for entry in leaderboard
            if entry["task"] == "regression" and entry["model"] in direct_reg_names
        ],
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
        tags={
            "task": "classification",
            "model_family": "heuristic",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {"model_type": "MajorityClassBaseline", "majority_class": majority.majority_class_}
        )
        mlflow.log_metrics({**m_maj, "train_time_sec": round(time_maj, 3)})
        mlflow.log_artifact(str(args.models_dir / "majority_class.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": "majority_class",
            **m_maj,
            "time_sec": round(time_maj, 2),
        }
    )

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
        tags={
            "task": "classification",
            "model_family": "hybrid",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {"model_type": "RegressThenBucketClassifier", "base_regressor": best_reg_name}
        )
        mlflow.log_metrics({**m_rb, "train_time_sec": round(time_rb, 3)})
        mlflow.log_artifact(str(args.models_dir / "regress_then_bucket.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": f"regress_then_bucket_{best_reg_name}",
            **m_rb,
            "time_sec": round(time_rb, 2),
        }
    )

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
        tags={
            "task": "classification",
            "model_family": "linear",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_lr, "train_time_sec": round(time_lr, 3)})
        mlflow.log_artifact(str(args.models_dir / "logistic_regression.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": "logistic_regression",
            **m_lr,
            "time_sec": round(time_lr, 2),
        }
    )

    # 4. Random Forest Classifier
    t0 = time.time()
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
    )
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
        tags={
            "task": "classification",
            "model_family": "tree_ensemble",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "class_weight": "balanced_subsample",
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_rfc, "train_time_sec": round(time_rfc, 3)})
        mlflow.log_artifact(str(args.models_dir / "random_forest_clf.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": "random_forest_clf",
            **m_rfc,
            "time_sec": round(time_rfc, 2),
        }
    )

    # 5. LightGBM Classifier
    t0 = time.time()
    lgbm_clf = LGBMClassifier(
        n_estimators=100, class_weight="balanced", max_depth=6, random_state=seed, verbosity=-1
    )
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
        tags={
            "task": "classification",
            "model_family": "gradient_boosting",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "LGBMClassifier",
                "n_estimators": 100,
                "class_weight": "balanced",
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_lgbmc, "train_time_sec": round(time_lgbmc, 3)})
        mlflow.log_artifact(str(args.models_dir / "lightgbm_clf.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": "lightgbm_clf",
            **m_lgbmc,
            "time_sec": round(time_lgbmc, 2),
        }
    )

    # 6. XGBoost Classifier (Step 0.5: was listed in docstring but had no saved model)
    # XGBClassifier requires integer class labels; encode string categories first
    from sklearn.preprocessing import LabelEncoder

    le_xgb = LabelEncoder()
    le_xgb.fit(CATEGORY_ORDER)  # fit on canonical order so indices are stable
    y_train_clf_enc = le_xgb.transform(y_train_clf)
    le_xgb.transform(y_val_clf)

    t0 = time.time()
    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb_clf.fit(X_train_df, y_train_clf_enc)
    preds_val_xgbc_enc = xgb_clf.predict(X_val_df)
    preds_val_xgbc = le_xgb.inverse_transform(preds_val_xgbc_enc)
    time_xgbc = time.time() - t0
    m_xgbc = calc_classification_metrics(y_val_clf, preds_val_xgbc)

    # Save both encoder and model together so evaluate.py can decode
    joblib.dump({"model": xgb_clf, "label_encoder": le_xgb}, args.models_dir / "xgboost_clf.joblib")
    saved_model_paths.append(str(args.models_dir / "xgboost_clf.joblib"))

    with start_run(
        step_id="008",
        logbook_entry="008-baseline-models.md",
        run_name="xgboost_clf",
        experiment_id=clf_exp_id,
        tags={
            "task": "classification",
            "model_family": "gradient_boosting",
            "stage": "baseline",
            "author": "Srijan Kumar G S",
        },
    ):
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "seed": seed,
            }
        )
        mlflow.log_metrics({**m_xgbc, "train_time_sec": round(time_xgbc, 3)})
        mlflow.log_artifact(str(args.models_dir / "xgboost_clf.joblib"))

    leaderboard.append(
        {
            "task": "classification",
            "model": "xgboost_clf",
            **m_xgbc,
            "time_sec": round(time_xgbc, 2),
        }
    )

    # Validation-only champion selection (ADR D-030/D-031). The chosen entry is
    # flagged so model_risk/package_model consume exactly this decision.
    selected = _mark_selected(leaderboard)
    logger.info(
        "Selected champion on validation: %s (RMSE %.3f, max city ratio %.3f)",
        selected["model"],
        selected["rmse"],
        selected.get("val_max_city_ratio", float("nan")),
    )

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
    print(
        reg_df[
            [
                "model",
                "mae",
                "rmse",
                "r2",
                "within_20_rate",
                "skill_score_vs_persistence",
                "time_sec",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("PHASE 6: VALIDATION LEADERBOARD (TASK B: CLASSIFICATION)")
    print("=" * 80)
    clf_df = pd.DataFrame([e for e in leaderboard if e["task"] == "classification"])
    print(
        clf_df[
            ["model", "accuracy", "macro_f1", "weighted_f1", "within_one_category_rate", "time_sec"]
        ].to_string(index=False)
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
