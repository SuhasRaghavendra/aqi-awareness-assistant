"""
scripts/log_versioned_experiments.py
------------------------------------
Option B: Non-destructively creates 3 genuine data versions and 4 registered
models x 3 versions each (12 real trained models evaluated on val.parquet) and
logs/registers them in MLflow (hosted DagsHub MLflow and/or local SQLite).

Frozen artifacts (data/processed/*.parquet, params.yaml, artifacts/champion/)
are NEVER modified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib  # noqa: E402

# ADR D-015: import lightgbm before pandas/pyarrow on Windows
import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.monotone import build_monotone_constraints  # noqa: E402

DATA_EXPERIMENT = "aqi-data-versions"
MODELS_EXPERIMENT = "aqi-versioned-models"
DAGSHUB_URI = "https://dagshub.com/SuhasRaghavendra/aqi-awareness-assistant.mlflow"


def _load_dotenv_if_available() -> None:
    """Load .env into os.environ without printing or inspecting its contents."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_md5(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.md5()
    arr = np.ascontiguousarray(df[cols].to_numpy(dtype=np.float64))
    h.update(arr.tobytes())
    return h.hexdigest()


def build_data_versions(
    train_df: pd.DataFrame,
    feature_names: list[str],
    train_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build 3 non-destructive training data versions in memory."""
    # Version 1: Frozen baseline split
    v1_df = train_df.copy()
    v1_md5 = _file_md5(train_path)

    # Version 2: Tighter [1%, 99%] winsorization on continuous features
    v2_df = train_df.copy()
    clipped_cols = 0
    for col in feature_names:
        if v2_df[col].nunique() > 10:
            lo, hi = v2_df[col].quantile([0.01, 0.99])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                v2_df[col] = v2_df[col].clip(lower=lo, upper=hi)
                clipped_cols += 1
    v2_md5 = _frame_md5(v2_df, feature_names)

    # Version 3: Recent training window (2020-2023, dropping sparse 2019 rows)
    dates = pd.to_datetime(train_df["date"])
    v3_df = train_df.loc[dates >= "2020-01-01"].reset_index(drop=True)
    v3_md5 = _frame_md5(v3_df, feature_names)

    return {
        "data-v1-baseline": {
            "df": v1_df,
            "md5": v1_md5,
            "description": "Frozen chronological split (2019-2023, winsor [0.5%, 99.5%], 103 features)",
            "winsor_bounds": "0.005-0.995",
            "train_window": "2019-01-01..2023-12-31",
            "extra": {"clipped_continuous_features": 0},
        },
        "data-v2-tight-winsor": {
            "df": v2_df,
            "md5": v2_md5,
            "description": "Tighter outlier clipping ([1.0%, 99.0%] winsorization on training features)",
            "winsor_bounds": "0.010-0.990",
            "train_window": "2019-01-01..2023-12-31",
            "extra": {"clipped_continuous_features": clipped_cols},
        },
        "data-v3-recent-2020plus": {
            "df": v3_df,
            "md5": v3_md5,
            "description": "Recent training window (2020-2023, dropping sparse 2019 station rows)",
            "winsor_bounds": "0.005-0.995",
            "train_window": "2020-01-01..2023-12-31",
            "extra": {"rows_dropped_2019": int(len(v1_df) - len(v3_df))},
        },
    }


def eval_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persist_rmse: float,
) -> dict[str, float]:
    y_pred_clipped = np.clip(y_pred, 0.0, 500.0)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true, y_pred_clipped))
    r2 = float(r2_score(y_true, y_pred_clipped))
    skill = float(1.0 - rmse / persist_rmse) if persist_rmse > 0 else 0.0
    return {
        "val_mae": round(mae, 4),
        "val_rmse": round(rmse, 4),
        "val_r2": round(r2, 4),
        "skill_score_vs_persistence": round(skill, 4),
    }


def _train_variant(
    family: str,
    hyperparams: dict[str, Any],
    x_train: np.ndarray,
    y_train_res: np.ndarray,
    y_train_abs: np.ndarray,
    x_val: np.ndarray,
    val_aqi_t: np.ndarray,
    monotone_vec: list[int],
) -> tuple[Any, np.ndarray]:
    """Train one real model variant and return (fitted_model, val_predictions)."""
    if family in ("Indian-AQI-Predictor", "AQI-Residual-LightGBM-Monotone"):
        model = lgb.LGBMRegressor(
            n_estimators=int(hyperparams["n_estimators"]),
            learning_rate=float(hyperparams["learning_rate"]),
            max_depth=int(hyperparams["max_depth"]),
            num_leaves=int(hyperparams.get("num_leaves", 31)),
            monotone_constraints=monotone_vec,
            monotone_constraints_method="advanced",
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(x_train, y_train_res)
        preds = val_aqi_t + model.predict(x_val)
        return model, preds

    if family == "AQI-XGBoost-Regressor":
        model = XGBRegressor(
            n_estimators=int(hyperparams["n_estimators"]),
            learning_rate=float(hyperparams["learning_rate"]),
            max_depth=int(hyperparams["max_depth"]),
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )
        model.fit(x_train, y_train_abs)
        preds = model.predict(x_val)
        return model, preds

    if family == "AQI-Residual-Ridge":
        model = Ridge(alpha=float(hyperparams["alpha"]))
        model.fit(x_train, y_train_res)
        preds = val_aqi_t + model.predict(x_val)
        return model, preds

    raise ValueError(f"Unknown model family: {family}")


def run_for_uri(
    tracking_uri: str,
    data_versions: dict[str, dict[str, Any]],
    val_df: pd.DataFrame,
    feature_names: list[str],
    scaler: Any,
    monotone_vec: list[int],
    git_sha: str,
) -> None:
    """Log all 3 data versions and 4 models x 3 versions to the target MLflow URI."""
    print("=" * 72)
    print(f"Publishing Option B versions to MLflow URI: {tracking_uri}")
    print("=" * 72)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Ensure experiments exist
    def _ensure_exp(name: str) -> str:
        exp = client.get_experiment_by_name(name)
        return exp.experiment_id if exp else client.create_experiment(name)

    data_exp_id = _ensure_exp(DATA_EXPERIMENT)
    models_exp_id = _ensure_exp(MODELS_EXPERIMENT)

    # 2. Log 3 Data Versions
    print(f"\n[1/2] Logging 3 data versions to '{DATA_EXPERIMENT}'...")
    for dv_name, dv_info in data_versions.items():
        df = dv_info["df"]
        with mlflow.start_run(
            experiment_id=data_exp_id,
            run_name=dv_name,
            tags={
                "dvc_data_version": dv_name,
                "data_md5": dv_info["md5"],
                "features_count": str(len(feature_names)),
                "git_commit": git_sha,
            },
        ):
            mlflow.log_params(
                {
                    "data_version": dv_name,
                    "data_md5": dv_info["md5"],
                    "winsor_bounds": dv_info["winsor_bounds"],
                    "train_window": dv_info["train_window"],
                    "rows": len(df),
                    "cities": int(df["city"].nunique()),
                    "features_count": len(feature_names),
                }
            )
            mlflow.log_metrics(
                {
                    "train_rows": float(len(df)),
                    "aqi_mean": round(float(df["aqi"].mean()), 3),
                    "aqi_std": round(float(df["aqi"].std()), 3),
                    "aqi_p95": round(float(df["aqi"].quantile(0.95)), 3),
                }
            )
            summary = {
                "data_version": dv_name,
                "md5": dv_info["md5"],
                "description": dv_info["description"],
                "rows": len(df),
                "cities": int(df["city"].nunique()),
                "date_min": str(pd.to_datetime(df["date"]).min().date()),
                "date_max": str(pd.to_datetime(df["date"]).max().date()),
                **dv_info["extra"],
            }
            mlflow.log_dict(summary, "data_version.json")
        print(f"  [OK] Logged data version '{dv_name}' (rows={len(df)}, md5={dv_info['md5'][:12]})")

    # 3. Precompute scaled arrays for all 3 data versions + validation split
    x_val = scaler.transform(val_df[feature_names].to_numpy(dtype=np.float64))
    y_val = val_df["target_aqi"].to_numpy(dtype=np.float64)
    val_aqi_t = val_df["aqi"].to_numpy(dtype=np.float64)
    persist_rmse = float(root_mean_squared_error(y_val, val_aqi_t))

    scaled_train: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dv_name, dv_info in data_versions.items():
        df = dv_info["df"]
        x_tr = scaler.transform(df[feature_names].to_numpy(dtype=np.float64))
        y_abs = df["target_aqi"].to_numpy(dtype=np.float64)
        y_res = y_abs - df["aqi"].to_numpy(dtype=np.float64)
        scaled_train[dv_name] = (x_tr, y_res, y_abs)

    # 4. Define 4 Registered Models x 3 Versions Each = 12 Real Versions
    specs: list[dict[str, Any]] = [
        # --- Model 1: Indian-AQI-Predictor ---
        {
            "model": "Indian-AQI-Predictor",
            "variant": "v1-monotone-baseline",
            "data_version": "data-v1-baseline",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 6,
                "num_leaves": 31,
            },
            "alias": "champion",
            "desc": "Production Champion: Monotone residual LightGBM on frozen data-v1-baseline.",
        },
        {
            "model": "Indian-AQI-Predictor",
            "variant": "v2-monotone-lr005-winsor1pct",
            "data_version": "data-v2-tight-winsor",
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
            },
            "alias": "challenger",
            "desc": "Challenger: Faster LR (0.05) + 250 trees trained on tighter winsorized data-v2.",
        },
        {
            "model": "Indian-AQI-Predictor",
            "variant": "v3-monotone-depth8-recent2020",
            "data_version": "data-v3-recent-2020plus",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 8,
                "num_leaves": 45,
            },
            "alias": "candidate",
            "desc": "Candidate: Deeper monotone trees (depth=8) retrained on recent 2020-2023 window.",
        },
        # --- Model 2: AQI-Residual-LightGBM-Monotone ---
        {
            "model": "AQI-Residual-LightGBM-Monotone",
            "variant": "v1-baseline-lr003-d6",
            "data_version": "data-v1-baseline",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 6,
                "num_leaves": 31,
            },
            "alias": "champion",
            "desc": "Monotone Residual LightGBM v1 on data-v1-baseline.",
        },
        {
            "model": "AQI-Residual-LightGBM-Monotone",
            "variant": "v2-winsor1pct-lr005-d6",
            "data_version": "data-v2-tight-winsor",
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
            },
            "alias": "challenger",
            "desc": "Monotone Residual LightGBM v2 (lr=0.05, n=250) on data-v2-tight-winsor.",
        },
        {
            "model": "AQI-Residual-LightGBM-Monotone",
            "variant": "v3-recent2020-lr003-d8",
            "data_version": "data-v3-recent-2020plus",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 8,
                "num_leaves": 45,
            },
            "alias": "candidate",
            "desc": "Monotone Residual LightGBM v3 (depth=8) on data-v3-recent-2020plus.",
        },
        # --- Model 3: AQI-XGBoost-Regressor ---
        {
            "model": "AQI-XGBoost-Regressor",
            "variant": "v1-baseline-lr003-d6",
            "data_version": "data-v1-baseline",
            "params": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 6},
            "alias": "baseline",
            "desc": "XGBoost Regressor v1 on data-v1-baseline.",
        },
        {
            "model": "AQI-XGBoost-Regressor",
            "variant": "v2-winsor1pct-lr005-d5",
            "data_version": "data-v2-tight-winsor",
            "params": {"n_estimators": 220, "learning_rate": 0.05, "max_depth": 5},
            "alias": "challenger",
            "desc": "XGBoost Regressor v2 (lr=0.05, depth=5) on data-v2-tight-winsor.",
        },
        {
            "model": "AQI-XGBoost-Regressor",
            "variant": "v3-recent2020-lr003-d6",
            "data_version": "data-v3-recent-2020plus",
            "params": {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 6},
            "alias": "candidate",
            "desc": "XGBoost Regressor v3 (n=250) on data-v3-recent-2020plus.",
        },
        # --- Model 4: AQI-Residual-Ridge ---
        {
            "model": "AQI-Residual-Ridge",
            "variant": "v1-baseline-alpha10",
            "data_version": "data-v1-baseline",
            "params": {"alpha": 10.0},
            "alias": "baseline",
            "desc": "Residual Ridge v1 (alpha=10) on data-v1-baseline.",
        },
        {
            "model": "AQI-Residual-Ridge",
            "variant": "v2-winsor1pct-alpha25",
            "data_version": "data-v2-tight-winsor",
            "params": {"alpha": 25.0},
            "alias": "challenger",
            "desc": "Residual Ridge v2 (alpha=25) on data-v2-tight-winsor.",
        },
        {
            "model": "AQI-Residual-Ridge",
            "variant": "v3-recent2020-alpha50",
            "data_version": "data-v3-recent-2020plus",
            "params": {"alpha": 50.0},
            "alias": "candidate",
            "desc": "Residual Ridge v3 (alpha=50) on data-v3-recent-2020plus.",
        },
    ]

    print(f"\n[2/2] Training & registering 4 models x 3 versions into '{MODELS_EXPERIMENT}'...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for spec in specs:
            model_name = spec["model"]
            dv_name = spec["data_version"]
            dv_md5 = data_versions[dv_name]["md5"]
            x_tr, y_res, y_abs = scaled_train[dv_name]

            fitted_model, val_preds = _train_variant(
                family=model_name,
                hyperparams=spec["params"],
                x_train=x_tr,
                y_train_res=y_res,
                y_train_abs=y_abs,
                x_val=x_val,
                val_aqi_t=val_aqi_t,
                monotone_vec=monotone_vec,
            )
            metrics = eval_predictions(y_val, val_preds, persist_rmse)
            tags = {
                "model_variant": spec["variant"],
                "dvc_data_version": dv_name,
                "data_md5": dv_md5,
                "features_count": str(len(feature_names)),
                "git_commit": git_sha,
                "target": "next_day_aqi",
            }

            with mlflow.start_run(
                experiment_id=models_exp_id,
                run_name=f"{model_name}__{spec['variant']}",
                tags=tags,
            ) as run:
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "variant": spec["variant"],
                        "data_version": dv_name,
                        "data_md5": dv_md5,
                        **spec["params"],
                    }
                )
                mlflow.log_metrics(metrics)

                artifact_file = tmp_path / f"{model_name}_{spec['variant']}.joblib"
                joblib.dump(fitted_model, artifact_file)
                mlflow.log_artifact(str(artifact_file), artifact_path="model")

                with contextlib.suppress(Exception):
                    client.create_registered_model(model_name, description=spec["desc"])

                mv = client.create_model_version(
                    name=model_name,
                    source=f"{run.info.artifact_uri}/model",
                    run_id=run.info.run_id,
                    description=spec["desc"],
                    tags=tags,
                )
                with contextlib.suppress(Exception):
                    client.set_registered_model_alias(model_name, spec["alias"], mv.version)

            print(
                f"  [OK] {model_name} v{mv.version} (@{spec['alias']}) | {dv_name} | "
                f"MAE={metrics['val_mae']:.3f} RMSE={metrics['val_rmse']:.3f} "
                f"R2={metrics['val_r2']:.4f} Skill={metrics['skill_score_vs_persistence']:+.4f}"
            )

    print("\nDone publishing to:", tracking_uri)


def main() -> None:
    _load_dotenv_if_available()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DAGSHUB_URI),
        help="Primary MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or DagsHub).",
    )
    parser.add_argument(
        "--also-local",
        action="store_true",
        help="Also mirror the versioned runs to local sqlite:///mlflow.db.",
    )
    args = parser.parse_args()

    train_path = REPO_ROOT / "data/processed/train.parquet"
    val_path = REPO_ROOT / "data/processed/val.parquet"
    feat_path = REPO_ROOT / "data/processed/feature_list.json"
    scaler_path = REPO_ROOT / "data/processed/scaler.joblib"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    feat_meta = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_names: list[str] = (
        feat_meta["feature_names"] if isinstance(feat_meta, dict) else list(feat_meta)
    )
    scaler = joblib.load(scaler_path)
    monotone_vec = build_monotone_constraints(feature_names)
    git_sha = _git_sha()

    data_versions = build_data_versions(train_df, feature_names, train_path)

    run_for_uri(
        tracking_uri=args.uri,
        data_versions=data_versions,
        val_df=val_df,
        feature_names=feature_names,
        scaler=scaler,
        monotone_vec=monotone_vec,
        git_sha=git_sha,
    )

    if args.also_local and args.uri != "sqlite:///mlflow.db":
        run_for_uri(
            tracking_uri="sqlite:///mlflow.db",
            data_versions=data_versions,
            val_df=val_df,
            feature_names=feature_names,
            scaler=scaler,
            monotone_vec=monotone_vec,
            git_sha=git_sha,
        )


if __name__ == "__main__":
    main()
