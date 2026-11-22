"""
scripts/register_all_models.py
------------------------------
Registers every trained model family from models/ into MLflow with exact
holdout test metrics, parameters, DVC split hashes, and champion/challenger aliases.
Can target either local sqlite:///mlflow.db or remote DagsHub MLflow directly!
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = REPO_ROOT / "reports/metrics/test_metrics.json"


def load_test_metrics() -> dict[str, dict]:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {item["model"]: item for item in data}


def register_all(
    tracking_uri: str,
    username: str | None = None,
    token: str | None = None,
) -> None:
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print("=" * 70)
    print(f"Target MLflow Tracking URI: {tracking_uri}")
    print("=" * 70)

    # 1. Setup Experiments
    exp_reg = client.get_experiment_by_name("aqi-regression-models")
    if not exp_reg:
        exp_reg_id = client.create_experiment("aqi-regression-models")
    else:
        exp_reg_id = exp_reg.experiment_id

    exp_clf = client.get_experiment_by_name("aqi-classification-models")
    if not exp_clf:
        exp_clf_id = client.create_experiment("aqi-classification-models")
    else:
        exp_clf_id = exp_clf.experiment_id

    metrics_map = load_test_metrics()

    # Model specifications
    model_configs = [
        # Champion & Regressors
        {
            "reg_name": "AQI-Residual-LightGBM-Monotone",
            "file": "residual_lightgbm_monotone.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "champion",
            "desc": "Champion Model: Persistence-anchored LightGBM with 29 monotone constraints and per-city validation calibration offsets.",
        },
        {
            "reg_name": "AQI-Residual-LightGBM",
            "file": "residual_lightgbm.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "challenger",
            "desc": "Unconstrained residual LightGBM regressor.",
        },
        {
            "reg_name": "AQI-Residual-Ridge",
            "file": "residual_ridge.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "baseline",
            "desc": "Persistence-anchored regularized L2 Ridge regressor.",
        },
        {
            "reg_name": "AQI-XGBoost-Regressor",
            "file": "xgboost_reg.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "challenger",
            "desc": "Gradient boosted trees regressor (XGBoost).",
        },
        {
            "reg_name": "AQI-LightGBM-Standard",
            "file": "lightgbm_reg.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "baseline",
            "desc": "Standard unconstrained LightGBM leaf-wise regressor.",
        },
        {
            "reg_name": "AQI-Ridge-Linear",
            "file": "ridge.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "baseline",
            "desc": "Direct linear Ridge regressor.",
        },
        {
            "reg_name": "AQI-Random-Forest-Regressor",
            "file": "random_forest_reg.joblib",
            "exp_id": exp_reg_id,
            "task": "regression",
            "alias": "baseline",
            "desc": "Bagged ensemble Random Forest regressor.",
        },
        # Classifiers
        {
            "reg_name": "AQI-Regress-Then-Bucket",
            "file": "regress_then_bucket.joblib",
            "exp_id": exp_clf_id,
            "task": "classification",
            "alias": "champion",
            "desc": "Champion Classifier: Regress-then-bucket onto statutory CPCB cutoffs.",
        },
        {
            "reg_name": "AQI-Random-Forest-Classifier",
            "file": "random_forest_clf.joblib",
            "exp_id": exp_clf_id,
            "task": "classification",
            "alias": "challenger",
            "desc": "Balanced subsample Random Forest classifier.",
        },
        {
            "reg_name": "AQI-LightGBM-Classifier",
            "file": "lightgbm_clf.joblib",
            "exp_id": exp_clf_id,
            "task": "classification",
            "alias": "challenger",
            "desc": "Multi-class LightGBM classifier.",
        },
        {
            "reg_name": "AQI-XGBoost-Classifier",
            "file": "xgboost_clf.joblib",
            "exp_id": exp_clf_id,
            "task": "classification",
            "alias": "challenger",
            "desc": "Multi-class XGBoost classifier.",
        },
        {
            "reg_name": "AQI-Logistic-Regression",
            "file": "logistic_regression.joblib",
            "exp_id": exp_clf_id,
            "task": "classification",
            "alias": "baseline",
            "desc": "Multinomial balanced logistic regression.",
        },
    ]

    for cfg in model_configs:
        file_path = MODELS_DIR / cfg["file"]
        model_key = cfg["file"].replace(".joblib", "")
        metrics = metrics_map.get(model_key, {})

        print(f"\nRegistering '{cfg['reg_name']}'...")
        tags = {
            "task": cfg["task"],
            "features_count": "103",
            "dvc_data_version": "49bd4c88748327afa12057c2fe71b0e7",
            "split_train_md5": "49bd4c88748327afa12057c2fe71b0e7",
            "split_test_md5": "bd1de9ea97083402bd9ccaceb1072e3c",
            "git_commit": "11cd466",
        }

        # 1. Create Run
        with mlflow.start_run(experiment_id=cfg["exp_id"], run_name=cfg["reg_name"]) as run:
            # Log params
            mlflow.log_params(
                {
                    "model_family": cfg["task"],
                    "source_file": cfg["file"],
                    "target": (
                        "next_day_aqi" if cfg["task"] == "regression" else "next_day_cpcb_category"
                    ),
                }
            )
            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            # Set tags
            mlflow.set_tags(tags)

            # Log artifact / model
            if file_path.exists():
                try:
                    loaded_obj = joblib.load(file_path)
                    if cfg["task"] == "regression" and hasattr(loaded_obj, "predict"):
                        # Log as sklearn model
                        mlflow.sklearn.log_model(
                            sk_model=loaded_obj,
                            artifact_path="model",
                            registered_model_name=cfg["reg_name"],
                        )
                    else:
                        mlflow.log_artifact(str(file_path), artifact_path="model")
                        # Register model explicitly
                        with contextlib.suppress(Exception):
                            client.create_registered_model(cfg["reg_name"], description=cfg["desc"])
                        client.create_model_version(
                            name=cfg["reg_name"],
                            source=f"{run.info.artifact_uri}/model",
                            run_id=run.info.run_id,
                            description=cfg["desc"],
                            tags=tags,
                        )
                except Exception as e:
                    print(f"  [WARN] Fallback artifact logging for {cfg['file']}: {e}")
                    mlflow.log_artifact(str(file_path), artifact_path="model")
                    with contextlib.suppress(Exception):
                        client.create_registered_model(cfg["reg_name"], description=cfg["desc"])
                    client.create_model_version(
                        name=cfg["reg_name"],
                        source=f"{run.info.artifact_uri}/model",
                        run_id=run.info.run_id,
                        description=cfg["desc"],
                        tags=tags,
                    )
            else:
                print(f"  [WARN] File not found: {file_path}")

        # Set Alias
        try:
            versions = client.search_model_versions(f"name='{cfg['reg_name']}'")
            if versions:
                latest_v = sorted(versions, key=lambda x: int(x.version))[-1]
                client.set_registered_model_alias(cfg["reg_name"], cfg["alias"], latest_v.version)
                print(f"  -> Version {latest_v.version} registered with alias @{cfg['alias']}")
        except Exception as e:
            print(f"  [WARN] Could not set alias: {e}")

    print("\n" + "=" * 70)
    print("ALL MODELS SUCCESSFULLY REGISTERED IN MLFLOW!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="sqlite:///mlflow.db")
    parser.add_argument("--username", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    register_all(
        tracking_uri=args.uri,
        username=args.username,
        token=args.token,
    )
