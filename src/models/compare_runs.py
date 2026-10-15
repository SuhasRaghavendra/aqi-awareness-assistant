"""
src/models/compare_runs.py
--------------------------
Phase 7: Query MLflow tracking database (sqlite:///mlflow.db)
and export a comprehensive run comparison table to reports/metrics/run_comparison.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Compare MLflow Runs")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/metrics/run_comparison.csv"))
    args = parser.parse_args()

    setup_logging(stage="compare_runs")
    logger.info(f"Connecting to MLflow at {args.tracking_uri}...")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    logger.info(f"Discovered {len(experiments)} MLflow experiments.")

    all_runs = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            data = run.data
            row = {
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
                "step_id": data.tags.get("step_id", ""),
                "task": data.tags.get("task", ""),
                "model_family": data.tags.get("model_family", ""),
                "model_type": data.params.get("model_type", ""),
                # Regression metrics
                "val_mae": data.metrics.get("mae", None),
                "val_rmse": data.metrics.get("rmse", None),
                "val_r2": data.metrics.get("r2", None),
                "val_within_20_rate": data.metrics.get("within_20_rate", None),
                "val_skill_score": data.metrics.get("skill_score_vs_persistence", None),
                # Classification metrics
                "val_accuracy": data.metrics.get("accuracy", None),
                "val_macro_f1": data.metrics.get("macro_f1", None),
                "val_within_one_category": data.metrics.get("within_one_category_rate", None),
                "train_time_sec": data.metrics.get("train_time_sec", None),
            }
            all_runs.append(row)

    if not all_runs:
        logger.warning("No MLflow runs found.")
        return

    df_runs = pd.DataFrame(all_runs)
    df_runs = df_runs.sort_values(by=["experiment_name", "val_rmse", "val_macro_f1"], ascending=[True, True, False])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_runs.to_csv(args.output_csv, index=False)
    logger.info(f"Exported {len(df_runs)} runs to {args.output_csv}")

    print("\n" + "=" * 90)
    print("MLFLOW RUN COMPARISON TABLE")
    print("=" * 90)
    cols_to_print = ["experiment_name", "run_name", "model_family", "val_rmse", "val_r2", "val_accuracy", "val_macro_f1", "train_time_sec"]
    print(df_runs[cols_to_print].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
