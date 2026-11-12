"""
src/models/model_risk.py
------------------------
Phase 1: Model testing and risk/quality audit.

Builds a champion artifact view from the validation leaderboard, evaluates the
Phase 1 quality gates on the held-out test split, and writes:

  - reports/metrics/reference_stats.json
  - reports/metrics/model_risk_card.json

The module is intentionally importable so the Phase 1 test-suite can reuse the
same feature-rebuild, prediction, gate, and bootstrap logic. Missing artifacts
raise ``FileNotFoundError`` (tests must fail, never skip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm  # noqa: F401  # import before pandas/pyarrow on Windows (ADR D-015)
import numpy as np
import pandas as pd

from src.config import load_params
from src.features.serving_features import build_feature_frame
from src.utils.io import atomic_write_json
from src.utils.reproducibility import frame_timestamp

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
POOR_PLUS = {"Poor", "Very Poor", "Severe"}

CHAMPION_EXCLUDED = {
    "persistence",
    "seasonal_naive",
    "station_month_mean",
    "random_forest_reg",
}

BASELINE_MODELS = {"persistence", "seasonal_naive", "station_month_mean"}

GATE_THRESHOLDS = {
    "G1": "skill vs persistence > 0 on val and test (RMSE basis)",
    "G2": "test R2 >= 0.65 and test MAE <= 30",
    "G3": "model MAE <= persistence MAE * 1.05 in every city",
    "G4": "within-one-category rate >= 0.95; Poor+ recall reported per city",
    "G5": "p95 single-row latency < 50 ms on the selected artifact",
}

RANGE_TOLERANCE_RELATIVE = 0.01


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    """Read JSON, raising if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def file_md5(path: Path) -> str:
    """Return the MD5 digest of a file."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str:
    """Return the current git commit SHA, or ``unknown`` outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or repo_root()),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the feature manifest (never hard-code the feature count)."""
    return read_json(path or repo_root() / "data/processed/feature_list.json")


def load_scaler(path: Path | None = None) -> Any:
    """Load the single train-fitted scaler."""
    scaler_path = path or repo_root() / "data/processed/scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {scaler_path}")
    return joblib.load(scaler_path)


def build_reference_stats(
    train_df: pd.DataFrame,
    feature_names: list[str],
    params: dict[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    """Compute train-split reference statistics used for range checks."""
    features: dict[str, dict[str, float]] = {}
    for name in feature_names:
        series = train_df[name].astype(float)
        features[name] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    medians = train_df[feature_names].median().fillna(0.0)
    target = train_df["target_aqi"].astype(float)
    split = params.get("split", {})
    return {
        "schema_version": "1.0",
        "generated_at": frame_timestamp(train_df),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "train_rows": int(len(train_df)),
        "train_medians": {name: float(medians[name]) for name in feature_names},
        "target_aqi": {
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "std": float(target.std(ddof=0)),
        },
        "features": features,
        "tolerance": {
            "relative": RANGE_TOLERANCE_RELATIVE,
            "description": (
                "Values outside train min/max expanded by 1% of the train range are "
                "flagged in the risk card; flags are observational and never raise."
            ),
        },
        "split": {
            "train_end": split.get("train_end"),
            "val_end": split.get("val_end"),
        },
    }


def flag_out_of_range(
    df: pd.DataFrame,
    reference_stats: dict[str, Any],
    feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Flag feature values outside the train range plus a documented tolerance."""
    flags: dict[str, dict[str, Any]] = {}
    for name in feature_names:
        ref = reference_stats["features"][name]
        lo, hi = float(ref["min"]), float(ref["max"])
        span = max(hi - lo, 1e-9)
        tol = RANGE_TOLERANCE_RELATIVE * span
        values = df[name].astype(float)
        below = int((values < lo - tol).sum())
        above = int((values > hi + tol).sum())
        if below or above:
            flags[name] = {
                "below": below,
                "above": above,
                "train_min": lo,
                "train_max": hi,
                "tolerance": tol,
            }
    return flags


def rebuild_features(
    panel: pd.DataFrame,
    feature_names: list[str],
    params: dict[str, Any],
    train_medians: dict[str, float],
) -> pd.DataFrame:
    """Rebuild the Phase 0 feature set from a raw city-day panel slice.

    Kept as a thin compatibility wrapper; the implementation is shared with the
    online service via ``src.features.serving_features``.
    """
    return build_feature_frame(panel, feature_names, params, train_medians)


def select_champion(val_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the champion recorded by the validation-only selection rule.

    ``train.py`` marks exactly one eligible entry with ``selected: true`` after
    applying the pre-registered criteria (D-030/D-031).  If no entry carries the
    flag (older leaderboards), fall back to the lowest validation RMSE.
    """
    candidates = [
        entry
        for entry in val_metrics
        if entry.get("task") == "regression" and entry.get("model") not in CHAMPION_EXCLUDED
    ]
    if not candidates:
        raise ValueError("No eligible regression candidates found in val_metrics.json")
    selected = [entry for entry in candidates if entry.get("selected")]
    if len(selected) == 1:
        return selected[0]
    return min(candidates, key=lambda entry: float(entry["rmse"]))


@dataclass(frozen=True)
class ChampionArtifact:
    """Selected champion model plus its artifact path and validation scores."""

    name: str
    model: Any
    path: Path
    val_metrics: dict[str, Any]


def load_champion_artifact(root: Path | None = None) -> ChampionArtifact:
    """Load the champion artifact, raising when it is missing (never skip)."""
    base = root or repo_root()
    val_metrics = read_json(base / "reports/metrics/val_metrics.json")
    selected = select_champion(val_metrics)
    model_path = base / "models" / f"{selected['model']}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Champion artifact is missing: {model_path}")
    return ChampionArtifact(
        name=str(selected["model"]),
        model=joblib.load(model_path),
        path=model_path,
        val_metrics=selected,
    )


def predict_regression(
    name: str,
    model: Any,
    df: pd.DataFrame,
    scaler: Any,
    feature_names: list[str],
    aqi_max: float,
) -> np.ndarray:
    """Predict and clip regression output to the training AQI scale.

    Calibrated champions receive ``city`` so their per-city offsets are
    applied identically offline and in serving.
    """
    from src.models.calibration import CityCalibratedRegressor

    scaled = scaler.transform(df[feature_names])
    scaled_df = pd.DataFrame(scaled, columns=feature_names)
    if isinstance(model, CityCalibratedRegressor):
        preds = model.predict(
            scaled_df,
            df["aqi"].to_numpy(dtype=float),
            city=df["city"].to_numpy(),
        )
    elif name.startswith("residual_"):
        preds = model.predict(scaled_df, df["aqi"].to_numpy(dtype=float))
    elif name == "ridge":
        preds = model.predict(scaled)
    elif name in BASELINE_MODELS:
        preds = model.predict(df)
    else:
        preds = model.predict(scaled_df)
    return np.clip(np.asarray(preds, dtype=float), 0.0, float(aqi_max))


def select_classifier(validation_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the classifier on validation: best within-one rate, then macro F1.

    Test metrics are never used to choose the classifier; the selected artifact
    is only evaluated on the holdout test split afterwards (gate G4).
    """
    candidates = [
        entry
        for entry in validation_metrics
        if entry.get("task") == "classification" and entry.get("model") != "majority_class"
    ]
    if not candidates:
        raise ValueError("No classification candidates found in reports/metrics/val_metrics.json")
    return max(
        candidates,
        key=lambda entry: (
            float(entry.get("within_one_category_rate", 0.0)),
            float(entry.get("macro_f1", 0.0)),
        ),
    )


def resolve_classifier_artifact(base: Path, leaderboard_name: str) -> tuple[str, Path]:
    """Map a validation leaderboard entry to its trained classifier artifact.

    ``train.py`` records the hybrid model as ``regress_then_bucket_<regressor>``
    while the artifact is always ``models/regress_then_bucket.joblib``.
    """
    candidates = [leaderboard_name]
    if leaderboard_name.startswith("regress_then_bucket_"):
        candidates.append("regress_then_bucket")
    for name in candidates:
        path = base / "models" / f"{name}.joblib"
        if path.exists():
            return name, path
    raise FileNotFoundError(
        f"Classifier artifact for {leaderboard_name!r} is missing under {base / 'models'}"
    )


def predict_classifier(
    name: str,
    model: Any,
    df: pd.DataFrame,
    scaler: Any,
    feature_names: list[str],
) -> np.ndarray:
    """Predict AQI categories for the supported classifier artifact formats."""
    scaled = scaler.transform(df[feature_names])
    scaled_df = pd.DataFrame(scaled, columns=feature_names)
    if isinstance(model, dict) and {"model", "label_encoder"} <= set(model):
        raw = model["model"].predict(scaled_df)
        return np.asarray(model["label_encoder"].inverse_transform(raw))
    if name == "logistic_regression":
        return np.asarray(model.predict(scaled))
    return np.asarray(model.predict(scaled_df))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics shared by gates and per-city reporting."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def bootstrap_skill_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    dates: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on skill vs persistence, resampling whole (year, week) blocks."""
    rng = np.random.default_rng(seed)
    stamps = pd.to_datetime(dates).isocalendar()
    weeks = stamps["week"].to_numpy() + stamps["year"].to_numpy() * 100
    unique_weeks = np.unique(weeks)
    skills: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
        idx = np.concatenate([np.where(weeks == week)[0] for week in sampled])
        rmse_model = float(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
        rmse_persist = float(np.sqrt(np.mean((y_true[idx] - y_persist[idx]) ** 2)))
        skills.append(1.0 - rmse_model / rmse_persist if rmse_persist > 0 else 0.0)
    alpha = 0.025
    return float(np.quantile(skills, alpha)), float(np.quantile(skills, 1 - alpha))


def skill_vs_persistence(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray) -> float:
    """Compute RMSE skill relative to persistence."""
    rmse_model = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse_persist = float(np.sqrt(np.mean((y_true - y_persist) ** 2)))
    return 1.0 - rmse_model / rmse_persist if rmse_persist > 0 else 0.0


def per_city_records(
    df: pd.DataFrame,
    pred: np.ndarray,
    persist: np.ndarray,
    categories: np.ndarray | None = None,
    category_pred: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build per-city MAE/RMSE/skill/CI rows, optionally with Poor+ recall."""
    frame = df[["city", "date", "target_aqi"]].copy()
    frame["pred"] = pred
    frame["persist"] = persist
    if categories is not None:
        frame["category"] = categories
    if category_pred is not None:
        frame["category_pred"] = category_pred
    frame["date"] = pd.to_datetime(frame["date"])

    records: list[dict[str, Any]] = []
    for city, group in frame.groupby("city"):
        y_true = group["target_aqi"].to_numpy(dtype=float)
        y_pred = group["pred"].to_numpy(dtype=float)
        y_persist = group["persist"].to_numpy(dtype=float)
        metrics = regression_metrics(y_true, y_pred)
        persist_metrics = regression_metrics(y_true, y_persist)
        ci_lo, ci_hi = bootstrap_skill_ci(y_true, y_pred, y_persist, group["date"].to_numpy())
        record: dict[str, Any] = {
            "city": city,
            "n_rows": int(len(group)),
            "mae": round(metrics["mae"], 3),
            "rmse": round(metrics["rmse"], 3),
            "mae_persistence": round(persist_metrics["mae"], 3),
            "rmse_persistence": round(persist_metrics["rmse"], 3),
            "mae_ratio_vs_persistence": round(
                metrics["mae"] / persist_metrics["mae"] if persist_metrics["mae"] > 0 else 0.0,
                4,
            ),
            "skill_vs_persistence": round(skill_vs_persistence(y_true, y_pred, y_persist), 4),
            "skill_ci_lo_95": round(ci_lo, 4),
            "skill_ci_hi_95": round(ci_hi, 4),
        }
        if "category" in group and "category_pred" in group:
            true_cat = group["category"].to_numpy()
            pred_cat = group["category_pred"].to_numpy()
            mask = np.isin(true_cat, list(POOR_PLUS))
            support = int(mask.sum())
            record["poor_plus_support"] = support
            record["poor_plus_recall"] = (
                round(float(np.mean(pred_cat[mask] == true_cat[mask])), 4) if support else None
            )
        records.append(record)
    return records


def measure_latency(
    predict_one: Any,
    iterations: int = 200,
    warmup: int = 10,
) -> dict[str, float]:
    """Measure single-row prediction latency percentiles in milliseconds."""
    for _ in range(warmup):
        predict_one()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        predict_one()
        samples.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(samples, dtype=float)
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "p99_ms": round(float(np.percentile(values, 99)), 4),
        "iterations": int(iterations),
    }


def build_risk_card(
    root: Path | None = None,
    reference_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Phase 1 risk card from the selected artifact and holdout split."""
    base = root or repo_root()
    params = load_params(base / "params.yaml")
    manifest = load_manifest(base / "data/processed/feature_list.json")
    feature_names = list(manifest["feature_names"])
    scaler = load_scaler(base / "data/processed/scaler.joblib")
    champion = load_champion_artifact(base)

    if reference_stats is None:
        reference_stats = read_json(base / "reports/metrics/reference_stats.json")

    train_df = pd.read_parquet(base / "data/processed/train.parquet")
    val_df = pd.read_parquet(base / "data/processed/val.parquet")
    test_df = pd.read_parquet(base / "data/processed/test.parquet")
    aqi_max = float(reference_stats["target_aqi"]["max"])

    # The holdout numbers come from the single-shot evaluation (ADR A1). Refuse
    # a stale report: it must describe this exact test file and champion.
    final_eval_path = base / "reports/metrics/final_test_eval.json"
    final_eval = read_json(final_eval_path)
    test_md5 = file_md5(base / "data/processed/test.parquet")
    if final_eval["splits"]["test"] != test_md5:
        raise RuntimeError(
            "final_test_eval.json was computed on a different test split; "
            "re-run scripts/final_test_eval.py"
        )
    if final_eval["candidate"]["artifact_md5"] != file_md5(champion.path):
        raise RuntimeError(
            "final_test_eval.json belongs to a different champion artifact; "
            "freeze the candidate and re-run scripts/final_test_eval.py"
        )

    persistence = joblib.load(base / "models/persistence.joblib")
    val_persist = predict_regression(
        "persistence", persistence, val_df, scaler, feature_names, aqi_max
    )
    val_pred = predict_regression(
        champion.name, champion.model, val_df, scaler, feature_names, aqi_max
    )

    test_metrics = {key: float(final_eval["metrics"][key]) for key in ("mae", "rmse", "r2")}
    val_skill = skill_vs_persistence(
        val_df["target_aqi"].to_numpy(dtype=float), val_pred, val_persist
    )
    test_skill = float(final_eval["skill_vs_persistence"])
    test_ci = (float(final_eval["skill_ci_lo_95"]), float(final_eval["skill_ci_hi_95"]))

    # Overfitting diagnostics (Plan A4). These use train/validation only; the
    # holdout is never re-used to judge the fit.
    train_pred = predict_regression(
        champion.name, champion.model, train_df, scaler, feature_names, aqi_max
    )
    train_rmse = float(
        regression_metrics(train_df["target_aqi"].to_numpy(dtype=float), train_pred)["rmse"]
    )
    val_rmse = float(champion.val_metrics.get("rmse", float("nan")))
    cv_mean = champion.val_metrics.get("cv_rmse_mean")
    cv_std = champion.val_metrics.get("cv_rmse_std")
    cv_folds = champion.val_metrics.get("cv_folds")
    train_val_gap = val_rmse - train_rmse
    train_val_ratio = val_rmse / train_rmse if train_rmse > 0 else float("inf")
    if not np.isfinite(val_rmse) or train_rmse <= 0:
        interpretation = "insufficient training statistics"
    elif train_val_ratio <= 1.10:
        interpretation = "no material overfitting: validation RMSE is within 10% of training RMSE"
    elif train_val_ratio <= 1.25:
        interpretation = "moderate train/validation gap; validation scores may be mildly optimistic"
    else:
        interpretation = "material train/validation gap; treat validation scores as optimistic"
    diagnostics = {
        "method": "train vs validation RMSE and TimeSeriesSplit CV (holdout not used)",
        "train_rmse": round(train_rmse, 3),
        "val_rmse": round(val_rmse, 3) if np.isfinite(val_rmse) else None,
        "train_val_rmse_gap": round(train_val_gap, 3) if np.isfinite(val_rmse) else None,
        "train_val_rmse_ratio": round(train_val_ratio, 4) if np.isfinite(val_rmse) else None,
        "cv_rmse_mean": round(float(cv_mean), 3) if cv_mean is not None else None,
        "cv_rmse_std": round(float(cv_std), 3) if cv_std is not None else None,
        "cv_folds": int(cv_folds) if cv_folds is not None else None,
        "interpretation": interpretation,
        "val_per_city": [
            {
                "city": row["city"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mae_ratio_vs_persistence": row["mae_ratio_vs_persistence"],
            }
            for row in per_city_records(val_df, val_pred, val_persist)
        ],
    }

    val_metrics_list = read_json(base / "reports/metrics/val_metrics.json")
    classifier_entry = select_classifier(val_metrics_list)
    classifier_name, classifier_path = resolve_classifier_artifact(
        base, str(classifier_entry["model"])
    )
    classifier = joblib.load(classifier_path)
    category_pred = predict_classifier(classifier_name, classifier, test_df, scaler, feature_names)
    true_categories = test_df["target_category"].to_numpy()
    within_one = float(
        np.mean(
            [
                abs(CATEGORY_ORDER.index(t) - CATEGORY_ORDER.index(p)) <= 1
                for t, p in zip(true_categories, category_pred, strict=False)
            ]
        )
    )

    # G3 uses the frozen holdout rows from final_test_eval; G4 adds the Poor+
    # recall that needs classifier predictions.
    poor_plus: dict[str, dict[str, Any]] = {}
    category_frame = pd.DataFrame(
        {"city": test_df["city"].to_numpy(), "category": true_categories, "pred": category_pred}
    )
    for city, group in category_frame.groupby("city"):
        mask = group["category"].isin(POOR_PLUS)
        support = int(mask.sum())
        poor_plus[str(city)] = {
            "poor_plus_support": support,
            "poor_plus_recall": (
                round(float(np.mean(group.loc[mask, "pred"] == group.loc[mask, "category"])), 4)
                if support
                else None
            ),
        }
    per_city = [{**row, **poor_plus.get(str(row["city"]), {})} for row in final_eval["per_city"]]
    failing_cities = [
        row["city"] for row in per_city if float(row["mae_ratio_vs_persistence"]) > 1.05
    ]

    sample = test_df.iloc[[0]].copy()
    feature_frame = rebuild_features(
        sample, feature_names, params, reference_stats["train_medians"]
    )

    def predict_one() -> None:
        predict_regression(
            champion.name, champion.model, feature_frame, scaler, feature_names, aqi_max
        )

    latency = measure_latency(predict_one)

    range_flags = flag_out_of_range(test_df, reference_stats, feature_names)
    manual_notes: list[str] = []

    gates = {
        "G1": {
            "name": "Skill vs persistence > 0 on val and test",
            "threshold": GATE_THRESHOLDS["G1"],
            "value": {
                "val_skill": round(val_skill, 4),
                "test_skill": round(test_skill, 4),
                "test_skill_ci_lo_95": round(test_ci[0], 4),
                "test_skill_ci_hi_95": round(test_ci[1], 4),
            },
            "status": "pass" if val_skill > 0 and test_skill > 0 else "fail",
        },
        "G2": {
            "name": "Provisional test floor",
            "threshold": GATE_THRESHOLDS["G2"],
            "value": {
                "test_r2": round(test_metrics["r2"], 4),
                "test_mae": round(test_metrics["mae"], 3),
            },
            "status": (
                "pass" if test_metrics["r2"] >= 0.65 and test_metrics["mae"] <= 30.0 else "fail"
            ),
        },
        "G3": {
            "name": "Per-city MAE within +5% of persistence",
            "threshold": GATE_THRESHOLDS["G3"],
            "value": {
                "max_ratio": round(
                    max(float(row["mae_ratio_vs_persistence"]) for row in per_city), 4
                ),
                "failing_cities": failing_cities,
            },
            "status": "pass" if not failing_cities else "fail",
        },
        "G4": {
            "name": "Category quality",
            "threshold": GATE_THRESHOLDS["G4"],
            "value": {
                "within_one_category_rate": round(within_one, 4),
                "classifier": classifier_name,
                "classifier_selection": {
                    "selected_on": "validation",
                    "leaderboard_entry": classifier_entry["model"],
                    "val_within_one_category_rate": round(
                        float(classifier_entry.get("within_one_category_rate", 0.0)), 4
                    ),
                    "val_macro_f1": round(float(classifier_entry.get("macro_f1", 0.0)), 4),
                },
                "poor_plus_recall_per_city": {
                    row["city"]: row.get("poor_plus_recall") for row in per_city
                },
            },
            "status": "pass" if within_one >= 0.95 else "fail",
        },
        "G5": {
            "name": "Single-row latency",
            "threshold": GATE_THRESHOLDS["G5"],
            "value": {
                # Wall-clock percentiles are not reproducible byte-for-byte, so
                # the measurement lives in reports/metrics/latency.json and the
                # risk card (which is hashed into the bundle) records only the
                # deterministic gate decision and its evidence pointer.
                "threshold_ms": 50.0,
                "iterations": int(latency["iterations"]),
                "report": "reports/metrics/latency.json",
                "champion": champion.name,
                "random_forest_excluded_as_champion": True,
            },
            "status": "pass" if latency["p95_ms"] < 50.0 else "fail",
        },
    }

    # Latency is measured wall-clock and therefore not reproducible byte-for-byte.
    # Keep it out of the hashed risk card, in its own metrics artifact.
    atomic_write_json(
        base / "reports/metrics/latency.json",
        {
            **latency,
            "threshold_ms": 50.0,
            "champion": champion.name,
        },
    )

    data_files = {
        "train": base / "data/processed/train.parquet",
        "val": base / "data/processed/val.parquet",
        "test": base / "data/processed/test.parquet",
        "feature_list": base / "data/processed/feature_list.json",
        "scaler": base / "data/processed/scaler.joblib",
    }
    split = params.get("split", {})
    commit = git_commit(base)
    return {
        "schema_version": "1.0",
        "generated_at": frame_timestamp(test_df),
        "model": {
            "id": f"aqi-champion-{commit[:8]}-{champion.name}",
            "name": champion.name,
            "path": str(champion.path.relative_to(base)).replace("\\", "/"),
            "artifact_md5": file_md5(champion.path),
            "artifact_size_bytes": int(champion.path.stat().st_size),
            "selection_rule": (
                "validation-only: accept a monotone candidate within 2% of the "
                "incumbent validation RMSE, then minimise validation RMSE subject "
                "to per-city MAE ratio <= 1.05 (ADRs D-030/D-031); Random Forest "
                "excluded on artifact size"
            ),
            "val_metrics": {
                key: round(float(value), 4)
                for key, value in champion.val_metrics.items()
                if isinstance(value, (int, float))
            },
            "calibration": champion.val_metrics.get("calibration"),
            "monotone_constraints": champion.val_metrics.get("monotone_constraints"),
            "test_metrics": {
                "mae": round(test_metrics["mae"], 3),
                "rmse": round(test_metrics["rmse"], 3),
                "r2": round(test_metrics["r2"], 4),
            },
        },
        "data": {
            "md5": file_md5(data_files["test"]),
            "file_md5": {name: file_md5(path) for name, path in data_files.items()},
            "n_features": len(feature_names),
            "split_dates": {
                "train_end": split.get("train_end"),
                "val_end": split.get("val_end"),
                "test_start": str(pd.to_datetime(test_df["date"]).min().date()),
                "test_end": str(pd.to_datetime(test_df["date"]).max().date()),
            },
            "counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
                "test_cities": int(test_df["city"].nunique()),
            },
        },
        "git_commit": commit,
        "gates": gates,
        "per_city": per_city,
        "diagnostics": diagnostics,
        "test_counts": {
            "rows": int(len(test_df)),
            "cities": int(test_df["city"].nunique()),
            "dates": int(test_df["date"].nunique()),
        },
        "latency": {**latency, "threshold_ms": 50.0},
        "range_check": {
            "tolerance": reference_stats["tolerance"],
            "n_flagged_features": len(range_flags),
            "flagged_features": range_flags,
        },
        "known_limitations": [
            "Pune is data-thin (900 train rows versus roughly 1,820-1,826 for other cities); its per-city error has wider uncertainty (ADR D-022).",
            "Training uses ERA5 reanalysis weather, while production inference uses forecast weather; distribution mismatch is possible.",
            "Annual vehicle registration data is interpolated to daily values, so short-term emission changes are smoothed.",
        ],
        "manual_actions": manual_notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Model testing and risk/quality audit")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--reference-stats",
        type=Path,
        default=Path("reports/metrics/reference_stats.json"),
    )
    parser.add_argument(
        "--risk-card", type=Path, default=Path("reports/metrics/model_risk_card.json")
    )
    args = parser.parse_args()

    base = args.root or repo_root()
    params = load_params(base / "params.yaml")
    manifest = load_manifest(base / "data/processed/feature_list.json")
    train_df = pd.read_parquet(base / "data/processed/train.parquet")

    reference_stats = build_reference_stats(train_df, manifest["feature_names"], params, root=base)
    stats_path = (
        args.reference_stats if args.reference_stats.is_absolute() else base / args.reference_stats
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stats_path, reference_stats)

    risk_card = build_risk_card(base, reference_stats=reference_stats)
    card_path = args.risk_card if args.risk_card.is_absolute() else base / args.risk_card
    card_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(card_path, risk_card)

    print("\n" + "=" * 78)
    print("PHASE 1: MODEL RISK / QUALITY AUDIT")
    print("=" * 78)
    print(f"Champion: {risk_card['model']['name']}  ({risk_card['model']['id']})")
    print(
        f"Features: {risk_card['data']['n_features']}  Test rows: {risk_card['test_counts']['rows']}"
    )
    for gate_id, gate in risk_card["gates"].items():
        detail = gate["value"]
        print(f"  {gate_id}: {gate['status'].upper():4s}  {detail}")
    print(f"Risk card -> {card_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
