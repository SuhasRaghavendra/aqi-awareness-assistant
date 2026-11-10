"""Phase 2: package the champion model into ``artifacts/champion``."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import load_params
from src.models import model_risk
from src.serving.bundle import (
    HISTORY_SEED_FILENAME,
    REFERENCE_STATS_FILENAME,
    AqiBundle,
    sha256_file,
)

GATE_WAIVERS = {
    "G3": (
        "Accepted red gate per ADR D-023: Kolkata fails the +5% per-city MAE rule; "
        "the threshold is unchanged and the failure stays visible in the risk card."
    )
}

HISTORY_SEED_DAYS = 45
DVC_DATA_PATH = "data/processed/train.parquet"
PACKAGE_NAMES = ["pandas", "numpy", "scikit-learn", "lightgbm", "joblib", "pyarrow", "mlflow"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def read_dvc_data_md5(root: Path, data_path: str = DVC_DATA_PATH) -> str:
    """Read the training-data md5 from dvc.lock."""
    lock_path = root / "dvc.lock"
    if not lock_path.exists():
        raise FileNotFoundError(f"dvc.lock is missing: {lock_path}")
    with lock_path.open(encoding="utf-8") as fh:
        lock = yaml.safe_load(fh)
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == data_path and out.get("md5"):
                return str(out["md5"])
    raise ValueError(f"No md5 for {data_path} found in dvc.lock")


def infer_input_kind(model_name: str) -> str:
    if model_name.startswith("residual_"):
        return "residual_scaled_df"
    if model_name == "ridge":
        return "scaled_array"
    if model_name in model_risk.BASELINE_MODELS:
        return "raw_df"
    return "scaled_df"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def residual_quantiles(val_df: pd.DataFrame, preds: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute per-city residual quantiles (10/50/90) on validation."""
    frame = val_df[["city", "target_aqi"]].copy()
    frame["residual"] = frame["target_aqi"].to_numpy(dtype=float) - np.asarray(preds)

    def _quantiles(values: pd.Series) -> dict[str, float]:
        q10, q50, q90 = np.quantile(values.to_numpy(dtype=float), [0.1, 0.5, 0.9])
        return {
            "q10": round(float(q10), 4),
            "q50": round(float(q50), 4),
            "q90": round(float(q90), 4),
        }

    result = {"__global__": _quantiles(frame["residual"])}
    for city, group in frame.groupby("city"):
        result[str(city)] = _quantiles(group["residual"])
    return result


def build_history_seed(val_df: pd.DataFrame, days: int = HISTORY_SEED_DAYS) -> pd.DataFrame:
    """Take the most recent ``days`` validation rows per city as serving history."""
    frame = val_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    last_date = frame.groupby("city")["date"].transform("max")
    seed = frame[frame["date"] >= last_date - pd.Timedelta(days=days - 1)].copy()
    seed["date"] = seed["date"].dt.strftime("%Y-%m-%d")
    return seed.sort_values(["city", "date"]).reset_index(drop=True)


def check_gate_policy(risk_card: dict[str, Any]) -> dict[str, str]:
    """Reject any un-waived failing gate; return the recorded status map."""
    gates = risk_card.get("gates")
    if gates is not None:
        status = {gate_id: gate["status"] for gate_id, gate in gates.items()}
    else:
        status = dict(risk_card.get("gate_status", {}))
    if not status:
        raise ValueError("No gate status found in the risk card or bundle manifest")
    blocked = {
        gate_id
        for gate_id, gate_status in status.items()
        if gate_status != "pass" and gate_id not in GATE_WAIVERS
    }
    if blocked:
        raise RuntimeError(f"Un-waived failing quality gates: {sorted(blocked)}")
    return status


def check_champion_rule(root: Path, risk_card: dict[str, Any]) -> str:
    """Verify the risk-card champion is the lowest-val-RMSE eligible model."""
    val_metrics = read_json(root / "reports/metrics/val_metrics.json")
    expected = model_risk.select_champion(val_metrics)["model"]
    recorded = str(risk_card["model"]["name"])
    if expected != recorded:
        raise RuntimeError(
            f"Champion mismatch: risk card has {recorded!r}, expected {expected!r}; "
            "re-run dvc repro model_risk"
        )
    return recorded


def build_bundle(root: Path | None = None) -> tuple[AqiBundle, dict[str, Any], Path]:
    """Build and persist the champion bundle, returning it with its manifest."""
    base = root or repo_root()
    params = load_params(base / "params.yaml")
    manifest_meta = model_risk.load_manifest(base / "data/processed/feature_list.json")
    feature_names = list(manifest_meta["feature_names"])
    scaler = model_risk.load_scaler(base / "data/processed/scaler.joblib")
    reference_stats = read_json(base / "reports/metrics/reference_stats.json")
    risk_card = read_json(base / "reports/metrics/model_risk_card.json")

    champion_name = check_champion_rule(base, risk_card)
    gate_status = check_gate_policy(risk_card)
    champion = model_risk.load_champion_artifact(base)
    if champion.name != champion_name:
        raise RuntimeError(
            f"Loaded champion {champion.name!r} does not match risk card {champion_name!r}"
        )
    if model_risk.file_md5(champion.path) != str(risk_card["model"]["artifact_md5"]):
        raise RuntimeError("Champion artifact hash does not match the risk card")

    val_df = pd.read_parquet(base / "data/processed/val.parquet")
    aqi_max = float(reference_stats["target_aqi"]["max"])
    val_preds = model_risk.predict_regression(
        champion.name, champion.model, val_df, scaler, feature_names, aqi_max
    )
    quantiles = residual_quantiles(val_df, val_preds)

    fallback_path = base / "models/persistence.joblib"
    if not fallback_path.exists():
        raise FileNotFoundError(f"Fallback artifact is missing: {fallback_path}")
    fallback_model = joblib.load(fallback_path)

    seed = build_history_seed(val_df)
    bundle_dir = base / "artifacts/champion"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    seed.to_parquet(bundle_dir / HISTORY_SEED_FILENAME, index=False)
    with (bundle_dir / REFERENCE_STATS_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(reference_stats, fh, indent=2)

    dvc_data_md5 = read_dvc_data_md5(base, DVC_DATA_PATH)
    feature_config = {
        "features": params.get("features", {}),
        "data": {"cities": params.get("data", {}).get("cities", [])},
    }
    feature_names_sha = hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()
    manifest = {
        "bundle_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "model": {
            "name": champion.name,
            "path": str(champion.path.relative_to(base)).replace("\\", "/"),
            "artifact_md5": model_risk.file_md5(champion.path),
            "artifact_size_bytes": int(champion.path.stat().st_size),
            "input_kind": infer_input_kind(champion.name),
            "residual_mode": champion.name.startswith("residual_"),
            "val_metrics": champion.val_metrics,
            "test_metrics": risk_card["model"]["test_metrics"],
        },
        "fallback": {
            "name": "persistence",
            "path": "models/persistence.joblib",
            "artifact_md5": model_risk.file_md5(fallback_path),
        },
        "features_count": len(feature_names),
        "feature_names_sha256": feature_names_sha,
        "git_commit": git_commit(base),
        "dvc_data_path": DVC_DATA_PATH,
        "dvc_data_md5": dvc_data_md5,
        "dvc_data_version": dvc_data_md5,
        "params_sha256": sha256_file(base / "params.yaml"),
        "risk_card_sha": sha256_file(base / "reports/metrics/model_risk_card.json"),
        "reference_stats_sha": sha256_file(bundle_dir / REFERENCE_STATS_FILENAME),
        "history_seed_sha": sha256_file(bundle_dir / HISTORY_SEED_FILENAME),
        "history_seed_rows": int(len(seed)),
        "history_seed_days": HISTORY_SEED_DAYS,
        "split": risk_card["data"]["split_dates"],
        "gate_status": gate_status,
        "gate_waivers": GATE_WAIVERS,
        "promotion_policy": (
            "Registration sets the challenger alias; champion is set only after "
            "MANIFEST sha and gate validation."
        ),
        "residual_quantiles": quantiles,
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
    }

    bundle = AqiBundle(
        model=champion.model,
        scaler=scaler,
        feature_names=feature_names,
        input_kind=infer_input_kind(champion.name),
        residual_quantiles=quantiles,
        reference_stats=reference_stats,
        history_seed=seed,
        manifest=manifest,
        feature_config=feature_config,
        fallback_model=fallback_model,
        fallback_name="persistence",
    )
    bundle.save(bundle_dir, include_payload=False)
    return bundle, manifest, bundle_dir


def main() -> None:
    bundle, manifest, bundle_dir = build_bundle()
    print("\n" + "=" * 78)
    print("PHASE 2: CHAMPION BUNDLE")
    print("=" * 78)
    print(f"Bundle dir:       {bundle_dir}")
    print(f"Champion:         {manifest['model']['name']} ({manifest['model']['input_kind']})")
    print(f"Features:         {manifest['features_count']}")
    print(
        f"History seed:     {manifest['history_seed_rows']} rows / {manifest['history_seed_days']} days"
    )
    print(f"DVC data md5:     {manifest['dvc_data_md5']}")
    print(f"Gate status:      {manifest['gate_status']}")
    print(f"Accepted waivers: {sorted(manifest['gate_waivers'])}")
    print(f"MANIFEST sha:     {(bundle_dir / 'MANIFEST.sha256').read_text().strip()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
