"""Phase 0-3 integrity audit.

Recomputes, from the frozen artifacts and splits, every property the pipeline
relies on and writes ``reports/metrics/integrity_report.json``. A failed check
exits non-zero, so CI can block on it.

Two modes:

* ``python scripts/audit_integrity.py`` - full audit (needs the DVC splits).
* ``python scripts/audit_integrity.py --bundle-only`` - bundle, manifest and
  risk-card identity only (works on a checkout without DVC data).

The audit is read-only apart from the report it writes. It never trains, never
downloads, and never touches the test split's role as a held-out set: it merely
verifies that the stored numbers still match a fresh computation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/audit_integrity.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import load_params  # noqa: E402
from src.models import model_risk as mr  # noqa: E402
from src.serving.bundle import (  # noqa: E402
    FORBIDDEN_SEED_COLUMNS,
    HISTORY_SEED_FILENAME,
    MANIFEST_FILENAME,
    REFERENCE_STATS_FILENAME,
    AqiBundle,
    sha256_file,
)
from src.utils.io import atomic_write_json  # noqa: E402
from src.utils.reproducibility import frame_timestamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
SCALER_TOLERANCE = 1e-6


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - DVC identity, not a security control
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    """Accumulates named pass/fail checks with evidence."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> bool:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    @property
    def ok(self) -> bool:
        return all(item["passed"] for item in self.results)

    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.results if not item["passed"]]


def _dvc_lock_md5(root: Path) -> dict[str, str]:
    lock_path = root / "dvc.lock"
    with lock_path.open(encoding="utf-8") as fh:
        lock = yaml.safe_load(fh)
    found: dict[str, str] = {}
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            path = str(out.get("path", ""))
            if path.startswith("data/processed/") and out.get("md5"):
                found[Path(path).name] = str(out["md5"])
    return found


def audit_bundle(audit: Audit, root: Path) -> dict[str, Any]:
    """Bundle/manifest/risk-card identity checks that need no DVC data."""
    bundle_dir = root / "artifacts/champion"
    try:
        bundle = AqiBundle.load(bundle_dir)  # verifies MANIFEST + every payload hash
        audit.check("bundle.load_verifies_hashes", True, str(bundle_dir))
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        audit.check("bundle.load_verifies_hashes", False, f"{type(exc).__name__}: {exc}")
        return {}

    manifest = bundle.manifest
    audit.check(
        "bundle.manifest_sha256_matches",
        (bundle_dir / "MANIFEST.sha256").read_text(encoding="utf-8").strip()
        == sha256_file(bundle_dir / MANIFEST_FILENAME),
        MANIFEST_FILENAME,
    )
    audit.check(
        "bundle.payload_sha256_matches",
        manifest.get("bundle_sha256") == sha256_file(bundle_dir / "bundle.joblib"),
        "bundle.joblib",
    )
    audit.check(
        "bundle.reference_stats_sha256_matches",
        manifest.get("reference_stats_sha") == sha256_file(bundle_dir / REFERENCE_STATS_FILENAME),
        REFERENCE_STATS_FILENAME,
    )
    audit.check(
        "bundle.history_seed_sha256_matches",
        manifest.get("history_seed_sha") == sha256_file(bundle_dir / HISTORY_SEED_FILENAME),
        HISTORY_SEED_FILENAME,
    )
    audit.check(
        "bundle.feature_count_matches_payload",
        manifest.get("features_count") == len(bundle.feature_names),
        {"manifest": manifest.get("features_count"), "bundle": len(bundle.feature_names)},
    )

    seed = bundle.history_seed
    leaked = sorted(FORBIDDEN_SEED_COLUMNS & set(seed.columns))
    audit.check("bundle.history_seed_has_no_labels", not leaked, leaked)
    counts = seed.groupby("city")["date"].count()
    audit.check(
        "bundle.history_seed_is_45_days_per_city",
        bool((counts == manifest.get("history_seed_days")).all()),
        {"days": int(manifest.get("history_seed_days", -1)), "cities": int(len(counts))},
    )

    risk_card_path = root / "reports/metrics/model_risk_card.json"
    risk_card = mr.read_json(risk_card_path)
    audit.check(
        "bundle.risk_card_sha256_matches",
        manifest.get("risk_card_sha") == sha256_file(risk_card_path),
        "reports/metrics/model_risk_card.json",
    )
    audit.check(
        "bundle.params_sha256_matches",
        manifest.get("params_sha256") == sha256_file(root / "params.yaml"),
        "params.yaml",
    )
    audit.check(
        "risk_card.generated_at_is_data_derived",
        risk_card.get("generated_at", "").startswith(
            str(risk_card["data"]["split_dates"]["test_end"])
        ),
        {
            "generated_at": risk_card.get("generated_at"),
            "test_end": risk_card["data"]["split_dates"]["test_end"],
        },
    )
    artifacts_by_name = {
        "train": root / "data/processed/train.parquet",
        "val": root / "data/processed/val.parquet",
        "test": root / "data/processed/test.parquet",
        "feature_list": root / "data/processed/feature_list.json",
        "scaler": root / "data/processed/scaler.joblib",
    }
    for name, expected_md5 in risk_card["data"].get("file_md5", {}).items():
        path = artifacts_by_name.get(name)
        audit.check(
            f"risk_card.file_md5_matches.{name}",
            path is not None and path.exists() and md5_file(path) == expected_md5,
            {"path": str(path.relative_to(root)) if path else None},
        )

    status = {gate: item["status"] for gate, item in risk_card["gates"].items()}
    audit.check("bundle.gate_status_matches_card", manifest.get("gate_status") == status, status)
    failing = {gate for gate, value in status.items() if value != "pass"}
    audit.check(
        "bundle.waivers_cover_exactly_the_failing_gates",
        set(manifest.get("gate_waivers", {})) == failing,
        {"failing": sorted(failing), "waived": sorted(manifest.get("gate_waivers", {}))},
    )
    return {"manifest": manifest, "risk_card": risk_card, "bundle": bundle}


def audit_splits(audit: Audit, root: Path) -> dict[str, pd.DataFrame]:
    """Frozen-split, schema, leakage and scaler checks."""
    params = load_params(root / "params.yaml")
    frames = {name: pd.read_parquet(root / "data/processed" / f"{name}.parquet") for name in SPLITS}

    expected = params.get("integrity", {})
    lock_md5 = _dvc_lock_md5(root)
    for name, frame in frames.items():
        path = root / "data/processed" / f"{name}.parquet"
        actual = md5_file(path)
        audit.check(
            f"split.{name}_md5_matches_params",
            actual == expected.get(f"{name}_md5"),
            {"actual": actual, "expected": expected.get(f"{name}_md5")},
        )
        audit.check(
            f"split.{name}_md5_matches_dvc_lock",
            actual == lock_md5.get(f"{name}.parquet"),
            {"actual": actual, "dvc_lock": lock_md5.get(f"{name}.parquet")},
        )
        audit.check(
            f"split.{name}_has_no_duplicate_city_dates",
            not frame.duplicated(subset=["city", "date"]).any(),
            int(frame.duplicated(subset=["city", "date"]).sum()),
        )
        dates = pd.to_datetime(frame["date"])
        audit.check(
            f"split.{name}_dates_are_monotonic_per_city",
            all(
                group["date"].is_monotonic_increasing
                for _, group in frame.assign(date=dates).groupby("city")
            ),
            None,
        )

    split = params.get("split", {})
    train_end = pd.Timestamp(split["train_end"])
    val_end = pd.Timestamp(split["val_end"])
    dates = {name: pd.to_datetime(frame["date"]) for name, frame in frames.items()}
    audit.check(
        "split.train_ends_on_train_end",
        dates["train"].max() <= train_end,
        str(dates["train"].max()),
    )
    audit.check(
        "split.val_fits_the_val_window",
        dates["val"].min() > train_end and dates["val"].max() <= val_end,
        {"min": str(dates["val"].min()), "max": str(dates["val"].max())},
    )
    audit.check(
        "split.test_starts_after_val_end", dates["test"].min() > val_end, str(dates["test"].min())
    )
    audit.check(
        "split.windows_do_not_overlap",
        dates["train"].max() < dates["val"].min() and dates["val"].max() < dates["test"].min(),
        None,
    )
    audit.check(
        "split.test_covers_every_city_with_1500_rows",
        len(frames["test"]) >= 1500 and frames["test"]["city"].nunique() == 9,
        {"rows": len(frames["test"]), "cities": int(frames["test"]["city"].nunique())},
    )

    feature_manifest = mr.read_json(root / "data/processed/feature_list.json")
    names = list(feature_manifest["feature_names"])
    audit.check(
        "features.count_matches_manifest",
        feature_manifest["n_features"] == len(names) > 0,
        {"n_features": feature_manifest["n_features"], "len": len(names)},
    )
    for name, frame in frames.items():
        audit.check(
            f"features.{name}_column_order_matches_manifest",
            [c for c in frame.columns if c in set(names)] == names,
            None,
        )
        values = frame[names].to_numpy(dtype=float)
        audit.check(f"features.{name}_are_finite", bool(np.isfinite(values).all()), None)

    # One scaler, fitted on train only.
    scaler = joblib.load(root / "data/processed/scaler.joblib")
    fresh = StandardScaler().fit(frames["train"][names])
    audit.check(
        "scaler.mean_matches_a_fresh_train_fit",
        float(np.max(np.abs(scaler.mean_ - fresh.mean_))) <= SCALER_TOLERANCE,
        float(np.max(np.abs(scaler.mean_ - fresh.mean_))),
    )
    audit.check(
        "scaler.scale_matches_a_fresh_train_fit",
        float(np.max(np.abs(scaler.scale_ - fresh.scale_))) <= SCALER_TOLERANCE,
        float(np.max(np.abs(scaler.scale_ - fresh.scale_))),
    )
    audit.check("scaler.feature_count_matches_manifest", scaler.n_features_in_ == len(names), None)

    # Target and lag identities: tomorrow's AQI, never today's.
    for name, frame in frames.items():
        ordered = frame.sort_values(["city", "date"])
        target_ok = True
        lag_ok = True
        for _, group in ordered.groupby("city", sort=False):
            group = group.reset_index(drop=True)
            dates = pd.to_datetime(group["date"])
            next_day = (dates.shift(-1) - dates).dt.days == 1
            prev_day = (dates - dates.shift(1)).dt.days == 1
            if next_day.any():
                target_ok &= bool(
                    np.allclose(
                        group.loc[next_day, "target_aqi"].to_numpy(dtype=float),
                        group["aqi"].shift(-1)[next_day].to_numpy(dtype=float),
                        atol=1e-6,
                    )
                )
            if prev_day.any():
                lag_ok &= bool(
                    np.allclose(
                        group.loc[prev_day, "aqi_lag_1"].to_numpy(dtype=float),
                        group["aqi"].shift(1)[prev_day].to_numpy(dtype=float),
                        atol=1e-6,
                    )
                )
        audit.check(f"leakage.{name}_target_is_next_day_aqi", target_ok, None)
        audit.check(f"leakage.{name}_aqi_lag_1_is_previous_day", lag_ok, None)

    # Reference stats / feature list timestamps are data-derived, not clocks.
    reference_stats = mr.read_json(root / "reports/metrics/reference_stats.json")
    audit.check(
        "repro.reference_stats_timestamp_is_data_derived",
        reference_stats.get("generated_at") == frame_timestamp(frames["train"]),
        reference_stats.get("generated_at"),
    )
    audit.check(
        "repro.feature_list_timestamp_is_data_derived",
        feature_manifest.get("generated_at") == frame_timestamp(frames["test"]),
        feature_manifest.get("generated_at"),
    )

    risk_card = mr.read_json(root / "reports/metrics/model_risk_card.json")
    champion = mr.load_champion_artifact(root)
    aqi_max = float(reference_stats["target_aqi"]["max"])
    preds = mr.predict_regression(
        champion.name, champion.model, frames["test"], scaler, names, aqi_max
    )
    test = frames["test"]
    for row in risk_card["per_city"]:
        mask = test["city"] == row["city"]
        recomputed = float(np.mean(np.abs(test.loc[mask, "target_aqi"] - preds[mask])))
        audit.check(
            f"risk_card.per_city_mae_matches.{row['city']}",
            abs(recomputed - float(row["mae"])) <= 1e-3,
            {"card": row["mae"], "recomputed": round(recomputed, 4)},
        )
    audit.check(
        "risk_card.champion_matches_artifact",
        risk_card["model"]["name"] == champion.name,
        {"card": risk_card["model"]["name"], "artifact": champion.name},
    )
    audit.check(
        "risk_card.champion_artifact_md5_matches",
        risk_card["model"]["artifact_md5"] == mr.file_md5(champion.path),
        None,
    )
    return frames


def run_audit(root: Path, bundle_only: bool = False) -> dict[str, Any]:
    audit = Audit()
    bundle_info = audit_bundle(audit, root)
    if not bundle_only:
        audit_splits(audit, root)
    report = {
        "schema_version": "1.0",
        "mode": "bundle-only" if bundle_only else "full",
        "passed": audit.ok,
        "checks_total": len(audit.results),
        "checks_failed": len(audit.failures()),
        "failures": audit.failures(),
        "checks": audit.results,
    }
    if bundle_info.get("manifest"):
        report["bundle_sha256"] = bundle_info["manifest"].get("bundle_sha256")
        report["manifest_sha256"] = sha256_file(root / "artifacts/champion/MANIFEST.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/metrics/integrity_report.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report

    report = run_audit(root, bundle_only=args.bundle_only)
    atomic_write_json(report_path, report)

    print("\n" + "=" * 78)
    print(f"INTEGRITY AUDIT ({report['mode']})")
    print("=" * 78)
    print(f"Checks: {report['checks_total']}  failed: {report['checks_failed']}")
    for item in report["failures"]:
        print(f"  FAIL {item['check']}: {item['detail']}")
    print(f"Report -> {report_path}")
    print("=" * 78 + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
