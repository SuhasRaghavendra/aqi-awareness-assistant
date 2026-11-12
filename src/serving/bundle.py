"""Serving-side champion bundle for the AQI predictor (Phase 2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import lightgbm  # noqa: F401  # import before pandas/pyarrow on Windows (ADR D-015)
import numpy as np
import pandas as pd

from src.features.serving_features import build_feature_frame
from src.utils.io import (
    atomic_write_bytes,
    atomic_write_joblib,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)

BUNDLE_FILENAME = "bundle.joblib"
MANIFEST_FILENAME = "MANIFEST.json"
MANIFEST_SHA_FILENAME = "MANIFEST.sha256"
REFERENCE_STATS_FILENAME = "reference_stats.json"
HISTORY_SEED_FILENAME = "history_seed.parquet"

KNOWN_INPUT_KINDS = frozenset({"residual_scaled_df", "scaled_array", "raw_df", "scaled_df"})
FORBIDDEN_SEED_COLUMNS = frozenset({"target_aqi", "target_category"})


class BundleIntegrityError(ValueError):
    """Raised when a bundle on disk fails hash, schema or consistency checks."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_estimators(model: Any) -> Iterator[Any]:
    """Yield the model and every wrapped base estimator.

    Calibrated champions nest ``CityCalibratedRegressor -> ResidualRegressor ->
    LGBMRegressor``, so the walk must follow the whole ``base_model`` chain.
    """
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        estimator = stack.pop()
        if id(estimator) in seen:
            continue
        seen.add(id(estimator))
        yield estimator
        base = getattr(estimator, "base_model", None)
        if base is not None and base is not estimator:
            stack.append(base)


@dataclass
class AqiBundle:
    """Champion model plus everything needed for serving-side inference."""

    model: Any
    scaler: Any
    feature_names: list[str]
    input_kind: str
    residual_quantiles: dict[str, dict[str, float]]
    reference_stats: dict[str, Any]
    history_seed: pd.DataFrame
    manifest: dict[str, Any]
    feature_config: dict[str, Any] = field(default_factory=dict)
    fallback_model: Any = None
    fallback_name: str = "persistence"

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def residual_mode(self) -> bool:
        return self.input_kind.startswith("residual")

    @property
    def aqi_max(self) -> float:
        return float(self.reference_stats["target_aqi"]["max"])

    def _scaled(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")
        scaled = self.scaler.transform(frame[self.feature_names])
        return scaled, pd.DataFrame(scaled, columns=self.feature_names)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict next-day AQI from a frame containing the manifest features."""
        from src.models.calibration import CityCalibratedRegressor

        scaled, scaled_df = self._scaled(frame)
        if isinstance(self.model, CityCalibratedRegressor):
            preds = self.model.predict(
                scaled_df,
                frame["aqi"].to_numpy(dtype=float),
                city=frame["city"].to_numpy(),
            )
        elif self.input_kind == "residual_scaled_df":
            preds = self.model.predict(scaled_df, frame["aqi"].to_numpy(dtype=float))
        elif self.input_kind == "scaled_array":
            preds = self.model.predict(scaled)
        elif self.input_kind == "raw_df":
            preds = self.model.predict(frame)
        else:
            preds = self.model.predict(scaled_df)
        return np.clip(np.asarray(preds, dtype=float), 0.0, self.aqi_max)

    def predict_raw(self, panel: pd.DataFrame) -> np.ndarray:
        """Rebuild features from a raw city-day panel slice, then predict."""
        frame = build_feature_frame(
            panel,
            self.feature_names,
            self.feature_config,
            self.reference_stats["train_medians"],
        )
        return self.predict(frame)

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, median, upper) prediction intervals from residual quantiles."""
        preds = self.predict(frame)
        global_q = self.residual_quantiles["__global__"]
        cities = frame["city"].to_numpy() if "city" in frame.columns else None
        lower: list[float] = []
        median: list[float] = []
        upper: list[float] = []
        for index, pred in enumerate(preds):
            city = str(cities[index]) if cities is not None else None
            quantiles = self.residual_quantiles.get(city, global_q)
            lower.append(float(np.clip(pred + quantiles["q10"], 0.0, self.aqi_max)))
            median.append(float(np.clip(pred + quantiles["q50"], 0.0, self.aqi_max)))
            upper.append(float(np.clip(pred + quantiles["q90"], 0.0, self.aqi_max)))
        return np.asarray(lower), np.asarray(median), np.asarray(upper)

    def predict_fallback(self, frame: pd.DataFrame) -> np.ndarray:
        """Persistence fallback used when the champion cannot serve."""
        if self.fallback_model is None:
            raise RuntimeError("No fallback model is bundled")
        preds = self.fallback_model.predict(frame)
        return np.clip(np.asarray(preds, dtype=float), 0.0, self.aqi_max)

    # ── Integrity ────────────────────────────────────────────────────────────
    def verify_structure(self) -> None:
        """Check that the payload, scaler, model and manifest agree.

        This is deliberately independent of the file hashes: it catches a bundle
        whose parts were hashed correctly but whose internal contract drifted
        (for example a scaler fitted on a different feature count).
        """
        names = list(self.feature_names)
        if not names:
            raise BundleIntegrityError("Bundle feature_names is empty")
        if self.input_kind not in KNOWN_INPUT_KINDS:
            raise BundleIntegrityError(f"Unknown input_kind: {self.input_kind!r}")

        declared = int(self.manifest.get("features_count", -1))
        if declared != len(names):
            raise BundleIntegrityError(
                f"features_count mismatch: manifest={declared}, bundle={len(names)}"
            )
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        if digest != self.manifest.get("feature_names_sha256"):
            raise BundleIntegrityError("feature_names_sha256 does not match the bundle features")

        for estimator in _iter_estimators(self.model):
            count = getattr(estimator, "n_features_in_", None)
            if count is not None and int(count) != len(names):
                raise BundleIntegrityError(
                    f"Model expects {int(count)} features but the bundle declares {len(names)}"
                )
            booster = getattr(estimator, "booster_", None)
            if (
                booster is not None
                and hasattr(booster, "num_feature")
                and int(booster.num_feature()) != len(names)
            ):
                raise BundleIntegrityError(
                    f"LightGBM booster expects {int(booster.num_feature())} features, "
                    f"bundle declares {len(names)}"
                )

        scaler_count = getattr(self.scaler, "n_features_in_", None)
        if scaler_count is not None and int(scaler_count) != len(names):
            raise BundleIntegrityError(
                f"Scaler expects {int(scaler_count)} features, bundle declares {len(names)}"
            )

        if "__global__" not in self.residual_quantiles:
            raise BundleIntegrityError("residual_quantiles is missing the __global__ entry")
        for city, quantiles in self.residual_quantiles.items():
            try:
                ordered = quantiles["q10"] <= quantiles["q50"] <= quantiles["q90"]
            except KeyError as exc:
                raise BundleIntegrityError(f"residual_quantiles[{city!r}] is incomplete") from exc
            if not ordered:
                raise BundleIntegrityError(f"residual_quantiles[{city!r}] are not ordered")

        seed = self.history_seed
        if seed is None or seed.empty:
            raise BundleIntegrityError("history_seed is empty")
        for column in ("city", "date", "aqi"):
            if column not in seed.columns:
                raise BundleIntegrityError(f"history_seed is missing {column!r}")
        leaked = FORBIDDEN_SEED_COLUMNS & set(seed.columns)
        if leaked:
            raise BundleIntegrityError(
                f"history_seed must not ship label columns: {sorted(leaked)}"
            )
        cities = set(seed["city"].astype(str))
        missing = cities - set(self.residual_quantiles)
        if missing:
            raise BundleIntegrityError(f"residual_quantiles is missing cities: {sorted(missing)}")

        from src.models.calibration import CityCalibratedRegressor

        if isinstance(self.model, CityCalibratedRegressor):
            uncalibrated = cities - set(self.model.offsets)
            if uncalibrated:
                raise BundleIntegrityError(
                    f"city calibration is missing cities: {sorted(uncalibrated)}"
                )

    def save(self, path: str | Path, include_payload: bool = True) -> dict[str, Any]:
        """Persist the bundle, MANIFEST.json and MANIFEST.sha256 atomically.

        ``MANIFEST.json`` is the single source of truth: it is written last (its
        digest is recorded in ``MANIFEST.sha256``) and it carries the SHA-256 of
        every payload, including ``bundle.joblib`` itself. The manifest is
        therefore *not* embedded in the pickle; :meth:`load` attaches it, which
        keeps one authoritative copy and lets the payload be hashed.

        Text artifacts are written with LF line endings on every platform so the
        digests describe the same bytes in every checkout (Windows, Linux, CI).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.manifest)

        if include_payload:
            atomic_write_parquet(self.history_seed, path / HISTORY_SEED_FILENAME)
            atomic_write_json(path / REFERENCE_STATS_FILENAME, self.reference_stats)
            manifest["reference_stats_sha"] = sha256_file(path / REFERENCE_STATS_FILENAME)
            manifest["history_seed_sha"] = sha256_file(path / HISTORY_SEED_FILENAME)
            manifest["history_seed_rows"] = int(len(self.history_seed))

        payload = replace(self, manifest={})
        atomic_write_joblib(payload, path / BUNDLE_FILENAME)
        manifest["bundle_sha256"] = sha256_file(path / BUNDLE_FILENAME)
        manifest["bundle_size_bytes"] = int((path / BUNDLE_FILENAME).stat().st_size)

        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        atomic_write_bytes(path / MANIFEST_FILENAME, manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        atomic_write_text(path / MANIFEST_SHA_FILENAME, f"{digest}\n")

        self.manifest = manifest
        return manifest

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> AqiBundle:
        """Load a bundle directory, verifying every hash before unpickling.

        With ``verify=True`` (the default, used by the service, the registry and
        every test) the loader refuses to deserialise ``bundle.joblib`` unless
        ``MANIFEST.sha256`` matches ``MANIFEST.json`` and ``MANIFEST.json``
        matches every payload on disk. ``verify=False`` exists only for
        forensics on a known-corrupt bundle and is never used by serving.
        """
        base = Path(path)
        bundle_path = base / BUNDLE_FILENAME
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle artifact is missing: {bundle_path}")

        if not verify:
            bundle = joblib.load(bundle_path)
            if not isinstance(bundle, cls):
                raise TypeError(f"Unexpected bundle type: {type(bundle)!r}")
            return bundle

        manifest_path = base / MANIFEST_FILENAME
        sha_path = base / MANIFEST_SHA_FILENAME
        for required in (manifest_path, sha_path):
            if not required.exists():
                raise FileNotFoundError(f"Bundle manifest is incomplete: {required}")

        expected_manifest_sha = sha_path.read_text(encoding="utf-8").strip()
        actual_manifest_sha = sha256_file(manifest_path)
        if expected_manifest_sha != actual_manifest_sha:
            raise BundleIntegrityError(
                f"MANIFEST.json does not match MANIFEST.sha256: "
                f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_sha = manifest.get("bundle_sha256")
        if not payload_sha:
            raise BundleIntegrityError("MANIFEST.json has no bundle_sha256")
        # Hash before unpickling: never execute bytes from a tampered payload.
        actual_payload_sha = sha256_file(bundle_path)
        if actual_payload_sha != payload_sha:
            raise BundleIntegrityError(
                f"bundle.joblib does not match MANIFEST.json: "
                f"expected {payload_sha}, got {actual_payload_sha}"
            )

        for filename, key in (
            (REFERENCE_STATS_FILENAME, "reference_stats_sha"),
            (HISTORY_SEED_FILENAME, "history_seed_sha"),
        ):
            artifact = base / filename
            if not artifact.exists():
                raise FileNotFoundError(f"Bundle payload is missing: {artifact}")
            actual = sha256_file(artifact)
            if actual != manifest.get(key):
                raise BundleIntegrityError(
                    f"{filename} does not match MANIFEST.json: "
                    f"expected {manifest.get(key)}, got {actual}"
                )

        bundle = joblib.load(bundle_path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Unexpected bundle type: {type(bundle)!r}")
        bundle.manifest = manifest
        bundle.verify_structure()
        return bundle
