"""Serving-side champion bundle for the AQI predictor (Phase 2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm  # noqa: F401  # import before pandas/pyarrow on Windows (ADR D-015)
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.models.model_risk import rebuild_features

BUNDLE_FILENAME = "bundle.joblib"
MANIFEST_FILENAME = "MANIFEST.json"
MANIFEST_SHA_FILENAME = "MANIFEST.sha256"
REFERENCE_STATS_FILENAME = "reference_stats.json"
HISTORY_SEED_FILENAME = "history_seed.parquet"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        scaled, scaled_df = self._scaled(frame)
        if self.input_kind == "residual_scaled_df":
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
        frame = rebuild_features(
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

    def save(self, path: str | Path, include_payload: bool = True) -> None:
        """Persist the bundle, MANIFEST.json and MANIFEST.sha256.

        Text artifacts are written with LF line endings on every platform so the
        SHA-256 digests in MANIFEST.json describe the same bytes in every
        checkout (Windows, Linux, CI).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if include_payload:
            self.history_seed.to_parquet(path / HISTORY_SEED_FILENAME, index=False)
            with (path / REFERENCE_STATS_FILENAME).open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(self.reference_stats, fh, indent=2)
        joblib.dump(self, path / BUNDLE_FILENAME)
        manifest_bytes = json.dumps(self.manifest, indent=2).encode("utf-8")
        (path / MANIFEST_FILENAME).write_bytes(manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        (path / MANIFEST_SHA_FILENAME).write_text(f"{digest}\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> AqiBundle:
        """Load a bundle directory, raising when the artifact is missing."""
        bundle_path = Path(path) / BUNDLE_FILENAME
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle artifact is missing: {bundle_path}")
        bundle = joblib.load(bundle_path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Unexpected bundle type: {type(bundle)!r}")
        return bundle


class AqiBundlePyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc adapter used by the model registry."""

    def load_context(self, context) -> None:
        self.bundle = AqiBundle.load(Path(context.artifacts["bundle_dir"]))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.bundle.predict(model_input)
