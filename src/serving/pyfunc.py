"""MLflow pyfunc adapter for the AQI bundle.

Kept in a separate module so the FastAPI serving image does not import MLflow
(or any of its transitive training dependencies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.serving.bundle import AqiBundle


class AqiBundlePyfunc(mlflow.pyfunc.PythonModel):
    """Load an ``AqiBundle`` from the logged artifact directory."""

    def load_context(self, context) -> None:
        self.bundle = AqiBundle.load(Path(context.artifacts["bundle_dir"]))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.bundle.predict(model_input)
