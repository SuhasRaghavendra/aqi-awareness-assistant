"""
src/config.py
Central config loader: reads params.yaml and .env.
All scripts import from here — no hard-coded paths or magic numbers.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
import os

# Project root is two levels up from this file (src/config.py → src → root)
ROOT = Path(__file__).resolve().parents[1]

# Load .env (silently skip if missing — CI environments may not have it)
load_dotenv(ROOT / ".env", override=False)


def load_params(params_path: Path | None = None) -> dict:
    """
    Load params.yaml.

    Args:
        params_path: Path to params.yaml. Defaults to project root.

    Returns:
        Parsed YAML as a nested dict.
    """
    if params_path is None:
        params_path = ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_openaq_api_key() -> str:
    """
    Return the OpenAQ API key from the environment.

    Raises:
        EnvironmentError: If the key is not set.
    """
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENAQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return key


# Convenience path constants
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
DOCS_DIR = ROOT / "docs"
