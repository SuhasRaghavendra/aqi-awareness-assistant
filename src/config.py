"""
src/config.py
-------------
Central configuration loader.

Reads params.yaml and (optionally) .env, exposing typed settings to all
pipeline stages.  Every src/ module imports from here — no hard-coded paths
or magic numbers anywhere else.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_params(params_path: Path | None = None) -> dict[str, Any]:
    """Load and cache params.yaml.

    Parameters
    ----------
    params_path:
        Override the default ``<repo>/params.yaml`` location (useful in tests).
    """
    path = params_path or (_REPO_ROOT / "params.yaml")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_env(env_path: Path | None = None) -> None:
    """Load .env into ``os.environ`` (no-op if the file does not exist)."""
    load_dotenv(env_path or (_REPO_ROOT / ".env"), override=False)


def get_openaq_api_key() -> str:
    """Return the OpenAQ API key from the environment.

    Raises
    ------
    EnvironmentError
        If ``OPENAQ_API_KEY`` is not set.
    """
    load_env()
    key = os.getenv("OPENAQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAQ_API_KEY is not set.  Copy .env.example → .env and add your key."
        )
    return key
