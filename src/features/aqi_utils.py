"""
src/features/aqi_utils.py
-------------------------
CPCB (Central Pollution Control Board, India) National Air Quality Index (NAQI)
sub-index and overall AQI computation utilities.

Reference:
  Central Pollution Control Board (2014), "National Air Quality Index"
  Official breakpoints: configs/cpcb_breakpoints.yaml

Rules:
  1. Sub-index calculation:
     I = I_low + [(I_high - I_low) / (C_high - C_low)] * (C - C_low)
  2. AQI validity criteria:
     - At least min_subindices (default 3) valid pollutant sub-indices.
     - At least one particulate matter sub-index (PM2.5 or PM10) required.
  3. Overall AQI:
     - AQI = max(sub_indices)
     - Dominant pollutant = argmax(sub_indices)
  4. Categories:
     - 0-50: Good
     - 51-100: Satisfactory
     - 101-200: Moderate
     - 201-300: Poor
     - 301-400: Very Poor
     - 401-500: Severe
     - >500: Severe
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


CATEGORY_BANDS = [
    (0.0, 50.0, "Good"),
    (50.0001, 100.0, "Satisfactory"),
    (100.0001, 200.0, "Moderate"),
    (200.0001, 300.0, "Poor"),
    (301.0001, 400.0, "Very Poor"),
    (400.0001, 500.0, "Severe"),
]


def load_cpcb_breakpoints(
    config_path: str | Path = "configs/cpcb_breakpoints.yaml",
) -> dict[str, Any]:
    """Load CPCB pollutant breakpoint tables from YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        # Fallback to root or default location
        cfg_file = Path(__file__).resolve().parent.parent.parent / "configs" / "cpcb_breakpoints.yaml"

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def calc_sub_index(
    pollutant: str,
    concentration: float | None,
    breakpoints_table: dict[str, Any] | None = None,
) -> float | None:
    """
    Calculate the CPCB sub-index for a single pollutant concentration using linear interpolation.

    Parameters:
      pollutant: 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3', 'pb'
      concentration: float concentration in standard CPCB units (mg/m³ for CO, µg/m³ for others)
      breakpoints_table: loaded breakpoint table from cpcb_breakpoints.yaml

    Returns:
      sub_index: rounded to nearest integer (or float), or None if invalid/uncomputable.
    """
    if concentration is None or math.isnan(concentration) or concentration < 0:
        return None

    if breakpoints_table is None:
        breakpoints_table = load_cpcb_breakpoints()

    pol_key = pollutant.lower().replace(".", "").replace("-", "").strip()
    if pol_key not in breakpoints_table:
        return None

    bp_list = breakpoints_table[pol_key].get("breakpoints", [])
    if not bp_list:
        return None

    # Exact zero concentration
    if concentration == 0.0:
        return 0.0

    for c_low, c_high, i_low, i_high in bp_list:
        if c_low <= concentration <= c_high:
            if c_high == c_low:
                return float(i_low)
            sub_idx = i_low + ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low)
            return round(sub_idx, 1)

    # Exceeds highest breakpoint
    max_c_low, max_c_high, max_i_low, max_i_high = bp_list[-1]
    if concentration > max_c_high:
        # Extrapolate slope above highest bracket or cap at 500
        slope = (max_i_high - max_i_low) / (max_c_high - max_c_low) if max_c_high > max_c_low else 1.0
        extrapolated = max_i_high + slope * (concentration - max_c_high)
        # Cap at 500 per official CPCB AQI cap guideline
        return min(round(extrapolated, 1), 500.0)

    return None


def aqi_to_category(aqi_val: float | None) -> str | None:
    """Map numeric AQI to official CPCB categorical band."""
    if aqi_val is None or math.isnan(aqi_val) or aqi_val < 0:
        return None

    if aqi_val <= 50.0:
        return "Good"
    elif aqi_val <= 100.0:
        return "Satisfactory"
    elif aqi_val <= 200.0:
        return "Moderate"
    elif aqi_val <= 300.0:
        return "Poor"
    elif aqi_val <= 400.0:
        return "Very Poor"
    else:
        return "Severe"


def calc_overall_aqi(
    sub_indices: dict[str, float | None],
    min_subindices: int = 3,
    require_pm: bool = True,
) -> tuple[float | None, str | None, str | None]:
    """
    Compute overall AQI and dominant pollutant according to CPCB validity rules.

    Parameters:
      sub_indices: dict mapping pollutant name to calculated sub-index
      min_subindices: minimum number of valid pollutant sub-indices required (CPCB rule: >= 3)
      require_pm: whether at least one of PM2.5 or PM10 is required (CPCB rule: True)

    Returns:
      (aqi, aqi_category, dominant_pollutant) or (None, None, None) if criteria are not met.
    """
    valid_subs = {
        k.lower(): v
        for k, v in sub_indices.items()
        if v is not None and not math.isnan(v) and v >= 0
    }

    # CPCB Rule 1: Minimum number of sub-indices
    if len(valid_subs) < min_subindices:
        return None, None, None

    # CPCB Rule 2: At least one particulate matter pollutant must be present
    if require_pm:
        has_pm = ("pm25" in valid_subs) or ("pm10" in valid_subs)
        if not has_pm:
            return None, None, None

    # CPCB Rule 3: AQI is the maximum of the valid sub-indices
    dominant_pol = max(valid_subs, key=lambda k: (valid_subs[k], k))
    overall_aqi = round(valid_subs[dominant_pol], 1)
    category = aqi_to_category(overall_aqi)

    return overall_aqi, category, dominant_pol
