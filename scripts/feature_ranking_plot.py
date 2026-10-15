"""
scripts/feature_ranking_plot.py
--------------------------------
Generates a feature importance (variance-based ranking) figure for train split.
Saved to: reports/figures/features/feature_importance_train.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
})

FIG_DIR = Path("reports/figures/features")
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("data/processed/feature_list.json") as f:
    fl = json.load(f)
feature_names = fl["feature_names"]

train = pd.read_parquet("data/processed/train.parquet")
X_train = train[feature_names].select_dtypes(include="number")
feature_names_num = list(X_train.columns)

# Use coefficient of variation (std/mean abs) as a proxy for variance-based importance
cv_scores = {}
for col in feature_names_num:
    col_data = X_train[col].dropna()
    if col_data.std() > 0 and col_data.abs().mean() > 0:
        cv_scores[col] = col_data.std() / (col_data.abs().mean() + 1e-9)
    else:
        cv_scores[col] = 0.0

sorted_feats = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
top_n = 30
top_feats = sorted_feats[:top_n]
names = [f[0] for f in top_feats]
scores = [f[1] for f in top_feats]

# Color by feature group
def get_color(name: str) -> str:
    if "lag" in name or "rolling" in name or "diff" in name:
        return "#4dd0e1"  # cyan — temporal
    elif "temperature" in name or "humidity" in name or "wind" in name or "precipitation" in name:
        return "#80cbc4"  # teal — weather
    elif "city_" in name or "station" in name:
        return "#ce93d8"  # purple — spatial
    elif "month" in name or "dow" in name or "season" in name or "weekend" in name or "festival" in name or "sin_" in name or "cos_" in name:
        return "#ffcc02"  # yellow — calendar
    elif "pm25" in name or "pm10" in name or "no2" in name or "aqi" in name:
        return "#ef9a9a"  # red — pollutant
    else:
        return "#a5d6a7"  # green — other

colors = [get_color(n) for n in names]

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(top_n), scores[::-1], color=colors[::-1], alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names[::-1], fontsize=8)
ax.set_xlabel("Coefficient of Variation (training set)")
ax.set_title(f"Fig — Top {top_n} Features by Variability (Training Set Proxy)")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#ef9a9a", label="Pollutant"),
    Patch(facecolor="#4dd0e1", label="Temporal (lags/rolling)"),
    Patch(facecolor="#80cbc4", label="Weather"),
    Patch(facecolor="#ffcc02", label="Calendar"),
    Patch(facecolor="#ce93d8", label="Spatial"),
    Patch(facecolor="#a5d6a7", label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = FIG_DIR / "feature_importance_train.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved {out}")
print(f"Total features: {len(feature_names_num)} numeric, showing top {top_n}")
