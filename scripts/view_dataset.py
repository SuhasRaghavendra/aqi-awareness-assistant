"""
scripts/view_dataset.py
-----------------------
Quick dataset inspection utility for parquet splits and interim tables.

Usage:
    python scripts/view_dataset.py --split train --head 5
    python scripts/view_dataset.py --split val --summary
    python scripts/view_dataset.py --split test --html
    python scripts/view_dataset.py --file data/interim/station_day_clean.parquet --head 10
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd


SPLIT_PATHS = {
    "train": Path("data/processed/train.parquet"),
    "val": Path("data/processed/val.parquet"),
    "test": Path("data/processed/test.parquet"),
    "clean": Path("data/interim/station_day_clean.parquet"),
    "merged": Path("data/interim/station_day.parquet"),
    "daily": Path("data/interim/pollutants_daily.parquet"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parquet datasets and splits.")
    parser.add_argument(
        "--split",
        choices=list(SPLIT_PATHS.keys()),
        default="train",
        help="Which dataset split to inspect (default: train)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Custom parquet file path (overrides --split)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to display (default: 5)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display summary statistics and missing value counts",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Export first 100 rows to interactive HTML and open in browser",
    )

    args = parser.parse_args()

    target_path = args.file if args.file is not None else SPLIT_PATHS[args.split]

    if not target_path.exists():
        print(f"[ERROR] Target file does not exist: {target_path}")
        return

    print(f"\n=======================================================")
    print(f"LOADING DATASET: {target_path}")
    print(f"=======================================================")

    df = pd.read_parquet(target_path)

    print(f"Shape:             {df.shape[0]:,} rows x {df.shape[1]} columns")
    if "date" in df.columns:
        print(f"Date Range:        {df['date'].min()} to {df['date'].max()}")
    if "city" in df.columns:
        print(f"Cities:            {list(df['city'].unique())}")
    if "target_aqi" in df.columns:
        print(f"Target AQI Range:  {df['target_aqi'].min():.1f} - {df['target_aqi'].max():.1f} (Mean: {df['target_aqi'].mean():.1f})")
    if "target_category" in df.columns:
        print(f"Target Categories: {df['target_category'].value_counts().to_dict()}")

    print("\n--- SAMPLE COLUMNS & FIRST FEW ROWS ---")
    # Show key columns first if present
    key_cols = [c for c in ["date", "city", "station_name", "location_id", "aqi", "target_aqi", "target_category", "pm25", "temperature_2m_mean"] if c in df.columns]
    remaining = [c for c in df.columns if c not in key_cols]
    display_cols = key_cols + remaining[:5]
    print(df[display_cols].head(args.head).to_string())

    if args.summary:
        print("\n--- MISSING VALUES SUMMARY (TOP 10) ---")
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if nulls.empty:
            print("No missing values found across all columns! (Clean dataset)")
        else:
            print(nulls.head(10).to_string())

    if args.html:
        html_file = Path("reports/dataset_preview.html")
        html_file.parent.mkdir(parents=True, exist_ok=True)
        # Style nicely
        styled_html = f"""
        <html>
        <head>
            <title>Dataset Preview: {target_path.name}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; background: #f8fafc; }}
                h2 {{ color: #0f172a; }}
                p {{ color: #475569; }}
                table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
                th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
                th {{ background-color: #3b82f6; color: white; position: sticky; top: 0; }}
                tr:hover {{ background-color: #f1f5f9; }}
            </style>
        </head>
        <body>
            <h2>Dataset Preview: {target_path}</h2>
            <p>Showing first 100 rows out of {len(df):,} total records ({df.shape[1]} columns).</p>
            {df.head(100).to_html(classes='table', index=False)}
        </body>
        </html>
        """
        html_file.write_text(styled_html, encoding="utf-8")
        print(f"\n[HTML Exported] Saved to: {html_file.resolve()}")
        webbrowser.open(html_file.as_uri())


if __name__ == "__main__":
    main()
