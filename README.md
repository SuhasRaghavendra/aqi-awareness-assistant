# AI-Based Air Quality Awareness Assistant — MLOps Milestone 1

> Predicting next-day AQI for Indian cities from OpenAQ station data and
> Open-Meteo ERA5 weather, using a fully reproducible MLOps pipeline.

## Problem Statement

Air quality in Indian cities is a serious public health concern.
This project builds the ML foundation for a system that predicts **next-day AQI**
(CPCB method) and categorises it into health risk bands (Good → Severe).

## Data Sources & Attribution

| Source | Role | License |
|---|---|---|
| [OpenAQ](https://openaq.org/) archive (`openaq-data-archive` S3, `us-east-1`) | Primary: hourly pollutant readings for Indian CPCB stations | Varies by data provider (see `docs/data_card.md`) |
| [Open-Meteo](https://open-meteo.com/) Historical Weather API (ERA5) | Enrichment: weather variables per station coordinate | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Attribution: Weather data from [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

## Pipeline Overview

```
OpenAQ S3 ──┐                        ┌── train.py ──── models/
            ├── validate ── daily ──── merge ── clean ── features ──┤
Open-Meteo ─┘                                                       └── evaluate
                                                                         ↓
                                                              reports/metrics/test_metrics.json
```

All stages are managed by **DVC** (`dvc repro`), tracked by **MLflow**, and
every step is documented in the **logbook** (`docs/logbook/`).

## Quick Start

```bash
# 1. Clone and create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAQ API key
cp .env.example .env
# Edit .env and fill in OPENAQ_API_KEY=<your key>

# 4. Pull data (after DVC remote is configured)
# dvc pull

# 5. Run the full pipeline
# dvc repro

# 6. Run tests
pytest -q

# 7. Launch MLflow UI
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

See `docs/setup_howto.md` for detailed setup instructions.

## Repository Structure

```
aqi-awareness-assistant/
├── configs/          # CPCB breakpoints, station list, festival dates
├── data/             # DVC-tracked: raw, interim, processed
├── docs/
│   ├── logbook/      # Step-by-step change log (one .md per step)
│   ├── CHANGELOG.md  # Human-readable release notes
│   ├── decisions.md  # Design decision register (ADR-style)
│   └── data_card.md  # Dataset documentation
├── models/           # DVC-tracked: trained model files
├── notebooks/        # EDA notebooks (read-only; logic lives in src/)
├── reports/          # Figures, metrics JSON, step reports
├── scripts/          # log_step.py, build_logbook_index.py, check_logbook.py
├── src/
│   ├── data/         # Pipeline stages: discover, download, ingest, validate, …
│   ├── features/     # AQI computation, feature engineering
│   ├── models/       # Baselines, train, evaluate, compare_runs
│   └── utils/        # logging_utils, stage_stats, mlflow_utils
└── tests/            # pytest test suite
```

## How the Logbook Works

Every step is documented in `docs/logbook/NNN-<slug>.md` following the template
in `docs/logbook/_TEMPLATE.md`. Steps are created automatically by running:

```bash
make step-close STEP=001 TITLE="Repo scaffold and logging scaffold" PHASE=2
```

The index of all entries is at `docs/logbook/INDEX.md`.

## Results

> _To be filled in once training and evaluation are complete (Phase 6–8)._

## Team

| Name | Role |
|---|---|
| Student A | Logging scaffold, data discovery, DVC, baselines, MLflow |
| Student B | log_step.py, weather download, EDA, features, boosting, report |

## Limitations

- Station coverage varies; some cities have < 3 years of data.
- ERA5 is gridded reanalysis, not station-level weather.
- 1-day forecast horizon only; multi-day horizon is future work.
- No traffic, fire, or emissions data.

## License

Code: MIT.  Data: see `docs/data_card.md` for per-source licenses.
