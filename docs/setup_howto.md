# Setup How-To

Complete instructions for setting up the development environment,
configuring secrets, and running each pipeline stage.

## Prerequisites

- Python 3.10 or higher
- Git
- AWS CLI (for OpenAQ data download; no account needed — anonymous access)
- Make (optional but recommended; on Windows use Git Bash or WSL)

## 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/aqi-awareness-assistant.git
cd aqi-awareness-assistant
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Secrets

Copy the example env file and fill in your OpenAQ API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAQ_API_KEY=<your key from https://explore.openaq.org/>
```

> **Never commit `.env` — it is in `.gitignore`.**

Register for a free OpenAQ API key at [explore.openaq.org](https://explore.openaq.org/).

## 5. Configure AWS CLI (Anonymous)

No AWS account is needed.  Test anonymous access:

```bash
aws s3 ls --no-sign-request s3://openaq-data-archive/ --region us-east-1
```

## 6. Set Up Pre-Commit Hooks

```bash
pre-commit install         # installs commit-time hooks
pre-commit install --hook-type pre-push  # installs the logbook check on push
```

## 7. Pull Data with DVC

After the DVC remote has been configured by the team:

```bash
dvc pull
```

## 8. Run the Pipeline

```bash
# Full pipeline
dvc repro

# Single stage (example)
dvc repro hourly_to_daily
```

## 9. Run Tests

```bash
pytest -q
```

Or with the Makefile:

```bash
make test
```

Tests are grouped by marker so CI can run the parts that do not need the DVC
remote:

| Marker | Meaning | CI job |
|---|---|---|
| *(none)* / `slow` | fast unit tests on synthetic data | `unit` |
| `service` | FastAPI contract, 1e-6 offline parity, p95 latency | `service` |
| `model` | bundle integrity, tamper detection, metamorphic and determinism checks | `model-gate` |
| `data` | needs the full DVC-managed splits (`dvc pull`) | `dvc-consistency` |
| `registry` | registry parity in a throwaway SQLite store | `registry` |

`service` and `model` run on a clean checkout: they read the committed
`tests/fixtures/val_snapshot.parquet` (hash-pinned in `tests/test_service.py`)
and `artifacts/champion/`, so no remote credentials are required. Reproduce
those jobs locally with:

```bash
pytest -m "not model and not data and not registry and not slow and not service"
pytest -m service tests/test_service.py
pytest -m "model and not data"
python scripts/audit_integrity.py --bundle-only
python scripts/verify_registry.py
```

## 10. Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open http://localhost:5000 in your browser
```

## 11. Close a Step (Logging Protocol)

After finishing a unit of work:

```bash
make step-close STEP=002 TITLE="Station discovery" PHASE=1
```

This creates the logbook entry, rebuilds the index, and checks completeness.
Then complete the TODO sections in `docs/logbook/002-*.md` and commit.

## 12. Continuous Integration

`.github/workflows/ci.yml` runs seven jobs on every push and pull request to
`main`:

| Job | What it proves |
|---|---|
| `lint` | `black --check .` and `ruff check .` |
| `unit` | fast tests, no DVC data required |
| `service` | service contract, 1e-6 offline parity, p95 latency budgets (committed snapshot) |
| `model-gate` | bundle hashes, tamper detection, metamorphic and determinism checks; `audit_integrity.py --bundle-only` |
| `registry` | `models:/…@champion` reproduces the bundle within `1e-6` in a temporary SQLite store |
| `docker` | image builds and `scripts/test_container.sh` passes |
| `dvc-consistency` | full-data `dvc status`, `data`-marked tests and the 64-check audit; runs only on `main` when `DAGSHUB_*` secrets exist |

`cd.yml` builds and pushes the container to GHCR, then runs a Trivy scan with
`exit-code: "1"` so a HIGH/CRITICAL finding fails the build.

### Required status checks

Configure branch protection on `main` (Settings → Branches → Branch protection
rules) with **Require status checks to pass before merging** and select the six
always-available jobs:

```
Lint & Code Style
Unit Tests
Service Contract & Latency
Model Gate & Bundle Integrity
Registry Parity
Container Build & Smoke
```

`DVC Consistency (full data)` is intentionally **not** required: it is skipped
when the DVC remote credentials are absent, and a skipped check would block
merges on forks and fresh clones. Keep `Require branches to be up to date
before merging` enabled so the model gate always runs against the current
`main`.

## 13. Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) with the step number:

```
feat(step-001): implement logging scaffold and repo structure
docs(step-001): logbook entry 001
fix(step-003): handle missing PM2.5 in AQI computation
```

## Troubleshooting

- **`ModuleNotFoundError: src`** — run commands from the repo root, or install with `pip install -e .`
- **DVC remote not configured** — see `docs/decisions.md` for the remote setup decision and follow the team's documented procedure.
- **Pre-commit fails on Windows** — ensure the `.venv\Scripts` folder is on your PATH when running `pre-commit`.
