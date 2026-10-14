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

## 12. Commit Message Convention

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
