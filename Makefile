.PHONY: help setup install test lint fmt check repro step-close logbook-index logbook-check clean-pyc

# ── defaults ──────────────────────────────────────────────────────────────────
STEP   ?= 001
TITLE  ?= "untitled step"
PHASE  ?= 2
MLFLOW ?=

# ── help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AQI Awareness Assistant — Makefile targets"
	@echo "==========================================="
	@echo "  make setup          Create .venv and install requirements"
	@echo "  make install        pip install -r requirements.txt"
	@echo "  make test           Run pytest"
	@echo "  make lint           Run ruff + black --check"
	@echo "  make fmt            Auto-format with ruff fix + black"
	@echo "  make repro          dvc repro (full pipeline)"
	@echo "  make logbook-index  Rebuild docs/logbook/INDEX.md"
	@echo "  make logbook-check  Check logbook completeness"
	@echo "  make step-close     STEP=NNN TITLE='...' PHASE=N [MLFLOW=id]"
	@echo "  make clean-pyc      Remove __pycache__ / *.pyc"
	@echo ""

# ── environment ───────────────────────────────────────────────────────────────
setup:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt
	@echo "✅  Virtual environment ready. Activate with: .venv\\Scripts\\activate"

install:
	pip install -r requirements.txt

# ── quality ───────────────────────────────────────────────────────────────────
test:
	pytest -q
	python scripts/check_logbook.py

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

fmt:
	ruff check --fix src/ scripts/ tests/
	black src/ scripts/ tests/

# ── pipeline ──────────────────────────────────────────────────────────────────
repro:
	dvc repro

# ── logging / logbook ─────────────────────────────────────────────────────────
logbook-index:
	python scripts/build_logbook_index.py

logbook-check:
	python scripts/check_logbook.py

step-close:
	@echo "-- Step-Close Protocol: Step $(STEP) --"
	python scripts/log_step.py \
		--step $(STEP) \
		--title $(TITLE) \
		--phase $(PHASE) \
		$(if $(MLFLOW),--mlflow-run $(MLFLOW),)
	python scripts/build_logbook_index.py
	python scripts/check_logbook.py
	@echo "-- Step $(STEP) closed. --"

# ── misc ──────────────────────────────────────────────────────────────────────
clean-pyc:
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
