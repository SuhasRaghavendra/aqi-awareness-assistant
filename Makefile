.PHONY: help repro test lint format step-close clean-logs

help:
	@echo "Available targets:"
	@echo "  repro          Run full DVC pipeline"
	@echo "  test           Run pytest + logbook check"
	@echo "  lint           Run ruff + black --check"
	@echo "  format         Auto-format with black + ruff --fix"
	@echo "  step-close     Create logbook entry (STEP=NNN TITLE=...)"
	@echo "  clean-logs     Remove log files"

repro:
	dvc repro

test:
	pytest -q
	python scripts/check_logbook.py

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

step-close:
	@if [ -z "$(STEP)" ]; then echo "Usage: make step-close STEP=001 TITLE='...' PHASE=1"; exit 1; fi
	python scripts/log_step.py --step $(STEP) --title "$(TITLE)" --phase $(PHASE)
	python scripts/build_logbook_index.py
	@echo "Step $(STEP) closed."

clean-logs:
	del /Q logs\*.jsonl 2>NUL || true
