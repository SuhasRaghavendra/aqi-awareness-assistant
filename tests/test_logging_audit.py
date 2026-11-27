"""Unit and integration tests for Pillar 2 structured JSON & JSONL audit logging."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.service.app import app
from src.service.monitoring import ServiceMonitor, monitor
from src.utils.logging_utils import get_audit_logger, log_audit_event, setup_logging


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_prediction_and_http_audit_events_written_to_jsonl(tmp_path: Path, monkeypatch) -> None:
    audit_file = tmp_path / "predictions_audit.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(audit_file))
    monitor.reset()

    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"city": "Delhi", "date": "2025-01-05", "aqi": 210.0},
            headers={"X-Request-ID": "req-audit-123"},
        )
        assert resp.status_code == 200

    records = _read_jsonl(audit_file)
    events = {r["event"] for r in records}
    assert "prediction_served" in events
    assert "http_request" in events

    pred_event = next(r for r in records if r["event"] == "prediction_served")
    assert pred_event["request_id"] == "req-audit-123"
    assert pred_event["city"] == "Delhi"
    assert pred_event["input_aqi"] == 210.0
    assert pred_event["model_name"] == "residual_lightgbm_monotone"
    assert pred_event["alias"] == "champion"
    assert "interval_80" in pred_event
    assert "latency_ms" in pred_event

    http_event = next(
        r for r in records if r["event"] == "http_request" and r.get("path") == "/predict"
    )
    assert http_event["request_id"] == "req-audit-123"
    assert http_event["status_code"] == 200
    assert http_event["latency_ms"] >= 0


def test_drift_mae_and_retrain_audit_events_written(tmp_path: Path, monkeypatch) -> None:
    audit_file = tmp_path / "alerts_audit.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(audit_file))

    m = ServiceMonitor(psi_window=50, mae_window=14, val_mae_baseline=10.0)
    m.configure(
        reference_stats={"features": {"aqi": {"mean": 80.0, "std": 20.0}}},
        val_mae_baseline=10.0,
    )

    # Trigger RED input_drift + retrain_triggered (12 shifted samples) + mae_alert (large errors)
    for day in range(1, 13):
        m.observe(
            payload={"aqi": 350.0},
            city="Delhi",
            date=f"2025-02-{day:02d}",
            target_date=f"2025-02-{day + 1:02d}",
            aqi_pred=100.0,
        )

    records = _read_jsonl(audit_file)
    events = {r["event"] for r in records}
    assert "input_drift" in events
    assert "mae_alert" in events
    assert "retrain_triggered" in events


def test_disable_jsonl_file_with_empty_env_var(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTION_LOG_PATH", "")
    logger = get_audit_logger()
    assert logger.handlers == []
    # Should still succeed cleanly via stdout propagation
    log_audit_event("health_check_ping", ok=True)


def test_setup_logging_backwards_compatibility(tmp_path: Path) -> None:
    stage_logger = setup_logging("test_pillar2_stage", log_dir=tmp_path)
    stage_logger.info("stage_started", extra={"rows": 42})
    jsonl_files = list(tmp_path.glob("test_pillar2_stage_*.jsonl"))
    assert len(jsonl_files) == 1
    records = _read_jsonl(jsonl_files[0])
    assert any(r["message"] == "stage_started" and r["rows"] == 42 for r in records)
