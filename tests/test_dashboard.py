"""Tests for Pillar 3 performance dashboard, Prometheus alerts, and Grafana provisioning."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.service.app import app
from src.service.monitoring import monitor

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_endpoint_serves_html() -> None:
    monitor.reset()
    with TestClient(app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert root_resp.json()["dashboard_url"] == "/dashboard"

        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "Performance &amp; Drift Dashboard" in body
        assert "/monitoring/summary" in body
        assert "/metadata" in body
        assert "/metrics" in body


def test_monitoring_summary_reports_live_p95_latency() -> None:
    monitor.reset()
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"city": "Delhi", "date": "2025-01-05", "aqi": 195.0},
        )
        assert resp.status_code == 200

        summary = client.get("/monitoring/summary").json()
        assert summary["total_predictions"] == 1
        assert summary["p95_latency_ms"] is not None
        assert summary["p95_latency_ms"] > 0


def test_prometheus_and_grafana_configs_are_valid() -> None:
    prom_cfg = yaml.safe_load((REPO_ROOT / "configs/prometheus.yml").read_text(encoding="utf-8"))
    assert "/etc/prometheus/prometheus_alerts.yml" in prom_cfg["rule_files"]

    alerts_cfg = yaml.safe_load(
        (REPO_ROOT / "configs/prometheus_alerts.yml").read_text(encoding="utf-8")
    )
    rules = alerts_cfg["groups"][0]["rules"]
    alert_names = {r["alert"] for r in rules}
    assert {
        "AQIHighFeatureDriftPSI",
        "AQIHighPredictionDriftPSI",
        "AQIHighRollingMAE",
        "AQIHighLatencyP95",
        "AQIFallbackSpike",
    }.issubset(alert_names)

    ds_cfg = yaml.safe_load(
        (REPO_ROOT / "configs/grafana/provisioning/datasources/prometheus.yml").read_text(
            encoding="utf-8"
        )
    )
    assert ds_cfg["datasources"][0]["url"] == "http://prometheus:9090"

    prov_cfg = yaml.safe_load(
        (REPO_ROOT / "configs/grafana/provisioning/dashboards/dashboards.yml").read_text(
            encoding="utf-8"
        )
    )
    assert prov_cfg["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    dash = json.loads(
        (REPO_ROOT / "configs/grafana/dashboards/aqi_mlops_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    assert dash["uid"] == "aqi-mlops-overview"
    assert len(dash["panels"]) >= 8

    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    prom_vols = compose["services"]["prometheus"]["volumes"]
    graf_vols = compose["services"]["grafana"]["volumes"]
    assert any("prometheus_alerts.yml" in v for v in prom_vols)
    assert any("configs/grafana/provisioning" in v for v in graf_vols)
    assert any("configs/grafana/dashboards" in v for v in graf_vols)
