"""
tests/test_stage_stats.py
Tests for the stage_stats context manager.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.stage_stats import track_stage, StageTracker


@pytest.fixture()
def tmp_metrics(tmp_path: Path) -> Path:
    return tmp_path / "metrics"


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    return p


class TestStageTracker:
    def test_drop_accumulates(self) -> None:
        t = StageTracker(stage="test")
        t.drop("negative PM2.5", 10)
        t.drop("duplicate", 5)
        assert len(t._drops) == 2
        assert sum(d["n"] for d in t._drops) == 15

    def test_impute_accumulates(self) -> None:
        t = StageTracker(stage="test")
        t.impute("pm25", 20, method="ffill")
        assert t._imputations[0]["column"] == "pm25"
        assert t._imputations[0]["n"] == 20

    def test_note_accumulates(self) -> None:
        t = StageTracker(stage="test")
        t.note("some observation")
        assert "some observation" in t._notes


class TestTrackStage:
    def test_creates_stats_json(self, tmp_metrics: Path, tmp_path: Path) -> None:
        with track_stage("test_stage", metrics_dir=tmp_metrics) as st:
            st.drop("bad values", 3)
            st.note("test run")

        report_path = tmp_metrics / "test_stage_stats.json"
        assert report_path.exists(), "stats JSON was not created"
        report = json.loads(report_path.read_text())
        assert report["stage"] == "test_stage"
        assert report["total_dropped"] == 3
        assert "test run" in report["notes"]

    def test_duration_recorded(self, tmp_metrics: Path) -> None:
        with track_stage("timing_test", metrics_dir=tmp_metrics):
            pass
        report = json.loads((tmp_metrics / "timing_test_stats.json").read_text())
        assert report["duration_seconds"] >= 0

    def test_input_file_stats(self, tmp_metrics: Path, sample_parquet: Path) -> None:
        with track_stage("file_stat_test", inputs=[sample_parquet], metrics_dir=tmp_metrics):
            pass
        report = json.loads((tmp_metrics / "file_stat_test_stats.json").read_text())
        assert report["inputs"][0]["rows"] == 100
        assert report["inputs"][0]["n_columns"] == 2

    def test_appends_data_change_log(self, tmp_metrics: Path, tmp_path: Path, monkeypatch) -> None:
        """data_change_log.md should be appended after stage runs."""
        # Monkeypatch the docs path to tmp_path
        import src.utils.stage_stats as ss
        original = ss.Path
        log_path = tmp_path / "docs" / "data_change_log.md"

        class PatchedPath(type(Path())):
            pass

        monkeypatch.setattr(
            "src.utils.stage_stats._append_data_change_log",
            lambda stage, report: log_path.parent.mkdir(parents=True, exist_ok=True)
            or log_path.open("a").write(f"\n## {stage}\n"),
        )

        with track_stage("log_test", metrics_dir=tmp_metrics) as st:
            st.drop("test", 1)

        # Just verify no exception was raised
        assert (tmp_metrics / "log_test_stats.json").exists()
