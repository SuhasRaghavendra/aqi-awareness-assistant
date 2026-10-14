"""
tests/test_stage_stats.py
-------------------------
Unit tests for src/utils/stage_stats.py and src/utils/logging_utils.py.

Run with: pytest -q tests/test_stage_stats.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.stage_stats import (
    StageStats,
    _null_pct,
    _params_hash,
    track_stage,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_df(rows: int = 100, cols: list[str] | None = None) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    columns = cols or ["pm25", "pm10", "no2", "aqi"]
    data = {c: rng.uniform(0, 300, size=rows) for c in columns}
    data["pm25"][:10] = float("nan")  # 10 % nulls in pm25
    return pd.DataFrame(data)


# ── StageStats unit tests ──────────────────────────────────────────────────────


class TestStageStats:
    def test_drop_accumulates(self) -> None:
        st = StageStats(stage="test", inputs=[], outputs=[])
        st.drop(reason="negative PM2.5", n=50)
        st.drop(reason="duplicate station-day", n=10)
        assert st.total_dropped == 60
        assert len(st._drops) == 2

    def test_record_input_frame(self) -> None:
        st = StageStats(stage="test", inputs=[], outputs=[])
        df = _make_df(200)
        st.record_input_frame(df)
        assert st.rows_in == 200
        assert "pm25" in st._input_frame_stats["null_pct"]
        # pm25 has 10 nulls → 5%
        assert st._input_frame_stats["null_pct"]["pm25"] == pytest.approx(5.0, abs=0.1)

    def test_record_output_frame(self) -> None:
        st = StageStats(stage="test", inputs=[], outputs=[])
        df_in = _make_df(200, ["pm25", "pm10", "no2"])
        df_out = _make_df(180, ["pm25", "pm10", "aqi"])
        st.record_input_frame(df_in)
        st.record_output_frame(df_out)
        assert st.rows_out == 180
        assert "aqi" in st.cols_added
        assert "no2" in st.cols_removed
        assert "pm25" not in st.cols_added
        assert "pm25" not in st.cols_removed

    def test_cols_added_removed(self) -> None:
        st = StageStats(stage="test", inputs=[], outputs=[])
        st._cols_before = ["a", "b", "c"]
        st._cols_after = ["b", "c", "d", "e"]
        assert sorted(st.cols_added) == ["d", "e"]
        assert st.cols_removed == ["a"]

    def test_to_dict_structure(self) -> None:
        st = StageStats(stage="clean", inputs=["data/interim/x.parquet"], outputs=[])
        st.start_time = "2026-10-14T00:00:00+00:00"
        st.end_time = "2026-10-14T00:01:00+00:00"
        st.duration_s = 60.0
        st.git_commit = "abc1234"
        st.params_hash = "xyz"
        d = st.to_dict()
        assert d["stage"] == "clean"
        assert "inputs" in d
        assert "outputs" in d
        assert "frame_stats" in d
        assert "drops" in d


# ── _null_pct unit tests ───────────────────────────────────────────────────────


class TestNullPct:
    def test_empty_df(self) -> None:
        df = pd.DataFrame({"a": []})
        result = _null_pct(df)
        assert result == {}

    def test_no_nulls(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _null_pct(df)
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_all_nulls(self) -> None:
        import numpy as np
        df = pd.DataFrame({"x": [float("nan"), float("nan")]})
        result = _null_pct(df)
        assert result["x"] == 100.0

    def test_partial_nulls(self) -> None:
        import numpy as np
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0, float("nan")]})
        result = _null_pct(df)
        assert result["x"] == pytest.approx(50.0)


# ── track_stage integration test ──────────────────────────────────────────────


class TestTrackStage:
    def test_writes_stats_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """track_stage writes a JSON stats file on exit."""
        import src.utils.stage_stats as ss

        metrics_dir = tmp_path / "reports" / "metrics"
        change_log = tmp_path / "docs" / "data_change_log.md"
        monkeypatch.setattr(ss, "_METRICS_DIR", metrics_dir)
        monkeypatch.setattr(ss, "_CHANGE_LOG", change_log)

        df_in = _make_df(100)
        df_out = _make_df(90)

        with track_stage("clean", inputs=[], outputs=[]) as st:
            st.record_input_frame(df_in)
            st.drop(reason="missing target", n=10)
            st.record_output_frame(df_out)

        stats_file = metrics_dir / "clean_stats.json"
        assert stats_file.exists(), "stats JSON not written"
        data = json.loads(stats_file.read_text())
        assert data["stage"] == "clean"
        assert data["total_dropped"] == 10
        assert data["frame_stats"]["input"]["rows"] == 100
        assert data["frame_stats"]["output"]["rows"] == 90

    def test_appends_change_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """track_stage appends a section to data_change_log.md."""
        import src.utils.stage_stats as ss

        metrics_dir = tmp_path / "reports" / "metrics"
        change_log = tmp_path / "docs" / "data_change_log.md"
        monkeypatch.setattr(ss, "_METRICS_DIR", metrics_dir)
        monkeypatch.setattr(ss, "_CHANGE_LOG", change_log)

        with track_stage("hourly_to_daily", inputs=[], outputs=[]) as st:
            st.drop(reason="unit parse failure", n=5)

        assert change_log.exists(), "data_change_log.md not created"
        content = change_log.read_text()
        assert "hourly_to_daily" in content
        assert "unit parse failure" in content
        assert "5" in content

    def test_no_exception_on_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """track_stage does not raise even when no frames are recorded."""
        import src.utils.stage_stats as ss

        monkeypatch.setattr(ss, "_METRICS_DIR", tmp_path / "metrics")
        monkeypatch.setattr(ss, "_CHANGE_LOG", tmp_path / "cl.md")

        with track_stage("validate", inputs=[], outputs=[]):
            pass  # deliberately do nothing


# ── logging_utils tests ────────────────────────────────────────────────────────


class TestLoggingUtils:
    def test_setup_logging_creates_jsonl(self, tmp_path: Path) -> None:
        from src.utils.logging_utils import setup_logging, _configured_stages

        stage = "test_stage_jsonl"
        _configured_stages.discard(stage)  # reset idempotency guard
        logger = setup_logging(stage, log_dir=tmp_path, console_level=logging.WARNING)
        logger.info("hello from test")
        logger.warning("a warning")

        jsonl_files = list(tmp_path.glob(f"{stage}_*.jsonl"))
        assert len(jsonl_files) == 1, "Expected exactly one JSONL file"
        lines = jsonl_files[0].read_text().splitlines()
        # At minimum the "Logging initialised" debug line and the warning
        assert any("a warning" in l for l in lines)

    def test_get_logger_returns_logger(self) -> None:
        from src.utils.logging_utils import get_logger
        log = get_logger("src.utils.test")
        assert isinstance(log, logging.Logger)

    def test_setup_logging_idempotent(self, tmp_path: Path) -> None:
        from src.utils.logging_utils import setup_logging, _configured_stages

        stage = "test_idempotent"
        _configured_stages.discard(stage)
        l1 = setup_logging(stage, log_dir=tmp_path)
        l2 = setup_logging(stage, log_dir=tmp_path)
        assert l1 is l2
        # Should still only have the handlers added in the first call
        # (idempotent means second call is a no-op)
        handler_count = len(l1.handlers)
        assert handler_count == 2  # console + jsonl
