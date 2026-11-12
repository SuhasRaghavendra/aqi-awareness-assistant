"""
tests/test_service.py
---------------------
Phase 3 contract, parity and latency tests for the FastAPI inference service.

These tests load the real champion bundle (no mocks): the parity test compares
the service's prediction for 50 validation rows with the offline pipeline
within 1e-6, and the latency test enforces the Plan's p95 budgets
(< 50 ms single row, < 100 ms for the 9-city batch).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.serving_features import (
    OBSERVATION_COLUMNS,
    STATIC_COLUMNS,
    build_grap_lookup,
    build_observation_frame,
    build_observation_record,
    load_serving_configs,
)
from src.service import predictor
from src.service.app import app
from src.service.history import HistoryStore
from src.service.schemas import Observation
from src.serving.bundle import AqiBundle

pytestmark = pytest.mark.service

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "artifacts" / "champion"
VAL_SNAPSHOT = REPO_ROOT / "tests" / "fixtures" / "val_snapshot.parquet"


def resolve_val_path() -> Path:
    """``AQI_VAL_PATH`` -> committed snapshot -> DVC-managed split.

    CI runs this suite on a clean checkout, so the snapshot must be enough; the
    DVC path is only a convenience for local runs with the full dataset.
    """
    override = os.environ.get("AQI_VAL_PATH")
    candidates = [Path(override)] if override else []
    candidates.extend([VAL_SNAPSHOT, REPO_ROOT / "data" / "processed" / "val.parquet"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No validation split found; set AQI_VAL_PATH or commit tests/fixtures/val_snapshot.parquet"
    )


# Fields a client sends: today's observations only.  Calendar, city-static,
# vehicle/IIP, GRAP and festival values are filled server-side.
REQUEST_FIELDS = [
    "aqi",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "surface_pressure_mean",
    "fire_count_500km",
    "fire_frp_sum_500km",
    "fire_dist_min_km",
]

HISTORY_FIELDS = REQUEST_FIELDS + ["grap_stage", "festival_flag", *STATIC_COLUMNS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bundle() -> AqiBundle:
    return AqiBundle.load(BUNDLE_DIR)


@pytest.fixture(scope="session")
def val_frame() -> pd.DataFrame:
    frame = pd.read_parquet(resolve_val_path())
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values(["city", "date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def client() -> TestClient:
    os.environ.setdefault("BUNDLE_DIR", str(BUNDLE_DIR))
    os.environ.setdefault("CONFIG_DIR", str(REPO_ROOT / "configs"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _fresh_history(client: TestClient):
    """Every test starts from the bundle's 45-day seed."""
    predictor.reset_history()
    yield
    predictor.reset_history()


def test_validation_snapshot_is_hash_pinned() -> None:
    """CI runs on the committed snapshot; it must be the frozen val.parquet."""
    assert VAL_SNAPSHOT.exists(), "commit tests/fixtures/val_snapshot.parquet"
    digest = hashlib.md5(VAL_SNAPSHOT.read_bytes()).hexdigest()  # noqa: S324 - content identity
    assert digest == "462b7712b4cea342fe36ca80868a29f4", (
        "val_snapshot.parquet drifted from the frozen validation split; "
        "update params.yaml integrity.val_md5 and the ADR deliberately"
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def make_payload(row: pd.Series, **overrides) -> dict:
    payload = {
        "city": str(row["city"]),
        "date": str(row["date"]),
        **{
            field: float(row[field])
            for field in REQUEST_FIELDS
            if field in row.index and pd.notna(row[field])
        },
    }
    payload.update(overrides)
    return payload


def make_history_payload(row: pd.Series, history: pd.DataFrame) -> dict:
    records = []
    for _, past in history.iterrows():
        records.append(
            {
                "city": str(past["city"]),
                "date": str(past["date"]),
                **{
                    field: float(past[field])
                    for field in HISTORY_FIELDS
                    if field in past.index and pd.notna(past[field])
                },
            }
        )
    return {**make_payload(row), "history": records}


def row_for(frame: pd.DataFrame, city: str, date: str) -> pd.Series:
    match = frame[(frame["city"] == city) & (frame["date"] == date)]
    assert not match.empty, f"missing {city} {date}"
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Health / readiness / metadata
# ---------------------------------------------------------------------------


class TestObservability:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}

    def test_ready_reports_bundle_and_seed(self, client, bundle):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True
        assert body["model_name"] == bundle.manifest["model"]["name"]
        assert body["model_version"] == bundle.manifest["bundle_version"]
        assert body["features_count"] == len(bundle.feature_names)
        assert body["history_seed_days"] == 45

    def test_ready_503_when_bundle_missing(self, client, monkeypatch):
        monkeypatch.setattr(predictor, "is_ready", lambda: False)
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["ready"] is False

    def test_metadata(self, client, bundle):
        resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_name"] == bundle.manifest["model"]["name"]
        assert body["features_count"] == 103
        assert body["history_seed_days"] == 45
        assert body["history_seed_rows"] == 405
        assert isinstance(body["gate_status"], dict)
        assert body["val_metrics"] and body["test_metrics"]

    def test_metadata_503_when_not_ready(self, client, monkeypatch):
        monkeypatch.setattr(predictor, "is_ready", lambda: False)
        assert client.get("/metadata").status_code == 503

    def test_metrics_exposes_prediction_instruments(self, client, val_frame):
        client.post("/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30")))
        text = client.get("/metrics").text
        for metric in (
            "aqi_http_requests_total",
            "aqi_http_request_duration_seconds",
            "aqi_predictions_total",
            "aqi_prediction_value",
            "aqi_out_of_range_total",
        ):
            assert metric in text


# ---------------------------------------------------------------------------
# Request contract
# ---------------------------------------------------------------------------


class TestPredictContract:
    def test_valid_request_returns_full_contract(self, client, val_frame, bundle):
        resp = client.post("/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30")))
        assert resp.status_code == 200
        body = resp.json()

        assert body["city"] == "Delhi"
        assert body["date"] == "2024-12-30"
        assert body["target_date"] == "2024-12-31"
        assert body["category"] in {
            "Good",
            "Satisfactory",
            "Moderate",
            "Poor",
            "Very Poor",
            "Severe",
        }
        interval = body["aqi_interval_80"]
        assert interval["lower"] <= interval["median"] <= interval["upper"]
        assert body["aqi_pred"] == pytest.approx(interval["median"])
        assert 0.0 <= body["aqi_pred"] <= 500.0
        assert sum(body["category_probs"].values()) == pytest.approx(1.0, abs=1e-5)
        assert body["model_name"] == bundle.manifest["model"]["name"]
        assert body["model_version"] == bundle.manifest["bundle_version"]
        assert body["alias"] == "champion"
        assert body["model_used"] == bundle.manifest["model"]["name"]
        assert body["fallback_used"] is False
        assert body["degraded"] is False
        assert body["history_days"] == 44
        assert body["latency_ms"] >= 0.0
        assert body["warnings"] == []
        assert "out_of_range_fields" not in body

    def test_request_id_is_echoed_and_generated(self, client, val_frame):
        payload = make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        echoed = client.post("/predict", json=payload, headers={"X-Request-ID": "abc-123"})
        assert echoed.headers["X-Request-ID"] == "abc-123"
        generated = client.post("/predict", json=payload)
        assert len(generated.headers["X-Request-ID"]) == 32

    def test_optional_fields_may_be_omitted(self, client):
        resp = client.post("/predict", json={"city": "Mumbai", "date": "2024-12-30", "aqi": 120.0})
        assert resp.status_code == 200
        assert resp.json()["aqi_pred"] > 0

    def test_all_nine_cities_accepted(self, client, val_frame):
        cities = [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Chennai",
            "Kolkata",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Lucknow",
        ]
        for city in cities:
            resp = client.post(
                "/predict", json=make_payload(row_for(val_frame, city, "2024-12-30"))
            )
            assert resp.status_code == 200, city

    def test_degraded_when_history_is_empty(self, client, val_frame):
        predictor.load_history(pd.DataFrame())
        resp = client.post("/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30")))
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_days"] == 0
        assert body["degraded"] is True
        assert any("history" in warning for warning in body["warnings"])

    def test_history_grows_with_accepted_requests(self, client, val_frame):
        first = client.post(
            "/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        )
        assert first.json()["history_days"] == 44
        second = client.post(
            "/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        )
        assert second.json()["history_days"] == 44  # same date replaces, no growth
        predictor.load_history(pd.DataFrame())
        third = client.post(
            "/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        )
        assert third.json()["history_days"] == 0

    def test_out_of_range_input_warns(self, client, val_frame):
        payload = make_payload(row_for(val_frame, "Delhi", "2024-12-30"), fire_count_500km=50_000.0)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert any("fire_count_500km" in warning for warning in body["warnings"])

    def test_champion_failure_falls_back_to_persistence(self, client, val_frame, monkeypatch):
        def boom(self, frame):
            raise RuntimeError("simulated model failure")

        monkeypatch.setattr(type(predictor.get_bundle()), "predict_interval", boom)
        payload = make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["fallback_used"] is True
        assert body["model_used"] == "fallback_persistence"
        assert body["aqi_pred"] == pytest.approx(payload["aqi"])
        assert body["degraded"] is True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_fields_is_422(self, client):
        assert client.post("/predict", json={"city": "Delhi"}).status_code == 422

    def test_unknown_city_is_422(self, client):
        resp = client.post(
            "/predict", json={"city": "Atlantis", "date": "2024-12-30", "aqi": 100.0}
        )
        assert resp.status_code == 422

    def test_bad_date_is_422(self, client):
        resp = client.post("/predict", json={"city": "Delhi", "date": "31-12-2024", "aqi": 100.0})
        assert resp.status_code == 422

    def test_unknown_field_is_422(self, client):
        resp = client.post(
            "/predict",
            json={"city": "Delhi", "date": "2024-12-30", "aqi": 100.0, "aqi_lag_1": 90.0},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        "overrides",
        [
            {"aqi": -10.0},
            {"relative_humidity_2m_mean": 150.0},
            {"wind_speed_10m_mean": -2.0},
            {"precipitation_sum": -1.0},
        ],
    )
    def test_impossible_values_are_400(self, client, val_frame, overrides):
        payload = make_payload(row_for(val_frame, "Delhi", "2024-12-30"), **overrides)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"]["errors"]

    def test_predict_503_when_not_ready(self, client, val_frame, monkeypatch):
        monkeypatch.setattr(predictor, "is_ready", lambda: False)
        resp = client.post("/predict", json=make_payload(row_for(val_frame, "Delhi", "2024-12-30")))
        assert resp.status_code == 503

    def test_batch_empty_is_422(self, client):
        assert client.post("/predict/batch", json={"rows": []}).status_code == 422

    def test_batch_with_impossible_row_is_400(self, client, val_frame):
        rows = [
            make_payload(row_for(val_frame, "Delhi", "2024-12-30")),
            make_payload(row_for(val_frame, "Mumbai", "2024-12-30"), aqi=-5.0),
        ]
        resp = client.post("/predict/batch", json={"rows": rows})
        assert resp.status_code == 400
        assert "row 1" in resp.json()["detail"]["message"]

    def test_batch_503_when_not_ready(self, client, val_frame, monkeypatch):
        monkeypatch.setattr(predictor, "is_ready", lambda: False)
        resp = client.post(
            "/predict/batch",
            json={"rows": [make_payload(row_for(val_frame, "Delhi", "2024-12-30"))]},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Batch contract
# ---------------------------------------------------------------------------


class TestBatch:
    def test_nine_city_batch(self, client, val_frame):
        cities = [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Chennai",
            "Kolkata",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Lucknow",
        ]
        rows = [make_payload(row_for(val_frame, city, "2024-12-30")) for city in cities]
        resp = client.post("/predict/batch", json={"rows": rows})
        assert resp.status_code == 200
        body = resp.json()
        assert body["rows_processed"] == 9
        assert [prediction["city"] for prediction in body["predictions"]] == cities
        assert all(
            prediction["history_days"] >= 43 and not prediction["degraded"]
            for prediction in body["predictions"]
        )

    def test_batch_matches_single(self, client, val_frame):
        cities = ["Delhi", "Mumbai", "Bengaluru"]
        rows = [make_payload(row_for(val_frame, city, "2024-12-30")) for city in cities]
        batch = client.post("/predict/batch", json={"rows": rows}).json()["predictions"]

        predictor.reset_history()
        singles = []
        for row in rows:
            singles.append(client.post("/predict", json=row).json())
        for batched, single in zip(batch, singles, strict=False):
            assert batched["aqi_pred"] == pytest.approx(single["aqi_pred"], abs=1e-9)


# ---------------------------------------------------------------------------
# Parity with the offline pipeline (Plan V5)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_rows(val_frame: pd.DataFrame) -> list[int]:
    candidates = val_frame[val_frame["date"] >= "2024-03-01"].index.to_numpy()
    rng = np.random.default_rng(20260101)
    return rng.choice(candidates, size=50, replace=False).tolist()


class TestParity:
    def test_predictions_match_offline_within_1e_6(self, client, bundle, val_frame, parity_rows):
        differences = []
        for index in parity_rows:
            row = val_frame.loc[index]
            history = val_frame[
                (val_frame["city"] == row["city"]) & (val_frame["date"] < row["date"])
            ].tail(45)
            predictor.load_history(history)

            service = predictor.predict_one(make_payload(row))
            offline = bundle.predict_interval(val_frame.loc[[index]])[1][0]
            differences.append(abs(service["aqi_pred"] - offline))

        differences = np.asarray(differences)
        assert differences.max() <= 1e-6, (
            f"max |service - offline| = {differences.max():.6g} " f"(mean {differences.mean():.6g})"
        )

    def test_features_match_offline_within_1e_6(self, client, bundle, val_frame, parity_rows):
        differences = []
        for index in parity_rows:
            row = val_frame.loc[index]
            history = val_frame[
                (val_frame["city"] == row["city"]) & (val_frame["date"] < row["date"])
            ].tail(45)
            predictor.load_history(history)

            features, _, _, _, _ = predictor.prepare_features(make_payload(row))
            online = features.iloc[0][bundle.feature_names].astype(float).to_numpy()
            offline = row[bundle.feature_names].astype(float).to_numpy()
            differences.append(np.nanmax(np.abs(online - offline)))

        assert max(differences) <= 1e-6, f"max feature delta {max(differences):.6g}"

    def test_stateless_history_matches_store_history(self, client, val_frame):
        row = row_for(val_frame, "Delhi", "2024-12-30")
        history = val_frame[
            (val_frame["city"] == "Delhi") & (val_frame["date"] < "2024-12-30")
        ].tail(45)

        predictor.load_history(history)
        store_result = predictor.predict_one(make_payload(row))

        predictor.load_history(pd.DataFrame())
        stateless_result = predictor.predict_one(make_history_payload(row, history))

        assert stateless_result["aqi_pred"] == pytest.approx(store_result["aqi_pred"], abs=1e-6)
        assert stateless_result["history_days"] == len(history)
        assert predictor.history_snapshot() == {}


# ---------------------------------------------------------------------------
# Latency budgets (Plan V6)
# ---------------------------------------------------------------------------


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


class TestLatency:
    def test_single_row_p95_under_50ms(self, client, val_frame):
        payload = make_payload(row_for(val_frame, "Delhi", "2024-12-30"))
        # Warm-up pass to let JIT, thread pool, and connection pool settle
        for _ in range(5):
            client.post("/predict", json=payload)
        timings = []
        for _ in range(40):
            start = time.perf_counter()
            resp = client.post("/predict", json=payload)
            timings.append((time.perf_counter() - start) * 1000.0)
            assert resp.status_code == 200
        assert p95(timings) < 50.0, f"p95 single-row latency {p95(timings):.1f} ms"

    def test_nine_city_batch_p95_under_100ms(self, client, val_frame):
        cities = [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Chennai",
            "Kolkata",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Lucknow",
        ]
        rows = [make_payload(row_for(val_frame, city, "2024-12-30")) for city in cities]
        # Warm-up pass to let batch execution allocate memory pools
        for _ in range(5):
            client.post("/predict/batch", json={"rows": rows})
        timings = []
        for _ in range(30):
            start = time.perf_counter()
            resp = client.post("/predict/batch", json={"rows": rows})
            timings.append((time.perf_counter() - start) * 1000.0)
            assert resp.status_code == 200
        assert p95(timings) < 100.0, f"p95 batch latency {p95(timings):.1f} ms"


# ---------------------------------------------------------------------------
# Fast serving helpers must mirror the pandas reference implementation
# ---------------------------------------------------------------------------


class TestServingFeatures:
    def test_scalar_observation_matches_pandas_on_policy_dates(self):
        grap_calendar, festival_days = load_serving_configs(REPO_ROOT / "configs")
        grap_lookup = build_grap_lookup(grap_calendar)
        static_reference = {
            "diesel_frac": 0.151613,
            "cng_frac": 0.03,
            "ev_frac": 0.005,
            "two_wheeler_frac": 0.72,
            "vehicles_per_lakh": 64990.0,
            "iip_manufacturing_index": 0.0,
        }
        record = {
            "aqi": 210.0,
            "temperature_2m_mean": 14.0,
            "relative_humidity_2m_mean": 82.0,
            "wind_speed_10m_mean": 2.0,
            "precipitation_sum": 6.0,
            "fire_count_500km": 25.0,
            **static_reference,
        }
        cases = [
            ("Delhi", "2024-10-31"),  # pre-Diwali GRAP/festival boundary
            ("Delhi", "2024-11-01"),  # Diwali + GRAP stage
            ("Delhi", "2024-12-30"),
            ("Mumbai", "2024-04-01"),  # BS-VI boundary is 2020, still exercise
            ("Lucknow", "2024-11-01"),
        ]

        for city, date in cases:
            fast = build_observation_record(
                {**record, "city": city, "date": date},
                grap_lookup=grap_lookup,
                festival_days=festival_days,
                static_reference=static_reference,
            )
            reference = build_observation_frame(
                [{**record, "city": city, "date": date}],
                params={},
                grap_calendar=grap_calendar,
                festival_days=festival_days,
                static_reference=static_reference,
            ).iloc[0]

            for column, expected in reference.items():
                actual = fast[column]
                if pd.isna(expected) and pd.isna(actual):
                    continue
                if isinstance(expected, float):
                    assert actual == pytest.approx(expected, abs=1e-12), (city, date, column)
                else:
                    assert actual == expected, (city, date, column)

    def test_grap_and_festival_overrides_win(self):
        grap_calendar, festival_days = load_serving_configs(REPO_ROOT / "configs")
        lookup = build_grap_lookup(grap_calendar)
        fast = build_observation_record(
            {
                "city": "Delhi",
                "date": "2024-12-30",
                "aqi": 300.0,
                "grap_stage": 4,
                "festival_flag": 1,
            },
            grap_lookup=lookup,
            festival_days=festival_days,
            train_medians={},
        )
        assert fast["grap_stage"] == 4
        assert fast["festival_flag"] == 1


class TestHistoryStore:
    def test_append_trim_and_lookup(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "city": ["Delhi", "Delhi"],
                "aqi": [100.0, 110.0],
            }
        )
        store = HistoryStore(frame, max_days=5)
        assert store.history_days("Delhi", "2024-01-03") == 2
        store.append(
            {
                "date": "2024-01-03",
                "city": "Delhi",
                "aqi": 120.0,
                "temperature_2m_mean": 15.0,
            }
        )
        assert store.snapshot() == {"Delhi": 3}
        assert store.last_date("Delhi") == "2024-01-03"
        assert store.history_before("Delhi", "2024-01-03")["aqi"].tolist() == [100.0, 110.0]

    def test_reappending_a_date_replaces_instead_of_duplicating(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "city": ["Delhi", "Delhi"],
                "aqi": [100.0, 110.0],
            }
        )
        store = HistoryStore(frame, max_days=10)
        store.append({"date": "2024-01-02", "city": "Delhi", "aqi": 999.0})

        assert store.snapshot() == {"Delhi": 2}
        assert store.history_before("Delhi", "2024-01-03")["aqi"].tolist() == [100.0, 999.0]

    def test_out_of_order_insert_keeps_dates_sorted(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-03"],
                "city": ["Delhi", "Delhi"],
                "aqi": [100.0, 120.0],
            }
        )
        store = HistoryStore(frame, max_days=10)
        store.append({"date": "2024-01-02", "city": "Delhi", "aqi": 130.0})

        history = store.history_before("Delhi", "2024-01-04")
        assert history["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert store.last_date("Delhi") == "2024-01-03"

    def test_append_beyond_max_days_trims_the_oldest_rows(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame({"date": dates, "city": "Delhi", "aqi": [10.0 + i for i in range(5)]})
        store = HistoryStore(frame, max_days=3)
        assert store.snapshot() == {"Delhi": 3}

        store.append({"date": "2024-01-06", "city": "Delhi", "aqi": 99.0})
        assert store.snapshot() == {"Delhi": 3}
        history = store.history_before("Delhi", "2024-01-07")
        assert history["date"].tolist() == ["2024-01-04", "2024-01-05", "2024-01-06"]

    def test_reset_does_not_mutate_the_callers_frame(self):
        frame = pd.DataFrame(
            {"date": ["2024-01-01"], "city": ["Delhi"], "aqi": [100.0], "target_aqi": [1.0]}
        )
        before = frame.copy(deep=True)
        store = HistoryStore(frame, max_days=5)
        store.append({"date": "2024-01-02", "city": "Delhi", "aqi": 101.0})

        pd.testing.assert_frame_equal(frame, before)

    def test_concurrent_appends_and_reads_stay_consistent(self):
        import threading

        store = HistoryStore(None, max_days=1000)
        errors: list[BaseException] = []

        def writer(city: str, start: int) -> None:
            try:
                for offset in range(50):
                    day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=start + offset)
                    store.append(
                        {"date": day.strftime("%Y-%m-%d"), "city": city, "aqi": float(offset)}
                    )
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(200):
                    store.snapshot()
                    store.history_before("Delhi", "2025-01-01")
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=("Delhi", 0)),
            threading.Thread(target=writer, args=("Mumbai", 10)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        assert store.snapshot() == {"Delhi": 50, "Mumbai": 50}
        assert store.history_days("Delhi", "2025-01-01") == 50

    def test_observation_schema_accepts_history_fields(self):
        allowed = set(Observation.model_fields)
        for field in HISTORY_FIELDS:
            assert field in allowed
        assert OBSERVATION_COLUMNS
