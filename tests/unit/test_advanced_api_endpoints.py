"""Unit tests for all new advanced API endpoints.

Uses FastAPI TestClient with all new services mocked.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.alert import AlertRecord, AlertStatus, ChannelTestResult
from app.models.case import Case, CaseAuditEntry, CaseStats, CaseStatus, PaginatedCases
from app.models.cluster import ClusterStats, IdentityCluster
from app.models.copilot import SimilarCase, SimilarCasesResult
from app.models.feature_store import FeatureStoreStats, FeatureVector
from app.models.graph import GraphScore, SubgraphResult
from app.models.monitoring import (
    CalibrationCurve, ModelMetrics, SLAStatus, UnifiedHealthSummary,
)
from app.models.sequence import BehavioralProfile, SequenceScore
from app.models.threshold import ThresholdChangeEvent, ThresholdSimulation, ThresholdState
from app.services.stream_consumer import StreamHealth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 1, 1, 0, 0, 0)


def _inject_mocks(app) -> None:
    """Inject mock services into app.state (called after lifespan startup)."""
    # Feature Store
    fs = MagicMock()
    fs.get = AsyncMock(return_value=FeatureVector(
        user_id="u1", schema_version="1", features={}, groups=[], last_updated=NOW, cold_start=True
    ))
    fs.get_batch = AsyncMock(return_value=[])
    fs.get_stats = AsyncMock(return_value=FeatureStoreStats(
        total_users=0, schema_version="1", last_updated=NOW, storage_utilization_bytes=0
    ))
    app.state.feature_store = fs

    # Stream Consumer — set AFTER lifespan so it isn't overwritten
    sc = MagicMock()
    sc.get_health.return_value = StreamHealth(
        broker_connected=True, consumer_lag=0, events_per_second=10.0, broker_type="kafka"
    )
    app.state.stream_consumer = sc

    # Graph Engine
    ge = MagicMock()
    ge.get_score.return_value = GraphScore(
        user_id="u1", graph_risk_score=0.5, embedding=[], hop1_count=2,
        hop2_count=5, betweenness_centrality=0.1, elevated=False
    )
    ge.get_subgraph.return_value = SubgraphResult(user_id="u1", nodes=[], edges=[], hops=2)
    app.state.graph_engine = ge

    # Identity Clusterer
    ic = MagicMock()
    ic.get_stats.return_value = ClusterStats(
        total_clusters=1, average_cluster_size=2.0, high_risk_cluster_count=0, computed_at=NOW
    )
    ic.get_cluster_for_account.return_value = None
    ic.get_cluster.return_value = None
    app.state.identity_clusterer = ic

    # Sequence Scorer
    ss = MagicMock()
    ss.score.return_value = SequenceScore(
        user_id="u1", sequence_anomaly_score=None, top_anomalous_events=[],
        model_version="seq-1.0", insufficient_history=True
    )
    ss.get_profile.return_value = BehavioralProfile(
        user_id="u1", transaction_count=0, lookback_days=90, last_scored=None, recent_events=[]
    )
    app.state.sequence_scorer = ss

    # Threshold Controller
    tc = MagicMock()
    tc.get_current.return_value = ThresholdState(
        high_threshold=0.7, medium_threshold=0.4, last_updated=NOW,
        last_change_reason="initial", is_override=False, override_expiry=None
    )
    tc.simulate.return_value = ThresholdSimulation(
        proposed_threshold=0.8, estimated_alert_volume=200,
        estimated_recall=0.2, estimated_precision=0.8, computed_at=NOW
    )
    tc.get_history.return_value = []
    app.state.threshold_controller = tc

    # Case Manager
    cm = MagicMock()
    cm.get_stats.return_value = CaseStats(
        open_count=0, avg_resolution_hours=0.0,
        false_positive_rate=0.0, confirmed_fraud_rate=0.0, computed_at=NOW
    )
    cm.list_cases.return_value = PaginatedCases(items=[], total=0, page=1, page_size=20)
    cm.get_case.return_value = None
    app.state.case_manager = cm

    # Alert Router
    ar = MagicMock()
    ar.get_history.return_value = []
    ar.send_test.return_value = [
        ChannelTestResult(channel="line", success=True, message="ok")
    ]
    app.state.alert_router = ar

    # AI Copilot
    async def _explain_gen(case_id):
        yield "explanation chunk"
    async def _suggest_gen(case_id):
        yield "suggestion chunk"
    cop = MagicMock()
    cop.explain = _explain_gen
    cop.suggest = _suggest_gen
    cop.compare = AsyncMock(return_value=SimilarCasesResult(
        case_id="c1", similar_cases=[], generated_at=NOW
    ))
    app.state.ai_copilot = cop

    # Monitoring System
    ms = MagicMock()
    ms.get_dashboard.return_value = UnifiedHealthSummary(
        models=[], sla_statuses=[], overall_healthy=True, computed_at=NOW
    )
    ms.get_model_calibration.return_value = CalibrationCurve(
        model_name="xgboost", points=[], computed_at=NOW
    )
    app.state.monitoring_system = ms


@pytest.fixture()
def client():
    app = create_app()
    # Enter TestClient context to trigger lifespan startup, then override with mocks
    with TestClient(app, raise_server_exceptions=True) as c:
        _inject_mocks(app)
        yield c


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

def test_get_feature_vector(client):
    r = client.get("/features/u1")
    assert r.status_code == 200
    assert r.json()["user_id"] == "u1"
    assert r.json()["cold_start"] is True


def test_get_feature_stats(client):
    r = client.get("/features/stats")
    assert r.status_code == 200
    assert "schema_version" in r.json()


def test_post_features_batch(client):
    r = client.post("/features/batch", json={"user_ids": ["u1", "u2"]})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

def test_stream_health(client):
    r = client.get("/stream/health")
    assert r.status_code == 200
    data = r.json()
    assert data["broker_connected"] is True
    assert data["broker_type"] == "kafka"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def test_graph_score(client):
    r = client.post("/graph/score", json={"user_ids": ["u1"]})
    assert r.status_code == 200
    assert r.json()[0]["user_id"] == "u1"


def test_graph_subgraph(client):
    r = client.get("/graph/subgraph/u1")
    assert r.status_code == 200
    assert r.json()["user_id"] == "u1"


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

def test_cluster_stats(client):
    r = client.get("/clusters/stats")
    assert r.status_code == 200
    assert "total_clusters" in r.json()


def test_cluster_account_not_found(client):
    r = client.get("/clusters/account/unknown_user")
    assert r.status_code == 404


def test_cluster_not_found(client):
    r = client.get("/clusters/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

def test_sequence_score(client):
    r = client.post("/sequence/score", json={"user_id": "u1"})
    assert r.status_code == 200
    assert r.json()["insufficient_history"] is True


def test_sequence_profile(client):
    r = client.get("/sequence/profile/u1")
    assert r.status_code == 200
    assert r.json()["user_id"] == "u1"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_thresholds_current(client):
    r = client.get("/thresholds/current")
    assert r.status_code == 200
    assert r.json()["high_threshold"] == 0.7


def test_thresholds_simulation(client):
    r = client.get("/thresholds/simulation", params={"proposed": 0.8})
    assert r.status_code == 200
    assert "estimated_alert_volume" in r.json()


def test_thresholds_history(client):
    r = client.get("/thresholds/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def test_cases_stats(client):
    r = client.get("/cases/stats")
    assert r.status_code == 200
    assert "open_count" in r.json()


def test_cases_list(client):
    r = client.get("/cases")
    assert r.status_code == 200
    assert "items" in r.json()


def test_case_not_found(client):
    r = client.get("/cases/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def test_alerts_history(client):
    r = client.get("/alerts/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_alerts_test(client):
    r = client.post("/alerts/test")
    assert r.status_code == 200
    assert r.json()[0]["channel"] == "line"


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------

def test_copilot_explain(client):
    r = client.post("/copilot/explain/c1")
    assert r.status_code == 200


def test_copilot_suggest(client):
    r = client.post("/copilot/suggest/c1")
    assert r.status_code == 200


def test_copilot_compare(client):
    r = client.post("/copilot/compare/c1")
    assert r.status_code == 200
    assert r.json()["case_id"] == "c1"


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def test_monitoring_dashboard(client):
    r = client.get("/monitoring/dashboard")
    assert r.status_code == 200
    assert r.json()["overall_healthy"] is True


def test_monitoring_calibration(client):
    r = client.get("/monitoring/model/xgboost/calibration")
    assert r.status_code == 200
    assert r.json()["model_name"] == "xgboost"
