"""Property-based tests for the Streaming Pipeline.

# Feature: aml-advanced-features, Property 1: streaming enrichment completeness
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.models.feature_store import FeatureVector
from app.services.stream_consumer import StreamConsumer


def _run_async(coro):
    """Run an async coroutine synchronously for Hypothesis compatibility."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=64,
)

# Feature store availability states:
#   "available"   – returns a FeatureVector with cold_start=False
#   "cold_start"  – returns a FeatureVector with cold_start=True
#   "unavailable" – raises an exception
_feature_store_states = st.sampled_from(["available", "cold_start", "unavailable"])

_feature_dicts = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L",)),
        min_size=1,
        max_size=8,
    ),
    values=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=5,
)

_xgb_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_consumer(
    user_id: str,
    fs_state: str,
    features: dict[str, float],
    xgb_score: float,
) -> StreamConsumer:
    """Build a StreamConsumer with mocked dependencies configured for the given state."""

    # --- Feature Store mock ---
    feature_store = AsyncMock()
    if fs_state == "available":
        fv = FeatureVector(
            user_id=user_id,
            schema_version="1",
            features=features,
            groups=["default"],
            last_updated=datetime.utcnow(),
            cold_start=False,
        )
        feature_store.get = AsyncMock(return_value=fv)
    elif fs_state == "cold_start":
        fv = FeatureVector(
            user_id=user_id,
            schema_version="1",
            features={},
            groups=[],
            last_updated=datetime.utcnow(),
            cold_start=True,
        )
        feature_store.get = AsyncMock(return_value=fv)
    else:  # unavailable
        feature_store.get = AsyncMock(side_effect=ConnectionError("Feature store unavailable"))

    # --- Predictor mock ---
    predictor = MagicMock()
    predictor.predict_single = MagicMock(return_value=xgb_score)

    # --- Ensemble scorer mock ---
    ensemble_scorer = MagicMock()
    ensemble_scorer.combine = MagicMock(return_value=max(0.0, min(1.0, xgb_score)))

    # --- Audit logger mock ---
    audit_logger = AsyncMock()
    audit_logger.log_prediction = AsyncMock(return_value=None)

    # --- Alert router mock ---
    alert_router = AsyncMock()
    alert_router.dispatch_async = AsyncMock(return_value=None)

    return StreamConsumer(
        broker_type="kafka",
        feature_store=feature_store,
        predictor=predictor,
        ensemble_scorer=ensemble_scorer,
        audit_logger=audit_logger,
        alert_router=alert_router,
    )


# ---------------------------------------------------------------------------
# Property 1: Streaming enrichment completeness
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    fs_state=_feature_store_states,
    features=_feature_dicts,
    xgb_score=_xgb_scores,
)
def test_property_1_streaming_enrichment_completeness(
    user_id: str,
    fs_state: str,
    features: dict[str, float],
    xgb_score: float,
):
    """Property 1: Streaming enrichment completeness — generate random events +
    feature store availability states, assert enrichment correctness and
    feature_degraded flag.

    **Validates: Requirements REQ-A1.3, REQ-A1.6**
    """
    consumer = _build_consumer(user_id, fs_state, features, xgb_score)
    event = {"user_id": user_id}

    result = _run_async(consumer.process_event(event))

    # 1. Returned record always has user_id matching the event
    assert result["user_id"] == user_id, (
        f"Expected user_id={user_id!r}, got {result['user_id']!r}"
    )

    # 2. When feature store is available and returns non-cold-start features,
    #    feature_degraded must be False
    if fs_state == "available":
        assert result["feature_degraded"] is False, (
            f"Expected feature_degraded=False when feature store is available, "
            f"got {result['feature_degraded']}"
        )

    # 3. When feature store returns cold_start=True, feature_degraded must be True
    if fs_state == "cold_start":
        assert result["feature_degraded"] is True, (
            f"Expected feature_degraded=True when feature store returns cold_start, "
            f"got {result['feature_degraded']}"
        )

    # 4. When feature store raises an exception, feature_degraded must be True
    if fs_state == "unavailable":
        assert result["feature_degraded"] is True, (
            f"Expected feature_degraded=True when feature store is unavailable, "
            f"got {result['feature_degraded']}"
        )

    # 5. Result always contains required keys
    assert "risk_score" in result, "Missing 'risk_score' in result"
    assert "model_version" in result, "Missing 'model_version' in result"

    # 6. risk_score is in [0, 1]
    assert 0.0 <= result["risk_score"] <= 1.0, (
        f"risk_score {result['risk_score']} not in [0, 1]"
    )


# ---------------------------------------------------------------------------
# Property 2: Streaming audit parity
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    fs_state=_feature_store_states,
    features=_feature_dicts,
    xgb_score=_xgb_scores,
)
def test_property_2_streaming_audit_parity(
    user_id: str,
    fs_state: str,
    features: dict[str, float],
    xgb_score: float,
):
    """Property 2: Streaming audit parity — generate random scored events,
    assert audit log contains matching record with same user_id, risk_score,
    model_version.

    # Feature: aml-advanced-features, Property 2: streaming audit parity

    **Validates: Requirements REQ-A1.8**
    """
    consumer = _build_consumer(user_id, fs_state, features, xgb_score)
    event = {"user_id": user_id}

    result = _run_async(consumer.process_event(event))

    # Audit logger must have been called exactly once
    consumer._audit_logger.log_prediction.assert_called_once()

    # Extract the record passed to log_prediction
    audit_record = consumer._audit_logger.log_prediction.call_args[0][0]

    # The audit record must match the returned result on key fields
    assert audit_record["user_id"] == result["user_id"], (
        f"Audit user_id={audit_record['user_id']!r} != result user_id={result['user_id']!r}"
    )
    assert audit_record["risk_score"] == result["risk_score"], (
        f"Audit risk_score={audit_record['risk_score']} != result risk_score={result['risk_score']}"
    )
    assert audit_record["model_version"] == result["model_version"], (
        f"Audit model_version={audit_record['model_version']!r} != result model_version={result['model_version']!r}"
    )


# ---------------------------------------------------------------------------
# Strategies for Property 4
# ---------------------------------------------------------------------------

_optional_scores = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
_positive_weights = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 4: Ensemble score bounds
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    xgb_score=_optional_scores,
    graph_score=_optional_scores,
    seq_score=_optional_scores,
    w_xgb=_positive_weights,
    w_graph=_positive_weights,
    w_seq=_positive_weights,
)
def test_property_4_ensemble_score_bounds(
    xgb_score: float | None,
    graph_score: float | None,
    seq_score: float | None,
    w_xgb: float,
    w_graph: float,
    w_seq: float,
):
    """Property 4: Ensemble score bounds — generate random (xgb_score,
    graph_score, seq_score, weight) triples, assert result in [0, 1].

    # Feature: aml-advanced-features, Property 4: ensemble score bounds

    **Validates: Requirements REQ-A2.6**
    """
    from app.services.ensemble_scorer import EnsembleScorer

    scorer = EnsembleScorer()
    weights = {"xgb": w_xgb, "graph": w_graph, "seq": w_seq}

    result = scorer.combine(
        xgb_score=xgb_score,
        graph_score=graph_score,
        seq_score=seq_score,
        weights=weights,
    )

    # Result must always be in [0, 1]
    assert 0.0 <= result <= 1.0, (
        f"Ensemble score {result} not in [0, 1] for "
        f"xgb={xgb_score}, graph={graph_score}, seq={seq_score}, weights={weights}"
    )

    # If all scores are None, result must be 0.0
    if xgb_score is None and graph_score is None and seq_score is None:
        assert result == 0.0, (
            f"Expected 0.0 when all scores are None, got {result}"
        )
