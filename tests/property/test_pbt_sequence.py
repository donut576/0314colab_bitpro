"""Property-based tests for the Sequence Scorer.

# Feature: aml-advanced-features, Property 7: sequence score range and insufficient history flag
"""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.services.sequence_scorer import SequenceScorer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=64,
)

_tx_counts = st.integers(min_value=0, max_value=20)

_amounts = st.floats(
    min_value=0.01,
    max_value=1e8,
    allow_nan=False,
    allow_infinity=False,
)

_timestamps = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2025, 12, 31),
)


def _build_transaction(amount: float, ts: datetime) -> dict:
    """Build a minimal transaction dict accepted by SequenceScorer."""
    return {
        "amount": amount,
        "timestamp": ts.isoformat(),
    }


# ---------------------------------------------------------------------------
# Property 7: Sequence score range and insufficient history flag
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    tx_count=_tx_counts,
    amounts=st.lists(_amounts, min_size=20, max_size=20),
    timestamps=st.lists(_timestamps, min_size=20, max_size=20),
)
def test_property_7_sequence_score_range_and_insufficient_history(
    user_id: str,
    tx_count: int,
    amounts: list[float],
    timestamps: list[datetime],
):
    """Property 7: Sequence score range and insufficient history — generate
    users with varying transaction counts, assert score in [0,1] for sufficient
    history and insufficient_history=True for insufficient.

    # Feature: aml-advanced-features, Property 7: sequence score range and insufficient history flag

    **Validates: Requirements REQ-A4.3, REQ-A4.9**
    """
    scorer = SequenceScorer()

    # Add exactly tx_count transactions for this user
    for i in range(tx_count):
        tx = _build_transaction(amounts[i], timestamps[i])
        scorer.add_transaction(user_id, tx)

    result = scorer.score(user_id)

    # Always check user_id matches
    assert result.user_id == user_id

    min_transactions = scorer._min_transactions  # default 5

    if tx_count < min_transactions:
        # Insufficient history: flag must be True, score must be None
        assert result.insufficient_history is True, (
            f"Expected insufficient_history=True for {tx_count} transactions "
            f"(min={min_transactions}), got {result.insufficient_history}"
        )
        assert result.sequence_anomaly_score is None, (
            f"Expected sequence_anomaly_score=None for insufficient history, "
            f"got {result.sequence_anomaly_score}"
        )
    else:
        # Sufficient history: flag must be False, score must be in [0, 1]
        assert result.insufficient_history is False, (
            f"Expected insufficient_history=False for {tx_count} transactions "
            f"(min={min_transactions}), got {result.insufficient_history}"
        )
        assert result.sequence_anomaly_score is not None, (
            f"Expected a numeric sequence_anomaly_score for sufficient history, "
            f"got None"
        )
        assert 0.0 <= result.sequence_anomaly_score <= 1.0, (
            f"sequence_anomaly_score {result.sequence_anomaly_score} not in [0, 1] "
            f"for {tx_count} transactions"
        )
