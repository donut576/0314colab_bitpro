"""Property-based tests for the Alert Router.

# Feature: aml-advanced-features, Property 12: alert rate limiting
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.models.alert import AlertStatus, RiskAlert
from app.services.alert_router import AlertRouter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_num_alerts = st.integers(min_value=5, max_value=20)

_risk_scores = st.floats(min_value=0.7, max_value=1.0, allow_nan=False, allow_infinity=False)


def _build_alert(user_id: str, risk_score: float) -> RiskAlert:
    """Build a RiskAlert with a unique user_id to avoid suppression."""
    return RiskAlert(
        case_id=f"case-{user_id}",
        user_id=user_id,
        risk_score=risk_score,
        risk_level="HIGH",
        top_signals=["signal_a", "signal_b", "signal_c"],
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Property 12: Alert rate limiting
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    num=_num_alerts,
    scores=st.lists(_risk_scores, min_size=20, max_size=20),
)
def test_property_12_alert_rate_limiting(num: int, scores: list[float]):
    """Property 12: Alert rate limiting — generate alert sequences with
    timestamps, assert per-channel count <= limit in any 60-min window;
    excess alerts queued not dropped.

    # Feature: aml-advanced-features, Property 12: alert rate limiting

    **Validates: Requirements REQ-A8.4, REQ-A8.5**
    """
    rate_limit = 5
    router = AlertRouter(rate_limit_per_hour=rate_limit)

    # Dispatch `num` alerts, each with a unique user_id to avoid suppression
    for i in range(num):
        alert = _build_alert(user_id=f"user-{i}", risk_score=scores[i])
        router.dispatch(alert)

    history = router.get_history()

    # All dispatched alerts go to all 3 channels (line, email, webhook)
    channels = router._channels

    for channel in channels:
        channel_records = [r for r in history if r.channel == channel]

        delivered = [r for r in channel_records if r.status == AlertStatus.DELIVERED]
        queued = [r for r in channel_records if r.status == AlertStatus.QUEUED]

        # 1. DELIVERED count must not exceed rate_limit
        assert len(delivered) <= rate_limit, (
            f"Channel '{channel}': {len(delivered)} delivered alerts "
            f"exceeds rate limit {rate_limit}"
        )

        # 2. Total (DELIVERED + QUEUED) must equal total dispatched — no drops
        assert len(delivered) + len(queued) == num, (
            f"Channel '{channel}': delivered ({len(delivered)}) + "
            f"queued ({len(queued)}) = {len(delivered) + len(queued)}, "
            f"expected {num} (no alerts dropped)"
        )

        # 3. If num > rate_limit, excess must be QUEUED
        if num > rate_limit:
            expected_queued = num - rate_limit
            assert len(queued) == expected_queued, (
                f"Channel '{channel}': expected {expected_queued} queued alerts, "
                f"got {len(queued)}"
            )

        # 4. If num <= rate_limit, nothing should be queued
        if num <= rate_limit:
            assert len(queued) == 0, (
                f"Channel '{channel}': expected 0 queued alerts when "
                f"num ({num}) <= rate_limit ({rate_limit}), got {len(queued)}"
            )


# ---------------------------------------------------------------------------
# Property 13: Alert suppression within cooldown
# ---------------------------------------------------------------------------

_repeat_counts = st.integers(min_value=2, max_value=5)

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=12,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    repeats=_repeat_counts,
    score=_risk_scores,
)
def test_property_13_alert_suppression_within_cooldown(
    user_id: str, repeats: int, score: float
):
    """Property 13: Alert suppression within cooldown — generate alert
    sequences for same user_id within cooldown window, assert suppression.

    # Feature: aml-advanced-features, Property 13: alert suppression within cooldown

    **Validates: Requirements REQ-A8.9**
    """
    router = AlertRouter()  # default cooldown = 3600s

    # Dispatch the same user_id alert `repeats` times (all within cooldown)
    for _ in range(repeats):
        alert = _build_alert(user_id=user_id, risk_score=score)
        router.dispatch(alert)

    history = router.get_history()

    # Separate records by status
    delivered = [r for r in history if r.status == AlertStatus.DELIVERED]
    suppressed = [r for r in history if r.status == AlertStatus.SUPPRESSED]

    # 1. First alert is DELIVERED (one record per channel = 3 channels)
    assert len(delivered) == len(router._channels), (
        f"Expected {len(router._channels)} delivered records (one per channel), "
        f"got {len(delivered)}"
    )

    # 2. All delivered records belong to the same user_id
    for rec in delivered:
        assert rec.user_id == user_id

    # 3. Subsequent alerts are SUPPRESSED — one suppressed record per repeat
    expected_suppressed = repeats - 1
    assert len(suppressed) == expected_suppressed, (
        f"Expected {expected_suppressed} suppressed records, got {len(suppressed)}"
    )

    # 4. Suppressed records have status=SUPPRESSED and channel="all"
    for rec in suppressed:
        assert rec.status == AlertStatus.SUPPRESSED, (
            f"Expected SUPPRESSED status, got {rec.status}"
        )
        assert rec.channel == "all", (
            f"Expected channel='all' for suppressed record, got '{rec.channel}'"
        )
        assert rec.user_id == user_id
