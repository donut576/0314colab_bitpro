"""Property-based tests for the Threshold Controller.

# Feature: aml-advanced-features, Property 8: adaptive threshold bounds
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.services.threshold_controller import ThresholdController


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_queue_depths = st.integers(min_value=0, max_value=1000)


# ---------------------------------------------------------------------------
# Property 8: Adaptive threshold bounds
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(queue_depth=_queue_depths)
def test_property_8_adaptive_threshold_bounds(queue_depth: int):
    """Property 8: Adaptive threshold bounds — generate random queue depths,
    assert threshold moves in correct direction and stays within [floor, ceiling].

    # Feature: aml-advanced-features, Property 8: adaptive threshold bounds

    **Validates: Requirements REQ-A5.2, REQ-A5.3**
    """
    controller = ThresholdController()

    floor = controller._high_floor
    ceiling = controller._high_ceiling
    max_queue = controller._max_queue
    min_queue = controller._min_queue

    # Record threshold before tick
    old_threshold = controller._high

    # Call tick with the generated queue depth
    controller.tick(queue_depth)

    new_threshold = controller._high

    # Direction assertions
    if queue_depth > max_queue:
        # Queue overloaded: threshold must never decrease
        assert new_threshold >= old_threshold, (
            f"Threshold decreased from {old_threshold} to {new_threshold} "
            f"when queue_depth={queue_depth} > max_queue={max_queue}"
        )
    elif queue_depth < min_queue:
        # Queue underloaded: threshold must never increase
        assert new_threshold <= old_threshold, (
            f"Threshold increased from {old_threshold} to {new_threshold} "
            f"when queue_depth={queue_depth} < min_queue={min_queue}"
        )

    # Bounds assertion: threshold must always stay within [floor, ceiling]
    assert floor <= new_threshold <= ceiling, (
        f"Threshold {new_threshold} outside bounds [{floor}, {ceiling}] "
        f"after tick(queue_depth={queue_depth})"
    )


# ---------------------------------------------------------------------------
# Property 9: Threshold override expiry reversion
# ---------------------------------------------------------------------------

_override_values = st.floats(min_value=0.5, max_value=0.95, allow_nan=False, allow_infinity=False)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(override_value=_override_values)
def test_property_9_threshold_override_expiry_reversion(override_value: float):
    """Property 9: Threshold override expiry reversion — set override with
    past expiry, call tick(), assert reversion to adaptive value and reversion
    event logged.

    # Feature: aml-advanced-features, Property 9: threshold override expiry reversion

    **Validates: Requirements REQ-A5.6**
    """
    from datetime import datetime

    controller = ThresholdController()

    # Set an override with a past expiry so tick() will revert it
    past_expiry = datetime(2020, 1, 1)
    controller.set_override(
        value=override_value,
        reason="manual_test_override",
        expiry=past_expiry,
        operator="test_operator",
    )

    # Confirm override is active before tick
    assert controller._is_override is True
    assert controller._override_expiry == past_expiry

    # Record history length before tick to isolate the reversion event
    history_before = len(controller._history)

    # tick() should detect the expired override and revert
    controller.tick()

    # 1. Override flag must be cleared
    assert controller._is_override is False, (
        f"_is_override should be False after expired override reversion, got True"
    )

    # 2. Override expiry must be cleared
    assert controller._override_expiry is None, (
        f"_override_expiry should be None after reversion, got {controller._override_expiry}"
    )

    # 3. Last reason must reflect the reversion
    assert controller._last_reason == "override_expired_reversion", (
        f"_last_reason should be 'override_expired_reversion', got '{controller._last_reason}'"
    )

    # 4. A reversion event must be appended to history
    assert len(controller._history) == history_before + 1, (
        f"Expected exactly one new history event, got {len(controller._history) - history_before}"
    )
    reversion_event = controller._history[-1]
    assert reversion_event.reason == "override_expired_reversion", (
        f"Reversion event reason should be 'override_expired_reversion', "
        f"got '{reversion_event.reason}'"
    )
    assert reversion_event.threshold_type == "HIGH"
