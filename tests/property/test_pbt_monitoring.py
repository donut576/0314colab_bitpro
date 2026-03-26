"""Property-based tests for the Monitoring System.

# Feature: aml-advanced-features, Property 16: monitoring F1 degradation detection
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.models.monitoring import ModelMetrics
from app.services.monitoring_system import MonitoringSystem


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_f1_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 16: Monitoring F1 degradation detection
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    baseline_f1=_f1_scores,
    rolling_f1=_f1_scores,
)
def test_property_16_monitoring_f1_degradation_detection(
    baseline_f1: float, rolling_f1: float
):
    """Property 16: Monitoring F1 degradation detection — generate
    (baseline_f1, rolling_f1) pairs, assert alert emitted iff drop > 0.05;
    no alert when within 0.05.

    # Feature: aml-advanced-features, Property 16: monitoring F1 degradation detection

    **Validates: Requirements REQ-A10.4**
    """
    mock_router = MagicMock()
    system = MonitoringSystem(alert_router=mock_router)

    # Set a model's F1 metrics directly
    model_name = "xgboost"
    m = system._metrics[model_name]
    system._metrics[model_name] = m.model_copy(update={
        "baseline_f1": baseline_f1,
        "rolling_f1_7d": rolling_f1,
    })

    # Run the monitoring tick
    system.tick()

    drop = baseline_f1 - rolling_f1

    if drop > 0.05:
        # Alert MUST have been emitted
        assert mock_router.dispatch.called, (
            f"Expected degradation alert when baseline_f1={baseline_f1} "
            f"and rolling_f1={rolling_f1} (drop={drop:.4f} > 0.05), "
            f"but dispatch was not called"
        )
        # Verify the alert contains the model degradation signal
        alert = mock_router.dispatch.call_args[0][0]
        assert f"model_degradation:{model_name}" in alert.top_signals, (
            f"Expected 'model_degradation:{model_name}' in alert top_signals, "
            f"got {alert.top_signals}"
        )
    else:
        # Alert must NOT have been emitted
        assert not mock_router.dispatch.called, (
            f"Expected no degradation alert when baseline_f1={baseline_f1} "
            f"and rolling_f1={rolling_f1} (drop={drop:.4f} <= 0.05), "
            f"but dispatch was called"
        )
