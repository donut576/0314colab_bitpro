"""Property-based tests for the Case Manager.

# Feature: aml-advanced-features, Property 10: case deduplication
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.models.case import CaseStatus
from app.services.case_manager import CaseManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
)

_num_predictions = st.integers(min_value=2, max_value=10)

_risk_scores = st.floats(min_value=0.7, max_value=1.0, allow_nan=False, allow_infinity=False)


def _build_prediction(user_id: str, risk_score: float, idx: int) -> dict:
    """Build a HIGH-risk prediction dict for the given user."""
    return {
        "user_id": user_id,
        "risk_score": risk_score,
        "risk_level": "HIGH",
        "model_version": f"1.0.{idx}",
        "shap_top_features": [],
        "cluster_id": None,
    }


# ---------------------------------------------------------------------------
# Property 10: Case deduplication
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(user_id=_user_ids, num_preds=_num_predictions, scores=st.lists(
    _risk_scores, min_size=10, max_size=10,
))
def test_property_10_case_deduplication(user_id: str, num_preds: int, scores: list[float]):
    """Property 10: Case deduplication — generate multiple HIGH predictions for
    same user, assert only one open case exists.

    # Feature: aml-advanced-features, Property 10: case deduplication

    **Validates: Requirements REQ-A6.11**
    """
    cm = CaseManager()

    # Use only the first num_preds scores from the fixed-size list
    pred_scores = scores[:num_preds]

    returned_cases = []
    for idx, score in enumerate(pred_scores):
        prediction = _build_prediction(user_id, score, idx)
        case = cm.create_case(prediction)
        returned_cases.append(case)

    # 1. All returned cases must share the same case_id (deduplication)
    case_ids = {c.case_id for c in returned_cases}
    assert len(case_ids) == 1, (
        f"Expected exactly 1 unique case_id for user '{user_id}', "
        f"got {len(case_ids)}: {case_ids}"
    )

    # 2. Count open cases for this user in _cases — must be exactly 1
    open_cases_for_user = [
        c for c in cm._cases.values()
        if c.user_id == user_id and c.status == CaseStatus.OPEN
    ]
    assert len(open_cases_for_user) == 1, (
        f"Expected exactly 1 open case for user '{user_id}', "
        f"got {len(open_cases_for_user)}"
    )

    # 3. The case's audit_trail must have entries for each appended prediction
    the_case = open_cases_for_user[0]
    # First prediction creates the case (action="case_created"),
    # subsequent predictions append (action="prediction_appended")
    created_entries = [e for e in the_case.audit_trail if e.action == "case_created"]
    appended_entries = [e for e in the_case.audit_trail if e.action == "prediction_appended"]

    assert len(created_entries) == 1, (
        f"Expected exactly 1 'case_created' audit entry, got {len(created_entries)}"
    )
    assert len(appended_entries) == num_preds - 1, (
        f"Expected {num_preds - 1} 'prediction_appended' audit entries, "
        f"got {len(appended_entries)}"
    )

    # Total audit entries = 1 (created) + (num_preds - 1) (appended)
    assert len(the_case.audit_trail) == num_preds, (
        f"Expected {num_preds} total audit entries, got {len(the_case.audit_trail)}"
    )


# ---------------------------------------------------------------------------
# Helpers for Property 11
# ---------------------------------------------------------------------------

# Mirror of the valid transitions from case_manager.py
_VALID_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.OPEN: {CaseStatus.IN_REVIEW, CaseStatus.ESCALATED},
    CaseStatus.IN_REVIEW: {CaseStatus.RESOLVED},
    CaseStatus.ESCALATED: {CaseStatus.RESOLVED},
    CaseStatus.RESOLVED: set(),
}

# Paths to reach each status from OPEN via valid transitions
_PATH_TO_STATUS: dict[CaseStatus, list[CaseStatus]] = {
    CaseStatus.OPEN: [],
    CaseStatus.IN_REVIEW: [CaseStatus.IN_REVIEW],
    CaseStatus.ESCALATED: [CaseStatus.ESCALATED],
    CaseStatus.RESOLVED: [CaseStatus.ESCALATED, CaseStatus.RESOLVED],
}


def _create_case_at_status(cm: CaseManager, status: CaseStatus) -> str:
    """Create a case and transition it to the desired status, returning case_id."""
    import uuid
    prediction = {
        "user_id": f"user-{uuid.uuid4().hex[:8]}",
        "risk_score": 0.95,
        "risk_level": "HIGH",
        "model_version": "1.0",
        "shap_top_features": [],
        "cluster_id": None,
    }
    case = cm.create_case(prediction)
    for intermediate in _PATH_TO_STATUS[status]:
        cm.update_status(case.case_id, intermediate, "setup-analyst", "setup")
    return case.case_id


# ---------------------------------------------------------------------------
# Property 11: Case status transition validity
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    current_status=st.sampled_from(list(CaseStatus)),
    target_status=st.sampled_from(list(CaseStatus)),
)
def test_property_11_case_status_transition_validity(
    current_status: CaseStatus,
    target_status: CaseStatus,
):
    """Property 11: Case status transition validity — generate random transition
    attempts, assert only valid transitions succeed and invalid ones raise ValueError.

    # Feature: aml-advanced-features, Property 11: case status transition validity

    **Validates: Requirements REQ-A6.3**
    """
    cm = CaseManager()
    case_id = _create_case_at_status(cm, current_status)

    # Verify the case is at the expected current status
    case = cm.get_case(case_id)
    assert case is not None
    assert case.status == current_status

    is_valid = target_status in _VALID_TRANSITIONS[current_status]

    if is_valid:
        updated = cm.update_status(case_id, target_status, "analyst-1", "transition test")
        assert updated.status == target_status, (
            f"Expected status {target_status} after valid transition "
            f"{current_status} -> {target_status}, got {updated.status}"
        )
    else:
        try:
            cm.update_status(case_id, target_status, "analyst-1", "transition test")
            # If we reach here, the transition should have been rejected
            assert False, (
                f"Expected ValueError for invalid transition "
                f"{current_status} -> {target_status}, but it succeeded"
            )
        except ValueError:
            # Invalid transition correctly rejected
            # Verify the case status is unchanged
            unchanged = cm.get_case(case_id)
            assert unchanged.status == current_status, (
                f"Case status changed to {unchanged.status} despite ValueError "
                f"for invalid transition {current_status} -> {target_status}"
            )
