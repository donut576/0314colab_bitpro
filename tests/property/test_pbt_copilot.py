"""Property-based tests for the AI Copilot PII exclusion.

# Feature: aml-advanced-features, Property 17: copilot PII exclusion
"""

from __future__ import annotations

import asyncio
import re

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.services.ai_copilot import _strip_pii, AICopilot
from app.services.case_manager import CaseManager


# ---------------------------------------------------------------------------
# PII regex patterns (mirrors ai_copilot.py)
# ---------------------------------------------------------------------------

_FULL_NAME_RE = re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')
_NATIONAL_ID_RE = re.compile(r'\b[A-Z]\d{9}\b')
_TW_MOBILE_RE = re.compile(r'\b09\d{8}\b')
_PHONE_RE = re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')

_ALL_PII_PATTERNS = [_FULL_NAME_RE, _NATIONAL_ID_RE, _TW_MOBILE_RE, _PHONE_RE]


# ---------------------------------------------------------------------------
# Strategies — generate PII-like strings
# ---------------------------------------------------------------------------

_first_names = st.sampled_from(["John", "Alice", "Bob", "Carol", "David", "Emily"])
_last_names = st.sampled_from(["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis"])

_full_names = st.tuples(_first_names, _last_names).map(lambda t: f"{t[0]} {t[1]}")

_national_ids = st.tuples(
    st.sampled_from(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    st.from_regex(r"\d{9}", fullmatch=True),
).map(lambda t: f"{t[0]}{t[1]}")

_tw_mobiles = st.from_regex(r"09\d{8}", fullmatch=True)

_phone_separators = st.sampled_from(["-", ".", " "])
_phone_numbers = st.tuples(
    st.from_regex(r"\d{3}", fullmatch=True),
    _phone_separators,
    st.from_regex(r"\d{3}", fullmatch=True),
    st.from_regex(r"\d{4}", fullmatch=True),
).map(lambda t: f"{t[0]}{t[1]}{t[2]}{t[1]}{t[3]}")

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=40,
)


# ---------------------------------------------------------------------------
# Property 17 — _strip_pii direct tests
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(name=_full_names, surrounding=_safe_text)
def test_property_17_strip_pii_removes_full_names(name: str, surrounding: str):
    """_strip_pii must redact full-name patterns from any text.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    text = f"User {name} performed a transaction. {surrounding}"
    result = _strip_pii(text)

    assert name not in result, f"Full name '{name}' was not redacted from output"
    assert "[REDACTED]" in result, "Expected [REDACTED] placeholder in output"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(nid=_national_ids, surrounding=_safe_text)
def test_property_17_strip_pii_removes_national_ids(nid: str, surrounding: str):
    """_strip_pii must redact national ID patterns from any text.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    text = f"National ID: {nid} {surrounding}"
    result = _strip_pii(text)

    assert nid not in result, f"National ID '{nid}' was not redacted from output"
    assert "[REDACTED]" in result, "Expected [REDACTED] placeholder in output"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(mobile=_tw_mobiles, surrounding=_safe_text)
def test_property_17_strip_pii_removes_tw_mobiles(mobile: str, surrounding: str):
    """_strip_pii must redact TW mobile number patterns from any text.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    text = f"Contact: {mobile} {surrounding}"
    result = _strip_pii(text)

    assert mobile not in result, f"TW mobile '{mobile}' was not redacted from output"
    assert "[REDACTED]" in result, "Expected [REDACTED] placeholder in output"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(phone=_phone_numbers, surrounding=_safe_text)
def test_property_17_strip_pii_removes_phone_numbers(phone: str, surrounding: str):
    """_strip_pii must redact phone number patterns from any text.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    text = f"Phone: {phone} {surrounding}"
    result = _strip_pii(text)

    assert phone not in result, f"Phone number '{phone}' was not redacted from output"
    assert "[REDACTED]" in result, "Expected [REDACTED] placeholder in output"


# ---------------------------------------------------------------------------
# Property 17 — mixed PII in a single string
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    name=_full_names,
    nid=_national_ids,
    mobile=_tw_mobiles,
    phone=_phone_numbers,
)
def test_property_17_strip_pii_removes_all_pii_types(
    name: str, nid: str, mobile: str, phone: str,
):
    """_strip_pii must redact ALL PII types when they appear together.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    text = f"User {name}, ID {nid}, mobile {mobile}, phone {phone} flagged."
    result = _strip_pii(text)

    for pii_pattern in _ALL_PII_PATTERNS:
        assert not pii_pattern.search(result), (
            f"PII pattern {pii_pattern.pattern} still found in: {result}"
        )


# ---------------------------------------------------------------------------
# Property 17 — full explain() flow with PII in case data
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    name=_full_names,
    nid=_national_ids,
    phone=_phone_numbers,
)
def test_property_17_explain_flow_excludes_pii(
    name: str, nid: str, phone: str,
):
    """Full explain() flow: PII injected into case shap_top_features must not
    appear in the streamed output.

    # Feature: aml-advanced-features, Property 17: copilot PII exclusion

    **Validates: Requirements REQ-A7.5**
    """
    cm = CaseManager()
    # Create a case with PII embedded in shap_top_features
    prediction = {
        "user_id": "user_12345",
        "risk_score": 0.92,
        "risk_level": "HIGH",
        "model_version": "1.0",
        "shap_top_features": [
            {"feature": f"contact_{name}", "value": 0.8},
            {"feature": f"id_{nid}", "value": 0.6},
            {"feature": f"phone_{phone}", "value": 0.4},
        ],
        "cluster_id": None,
    }
    case = cm.create_case(prediction)

    copilot = AICopilot(case_manager=cm)

    # Collect all chunks from the async generator
    chunks: list[str] = []

    async def _collect():
        async for chunk in copilot.explain(case.case_id):
            chunks.append(chunk)

    asyncio.get_event_loop().run_until_complete(_collect())

    full_output = "".join(chunks)

    # Assert no PII patterns remain in the output
    for pii_pattern in _ALL_PII_PATTERNS:
        assert not pii_pattern.search(full_output), (
            f"PII pattern {pii_pattern.pattern} found in explain() output: {full_output}"
        )

    # Assert user_id reference is present (user references use user_id only)
    assert "user_12345" in full_output or "user_id" in full_output, (
        "Expected user_id reference in explain() output"
    )
