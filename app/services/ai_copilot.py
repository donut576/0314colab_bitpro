from __future__ import annotations
import logging
import os
import re
import time
from datetime import datetime
from typing import AsyncIterator, TYPE_CHECKING

from app.models.copilot import CopilotExplanation, CopilotSuggestion, SimilarCase, SimilarCasesResult

if TYPE_CHECKING:
    from app.services.case_manager import CaseManager

logger = logging.getLogger(__name__)

# PII patterns to strip before sending to LLM
_PII_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),       # Full names (simple heuristic)
    re.compile(r'\b[A-Z]\d{9}\b'),                      # National ID (e.g. A123456789)
    re.compile(r'\b09\d{8}\b'),                         # TW mobile numbers
    re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),    # Phone numbers
]


def _strip_pii(text: str) -> str:
    """Replace PII-like patterns with [REDACTED]."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub('[REDACTED]', text)
    return text


class AICopilot:
    MODEL_VERSION = "copilot-1.0"

    def __init__(self, case_manager: "CaseManager") -> None:
        self._case_manager = case_manager
        self._provider = os.getenv("COPILOT_LLM_PROVIDER", "ollama")

    def _build_case_context(self, case_id: str) -> str:
        """Build a PII-safe context string from case data."""
        case = self._case_manager.get_case(case_id)
        if case is None:
            return f"Case {case_id}: not found."
        shap_summary = ", ".join(
            f"{f.get('feature', '?')}={f.get('value', '?')}"
            for f in (case.shap_top_features or [])[:3]
        )
        context = (
            f"Case ID: {case.case_id}\n"
            f"User ID: {case.user_id}\n"
            f"Risk Score: {case.risk_score:.3f}\n"
            f"Risk Level: {case.risk_level}\n"
            f"Status: {case.status.value}\n"
            f"Top SHAP features: {shap_summary or 'N/A'}\n"
            f"Model version: {case.model_version}\n"
        )
        return _strip_pii(context)

    async def explain(self, case_id: str) -> AsyncIterator[str]:
        """Stream a natural-language explanation for the case."""
        start = time.monotonic()
        context = self._build_case_context(case_id)
        case = self._case_manager.get_case(case_id)
        user_ref = case.user_id if case else "unknown"
        # Fallback: structured template (no LLM call in stub)
        explanation = (
            f"This account (user_id: {user_ref}) "
            f"was flagged with a risk score based on the following signals:\n{context}\n"
            f"The model identified anomalous patterns consistent with mule account behavior."
        )
        explanation = _strip_pii(explanation)
        latency_ms = (time.monotonic() - start) * 1000
        # Yield as SSE chunks
        for chunk in explanation.split(". "):
            yield chunk + ". "

    async def suggest(self, case_id: str) -> AsyncIterator[str]:
        """Stream investigation suggestions for the case."""
        suggestions = [
            "1. Review the user's transaction history for rapid fund movement patterns.",
            "2. Check for shared IP addresses or device fingerprints with known fraud accounts.",
            "3. Verify KYC documents and submission timing for inconsistencies.",
            "4. Examine withdrawal wallet addresses against known blacklists.",
            "5. Escalate to senior analyst if cluster risk score exceeds HIGH threshold.",
        ]
        for s in suggestions:
            yield s + "\n"

    async def compare(self, case_id: str) -> SimilarCasesResult:
        """Return top-3 most similar historical resolved cases (stub)."""
        # Production: compute embedding similarity against resolved cases
        similar = [
            SimilarCase(
                case_id=f"historical-{i}",
                similarity_score=round(0.9 - i * 0.1, 2),
                resolution_type="confirmed_fraud",
                resolution_summary="Account exhibited rapid fund movement and shared IP with known mule network.",
            )
            for i in range(3)
        ]
        return SimilarCasesResult(
            case_id=case_id,
            similar_cases=similar,
            generated_at=datetime.utcnow(),
        )
