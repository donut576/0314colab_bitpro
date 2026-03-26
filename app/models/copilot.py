from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class CopilotExplanation(BaseModel):
    case_id: str
    explanation: str
    model_version: str
    latency_ms: float
    is_fallback: bool = False
    generated_at: datetime


class CopilotSuggestion(BaseModel):
    case_id: str
    suggestions: list[str]
    model_version: str
    latency_ms: float
    is_fallback: bool = False
    generated_at: datetime


class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float
    resolution_type: str
    resolution_summary: str


class SimilarCasesResult(BaseModel):
    case_id: str
    similar_cases: list[SimilarCase]
    generated_at: datetime
