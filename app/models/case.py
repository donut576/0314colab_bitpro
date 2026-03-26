from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class CaseStatus(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class ResolutionType(str, Enum):
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CaseAuditEntry(BaseModel):
    analyst_id: str
    action: str
    old_status: str | None
    new_status: str | None
    note: str | None
    created_at: datetime


class Case(BaseModel):
    case_id: str
    user_id: str
    risk_score: float
    risk_level: str
    status: CaseStatus
    assigned_analyst: str | None
    model_version: str
    shap_top_features: list[dict]
    cluster_id: str | None
    audit_trail: list[CaseAuditEntry]
    created_at: datetime
    updated_at: datetime


class CaseResolution(BaseModel):
    resolution_type: ResolutionType
    resolution_note: str


class CaseFilters(BaseModel):
    status: CaseStatus | None = None
    risk_level: str | None = None
    assigned_analyst: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class CaseStats(BaseModel):
    open_count: int
    avg_resolution_hours: float
    false_positive_rate: float
    confirmed_fraud_rate: float
    computed_at: datetime


class AssignRequest(BaseModel):
    analyst_id: str


class PaginatedCases(BaseModel):
    items: list[Case]
    total: int
    page: int
    page_size: int
