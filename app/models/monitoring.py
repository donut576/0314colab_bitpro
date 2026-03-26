from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class ModelMetrics(BaseModel):
    model_name: str
    prediction_count: int
    avg_risk_score: float
    high_count: int
    medium_count: int
    low_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    rolling_f1_7d: float | None
    rolling_f1_30d: float | None
    baseline_f1: float | None
    updated_at: datetime


class SLAStatus(BaseModel):
    component: str
    metric: str
    current_value: float
    threshold: float
    breached: bool


class CalibrationPoint(BaseModel):
    threshold: float
    precision: float
    recall: float


class CalibrationCurve(BaseModel):
    model_name: str
    points: list[CalibrationPoint]
    computed_at: datetime


class UnifiedHealthSummary(BaseModel):
    models: list[ModelMetrics]
    sla_statuses: list[SLAStatus]
    overall_healthy: bool
    computed_at: datetime
