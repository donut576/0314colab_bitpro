from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class ThresholdState(BaseModel):
    high_threshold: float
    medium_threshold: float
    last_updated: datetime
    last_change_reason: str
    is_override: bool = False
    override_expiry: datetime | None = None


class ThresholdSimulation(BaseModel):
    proposed_threshold: float
    estimated_alert_volume: int
    estimated_recall: float
    estimated_precision: float
    computed_at: datetime


class ThresholdChangeEvent(BaseModel):
    threshold_type: str   # "HIGH" or "MEDIUM"
    old_value: float
    new_value: float
    reason: str
    operator: str | None
    expiry: datetime | None
    created_at: datetime


class ThresholdOverrideRequest(BaseModel):
    value: float
    reason: str
    expiry: datetime
