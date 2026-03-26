from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class AlertStatus(str, Enum):
    DELIVERED = "delivered"
    FAILED = "failed"
    QUEUED = "queued"
    SUPPRESSED = "suppressed"


class RiskAlert(BaseModel):
    case_id: str
    user_id: str
    risk_score: float
    risk_level: str
    top_signals: list[str]
    timestamp: datetime
    deep_link: str = ""


class AlertRecord(BaseModel):
    alert_id: str
    case_id: str
    user_id: str
    channel: str
    status: AlertStatus
    risk_score: float
    risk_level: str
    timestamp: datetime
    error_message: str | None = None


class ChannelTestResult(BaseModel):
    channel: str
    success: bool
    message: str
