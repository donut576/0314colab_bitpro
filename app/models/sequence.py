from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class AnomalousEvent(BaseModel):
    timestamp: datetime
    channel: str
    amount: float
    anomaly_score: float


class SequenceScore(BaseModel):
    user_id: str
    sequence_anomaly_score: float | None  # None when insufficient_history=True
    top_anomalous_events: list[AnomalousEvent]
    model_version: str
    insufficient_history: bool = False


class BehavioralProfile(BaseModel):
    user_id: str
    transaction_count: int
    lookback_days: int
    last_scored: datetime | None
    recent_events: list[AnomalousEvent]  # last 10


class ScoreRequest(BaseModel):
    user_id: str
