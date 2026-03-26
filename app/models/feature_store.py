"""Pydantic schemas for the Feature Store."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FeatureVector(BaseModel):
    user_id: str
    schema_version: str
    features: dict[str, float]
    groups: list[str]
    last_updated: datetime
    cold_start: bool = False


class FeatureStoreStats(BaseModel):
    total_users: int
    schema_version: str
    last_updated: datetime
    storage_utilization_bytes: int


class BatchFeaturesRequest(BaseModel):
    user_ids: list[str]
