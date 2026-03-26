from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class IdentityCluster(BaseModel):
    cluster_id: str
    member_user_ids: list[str]
    shared_signals: dict  # {ips: [...], wallets: [...], devices: [...]}
    cluster_risk_score: float
    created_at: datetime
    updated_at: datetime


class ClusterDiff(BaseModel):
    new_clusters: list[str]       # cluster_ids
    merged_clusters: list[str]
    dissolved_clusters: list[str]
    computed_at: datetime


class ClusterStats(BaseModel):
    total_clusters: int
    average_cluster_size: float
    high_risk_cluster_count: int
    computed_at: datetime
