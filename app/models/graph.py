from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class TransactionEdge(BaseModel):
    sender_user_id: str
    receiver_user_id: str
    amount: float
    timestamp: datetime
    channel: str


class GraphScore(BaseModel):
    user_id: str
    graph_risk_score: float  # [0, 1]
    embedding: list[float]
    hop1_count: int
    hop2_count: int
    betweenness_centrality: float
    elevated: bool  # True if neighborhood HIGH fraction exceeded threshold


class SubgraphResult(BaseModel):
    user_id: str
    nodes: list[dict]   # node-link format: [{id, risk_level, ...}]
    edges: list[dict]   # [{source, target, amount, ...}]
    hops: int


class GraphScoreRequest(BaseModel):
    user_ids: list[str]
