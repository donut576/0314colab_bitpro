from __future__ import annotations
from fastapi import APIRouter, Request
from app.models.graph import GraphScore, GraphScoreRequest, SubgraphResult

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.post("/score", response_model=list[GraphScore])
async def graph_score(body: GraphScoreRequest, request: Request) -> list[GraphScore]:
    engine = request.app.state.graph_engine
    return [engine.get_score(uid) for uid in body.user_ids]


@router.get("/subgraph/{user_id}", response_model=SubgraphResult)
async def get_subgraph(
    user_id: str, request: Request, hops: int = 2
) -> SubgraphResult:
    engine = request.app.state.graph_engine
    return engine.get_subgraph(user_id, hops=hops)
