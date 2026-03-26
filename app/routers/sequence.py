from __future__ import annotations
from fastapi import APIRouter, Request
from app.models.sequence import BehavioralProfile, ScoreRequest, SequenceScore

router = APIRouter(prefix="/sequence", tags=["Sequence"])


@router.post("/score", response_model=SequenceScore)
async def score_sequence(body: ScoreRequest, request: Request) -> SequenceScore:
    scorer = request.app.state.sequence_scorer
    return scorer.score(body.user_id)


@router.get("/profile/{user_id}", response_model=BehavioralProfile)
async def get_profile(user_id: str, request: Request) -> BehavioralProfile:
    scorer = request.app.state.sequence_scorer
    return scorer.get_profile(user_id)
