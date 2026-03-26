from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.models.copilot import SimilarCasesResult

router = APIRouter(prefix="/copilot", tags=["Copilot"])


async def _sse_generator(async_iter):
    async for chunk in async_iter:
        yield f"data: {chunk}\n\n"


@router.post("/explain/{case_id}")
async def explain(case_id: str, request: Request) -> StreamingResponse:
    copilot = request.app.state.ai_copilot
    return StreamingResponse(
        _sse_generator(copilot.explain(case_id)),
        media_type="text/event-stream",
    )


@router.post("/suggest/{case_id}")
async def suggest(case_id: str, request: Request) -> StreamingResponse:
    copilot = request.app.state.ai_copilot
    return StreamingResponse(
        _sse_generator(copilot.suggest(case_id)),
        media_type="text/event-stream",
    )


@router.post("/compare/{case_id}", response_model=SimilarCasesResult)
async def compare(case_id: str, request: Request) -> SimilarCasesResult:
    copilot = request.app.state.ai_copilot
    return await copilot.compare(case_id)
