"""Stream router — health endpoint for the streaming pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.services.stream_consumer import StreamHealth

router = APIRouter(prefix="/stream", tags=["Streaming"])


@router.get("/health", response_model=StreamHealth)
async def stream_health(request: Request) -> StreamHealth:
    consumer = request.app.state.stream_consumer
    return consumer.get_health()
