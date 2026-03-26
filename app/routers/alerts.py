from __future__ import annotations
from fastapi import APIRouter, Request
from app.models.alert import AlertRecord, ChannelTestResult

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/history", response_model=list[AlertRecord])
async def get_history(request: Request) -> list[AlertRecord]:
    return request.app.state.alert_router.get_history()


@router.post("/test", response_model=list[ChannelTestResult])
async def send_test(request: Request) -> list[ChannelTestResult]:
    return request.app.state.alert_router.send_test()
