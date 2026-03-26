from __future__ import annotations
from fastapi import APIRouter, Request
from app.models.monitoring import CalibrationCurve, UnifiedHealthSummary

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/dashboard", response_model=UnifiedHealthSummary)
async def get_dashboard(request: Request) -> UnifiedHealthSummary:
    return request.app.state.monitoring_system.get_dashboard()


@router.get("/model/{model_name}/calibration", response_model=CalibrationCurve)
async def get_calibration(model_name: str, request: Request) -> CalibrationCurve:
    return request.app.state.monitoring_system.get_model_calibration(model_name)
