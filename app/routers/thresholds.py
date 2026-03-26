from __future__ import annotations
from fastapi import APIRouter, Request
from app.models.threshold import ThresholdChangeEvent, ThresholdOverrideRequest, ThresholdSimulation, ThresholdState

router = APIRouter(prefix="/thresholds", tags=["Thresholds"])


@router.get("/current", response_model=ThresholdState)
async def get_current(request: Request) -> ThresholdState:
    return request.app.state.threshold_controller.get_current()


@router.post("/override", response_model=ThresholdState)
async def set_override(body: ThresholdOverrideRequest, request: Request) -> ThresholdState:
    ctrl = request.app.state.threshold_controller
    ctrl.set_override(value=body.value, reason=body.reason, expiry=body.expiry)
    return ctrl.get_current()


@router.get("/simulation", response_model=ThresholdSimulation)
async def simulate(proposed: float, request: Request) -> ThresholdSimulation:
    return request.app.state.threshold_controller.simulate(proposed)


@router.get("/history", response_model=list[ThresholdChangeEvent])
async def get_history(request: Request) -> list[ThresholdChangeEvent]:
    return request.app.state.threshold_controller.get_history()
