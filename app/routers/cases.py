from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, Request
from app.models.case import (
    AssignRequest,
    Case,
    CaseFilters,
    CaseResolution,
    CaseStats,
    CaseStatus,
    PaginatedCases,
)

router = APIRouter(prefix="/cases", tags=["Cases"])


@router.get("/stats", response_model=CaseStats)
async def get_stats(request: Request) -> CaseStats:
    return request.app.state.case_manager.get_stats()


@router.get("", response_model=PaginatedCases)
async def list_cases(
    request: Request,
    status: CaseStatus | None = None,
    risk_level: str | None = None,
    assigned_analyst: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PaginatedCases:
    filters = CaseFilters(
        status=status,
        risk_level=risk_level,
        assigned_analyst=assigned_analyst,
    )
    return request.app.state.case_manager.list_cases(filters, page=page, page_size=page_size)


@router.get("/{case_id}", response_model=Case)
async def get_case(case_id: str, request: Request) -> Case:
    case = request.app.state.case_manager.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return case


@router.post("/{case_id}/assign", response_model=Case)
async def assign_case(case_id: str, body: AssignRequest, request: Request) -> Case:
    try:
        return request.app.state.case_manager.assign(case_id, body.analyst_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{case_id}/resolve", response_model=Case)
async def resolve_case(case_id: str, body: CaseResolution, request: Request) -> Case:
    try:
        return request.app.state.case_manager.resolve(case_id, body)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
