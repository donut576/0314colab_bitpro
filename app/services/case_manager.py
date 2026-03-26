from __future__ import annotations
import logging
import uuid
from datetime import datetime
from app.models.case import (
    AssignRequest,
    Case,
    CaseAuditEntry,
    CaseFilters,
    CaseResolution,
    CaseStats,
    CaseStatus,
    PaginatedCases,
    ResolutionType,
)

logger = logging.getLogger(__name__)

# Valid status transitions
VALID_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.OPEN: {CaseStatus.IN_REVIEW, CaseStatus.ESCALATED},
    CaseStatus.IN_REVIEW: {CaseStatus.RESOLVED},
    CaseStatus.ESCALATED: {CaseStatus.RESOLVED},
    CaseStatus.RESOLVED: set(),
}


class CaseManager:
    def __init__(self, high_threshold: float = 0.7) -> None:
        self._cases: dict[str, Case] = {}          # case_id -> Case
        self._user_open_case: dict[str, str] = {}  # user_id -> case_id (open only)
        self._confirmed_fraud_registry: set[str] = set()
        self._high_threshold = high_threshold
        self._alert_router = None  # injected after init

    def set_alert_router(self, alert_router) -> None:
        self._alert_router = alert_router

    def create_case(self, prediction: dict) -> Case:
        """Create a new case or append to existing open case for the user."""
        user_id = prediction.get("user_id", "")
        existing_id = self._user_open_case.get(user_id)
        if existing_id and existing_id in self._cases:
            existing = self._cases[existing_id]
            entry = CaseAuditEntry(
                analyst_id="system",
                action="prediction_appended",
                old_status=existing.status.value,
                new_status=existing.status.value,
                note=f"risk_score={prediction.get('risk_score', 0.0):.3f}",
                created_at=datetime.utcnow(),
            )
            updated = existing.model_copy(update={
                "audit_trail": existing.audit_trail + [entry],
                "updated_at": datetime.utcnow(),
            })
            self._cases[existing_id] = updated
            logger.info("Appended prediction to existing case %s for user %s", existing_id, user_id)
            return updated

        case_id = str(uuid.uuid4())
        now = datetime.utcnow()
        case = Case(
            case_id=case_id,
            user_id=user_id,
            risk_score=float(prediction.get("risk_score", 0.0)),
            risk_level=prediction.get("risk_level", "HIGH"),
            status=CaseStatus.OPEN,
            assigned_analyst=None,
            model_version=prediction.get("model_version", "1.0"),
            shap_top_features=prediction.get("shap_top_features", []),
            cluster_id=prediction.get("cluster_id"),
            audit_trail=[CaseAuditEntry(
                analyst_id="system",
                action="case_created",
                old_status=None,
                new_status=CaseStatus.OPEN.value,
                note=None,
                created_at=now,
            )],
            created_at=now,
            updated_at=now,
        )
        self._cases[case_id] = case
        self._user_open_case[user_id] = case_id
        logger.info("Created case %s for user %s", case_id, user_id)
        return case

    def get_case(self, case_id: str) -> Case | None:
        return self._cases.get(case_id)

    def list_cases(self, filters: CaseFilters, page: int = 1, page_size: int = 20) -> PaginatedCases:
        items = list(self._cases.values())
        if filters.status:
            items = [c for c in items if c.status == filters.status]
        if filters.risk_level:
            items = [c for c in items if c.risk_level == filters.risk_level]
        if filters.assigned_analyst:
            items = [c for c in items if c.assigned_analyst == filters.assigned_analyst]
        if filters.date_from:
            items = [c for c in items if c.created_at >= filters.date_from]
        if filters.date_to:
            items = [c for c in items if c.created_at <= filters.date_to]
        total = len(items)
        start = (page - 1) * page_size
        return PaginatedCases(
            items=items[start:start + page_size],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_status(
        self,
        case_id: str,
        new_status: CaseStatus,
        analyst_id: str,
        note: str | None = None,
    ) -> Case:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Case {case_id} not found")
        allowed = VALID_TRANSITIONS.get(case.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition {case.status} -> {new_status}. Allowed: {allowed}"
            )
        entry = CaseAuditEntry(
            analyst_id=analyst_id,
            action="status_updated",
            old_status=case.status.value,
            new_status=new_status.value,
            note=note,
            created_at=datetime.utcnow(),
        )
        updated = case.model_copy(update={
            "status": new_status,
            "audit_trail": case.audit_trail + [entry],
            "updated_at": datetime.utcnow(),
        })
        self._cases[case_id] = updated
        if new_status != CaseStatus.OPEN:
            self._user_open_case.pop(case.user_id, None)
        return updated

    def assign(self, case_id: str, analyst_id: str) -> Case:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Case {case_id} not found")
        entry = CaseAuditEntry(
            analyst_id="supervisor",
            action="assigned",
            old_status=case.status.value,
            new_status=case.status.value,
            note=f"assigned_to={analyst_id}",
            created_at=datetime.utcnow(),
        )
        updated = case.model_copy(update={
            "assigned_analyst": analyst_id,
            "audit_trail": case.audit_trail + [entry],
            "updated_at": datetime.utcnow(),
        })
        self._cases[case_id] = updated
        return updated

    def resolve(self, case_id: str, resolution: CaseResolution) -> Case:
        case = self.update_status(
            case_id,
            CaseStatus.RESOLVED,
            analyst_id="system",
            note=resolution.resolution_note,
        )
        if resolution.resolution_type == ResolutionType.CONFIRMED_FRAUD:
            self._confirmed_fraud_registry.add(case.user_id)
            logger.info("User %s added to confirmed fraud registry", case.user_id)
        return case

    def get_stats(self, window_days: int = 30) -> CaseStats:
        cases = list(self._cases.values())
        open_count = sum(1 for c in cases if c.status == CaseStatus.OPEN)
        resolved = [c for c in cases if c.status == CaseStatus.RESOLVED]
        confirmed = sum(
            1 for c in resolved
            if any("confirmed_fraud" in (e.note or "") for e in c.audit_trail)
        )
        fp = sum(
            1 for c in resolved
            if any("false_positive" in (e.note or "") for e in c.audit_trail)
        )
        total_resolved = len(resolved)
        return CaseStats(
            open_count=open_count,
            avg_resolution_hours=0.0,
            false_positive_rate=fp / max(total_resolved, 1),
            confirmed_fraud_rate=confirmed / max(total_resolved, 1),
            computed_at=datetime.utcnow(),
        )
