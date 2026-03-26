from __future__ import annotations
import logging
from datetime import datetime
from typing import TYPE_CHECKING
from app.models.monitoring import (
    CalibrationCurve, CalibrationPoint, ModelMetrics,
    SLAStatus, UnifiedHealthSummary,
)

if TYPE_CHECKING:
    from app.services.alert_router import AlertRouter

logger = logging.getLogger(__name__)

_MODEL_NAMES = ["xgboost", "graph", "sequence"]

# SLA thresholds
_SLA_THRESHOLDS = {
    "streaming_p95_latency_ms": 5000.0,
    "feature_store_p99_latency_ms": 20.0,
    "case_manager_p95_latency_ms": 500.0,
}


class MonitoringSystem:
    def __init__(self, alert_router: AlertRouter | None = None) -> None:
        self._alert_router = alert_router
        self._metrics: dict[str, ModelMetrics] = {
            name: ModelMetrics(
                model_name=name,
                prediction_count=0,
                avg_risk_score=0.0,
                high_count=0,
                medium_count=0,
                low_count=0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                rolling_f1_7d=None,
                rolling_f1_30d=None,
                baseline_f1=None,
                updated_at=datetime.utcnow(),
            )
            for name in _MODEL_NAMES
        }
        # Simulated SLA values (production: read from metrics store)
        self._sla_values: dict[str, float] = {
            "streaming_p95_latency_ms": 0.0,
            "feature_store_p99_latency_ms": 0.0,
            "case_manager_p95_latency_ms": 0.0,
        }

    def record_prediction(self, model_name: str, risk_score: float, latency_ms: float) -> None:
        """Update per-model metrics with a new prediction."""
        if model_name not in self._metrics:
            return
        m = self._metrics[model_name]
        n = m.prediction_count + 1
        new_avg = (m.avg_risk_score * m.prediction_count + risk_score) / n
        high = m.high_count + (1 if risk_score >= 0.7 else 0)
        medium = m.medium_count + (1 if 0.4 <= risk_score < 0.7 else 0)
        low = m.low_count + (1 if risk_score < 0.4 else 0)
        self._metrics[model_name] = m.model_copy(update={
            "prediction_count": n,
            "avg_risk_score": round(new_avg, 4),
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
            "p95_latency_ms": latency_ms,  # simplified: last value
            "updated_at": datetime.utcnow(),
        })

    def _check_f1_degradation(self) -> None:
        """Emit alert if any model's rolling F1 drops > 0.05 below baseline."""
        if self._alert_router is None:
            return
        for m in self._metrics.values():
            if m.baseline_f1 is not None and m.rolling_f1_7d is not None:
                if m.baseline_f1 - m.rolling_f1_7d > 0.05:
                    from app.models.alert import RiskAlert
                    alert = RiskAlert(
                        case_id="monitoring",
                        user_id="system",
                        risk_score=1.0,
                        risk_level="HIGH",
                        top_signals=[f"model_degradation:{m.model_name}"],
                        timestamp=datetime.utcnow(),
                    )
                    self._alert_router.dispatch(alert)
                    logger.warning("F1 degradation alert for model %s", m.model_name)

    def _check_sla(self) -> list[SLAStatus]:
        statuses = []
        for metric, threshold in _SLA_THRESHOLDS.items():
            current = self._sla_values.get(metric, 0.0)
            breached = current > threshold
            statuses.append(SLAStatus(
                component=metric.split("_")[0],
                metric=metric,
                current_value=current,
                threshold=threshold,
                breached=breached,
            ))
            if breached:
                logger.warning("SLA breach: %s=%.1f > %.1f", metric, current, threshold)
        return statuses

    def tick(self) -> None:
        """Periodic tick: check F1 degradation and SLA breaches."""
        self._check_f1_degradation()
        self._check_sla()
        logger.debug("MonitoringSystem tick completed")

    def get_dashboard(self) -> UnifiedHealthSummary:
        sla_statuses = self._check_sla()
        overall_healthy = not any(s.breached for s in sla_statuses)
        return UnifiedHealthSummary(
            models=list(self._metrics.values()),
            sla_statuses=sla_statuses,
            overall_healthy=overall_healthy,
            computed_at=datetime.utcnow(),
        )

    def get_model_calibration(self, model_name: str) -> CalibrationCurve:
        """Return calibration curve at 20 threshold points (stub)."""
        points = [
            CalibrationPoint(
                threshold=round(t / 20, 2),
                precision=round(0.5 + t / 40, 3),
                recall=round(1.0 - t / 20, 3),
            )
            for t in range(20)
        ]
        return CalibrationCurve(
            model_name=model_name,
            points=points,
            computed_at=datetime.utcnow(),
        )
