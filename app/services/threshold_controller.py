from __future__ import annotations
import logging
import os
from datetime import datetime
from app.models.threshold import ThresholdChangeEvent, ThresholdSimulation, ThresholdState

logger = logging.getLogger(__name__)


class ThresholdController:
    def __init__(self) -> None:
        self._high = float(os.getenv("THRESHOLD_HIGH_FLOOR", "0.7"))
        self._medium = float(os.getenv("THRESHOLD_MEDIUM_FLOOR", "0.4"))
        self._high_floor = float(os.getenv("THRESHOLD_HIGH_FLOOR", "0.5"))
        self._high_ceiling = float(os.getenv("THRESHOLD_HIGH_CEILING", "0.95"))
        self._max_queue = int(os.getenv("THRESHOLD_MAX_QUEUE", "500"))
        self._min_queue = int(os.getenv("THRESHOLD_MIN_QUEUE", "50"))
        self._is_override = False
        self._override_expiry: datetime | None = None
        self._last_reason = "initial"
        self._last_updated = datetime.utcnow()
        self._history: list[ThresholdChangeEvent] = []

    def get_current(self) -> ThresholdState:
        return ThresholdState(
            high_threshold=self._high,
            medium_threshold=self._medium,
            last_updated=self._last_updated,
            last_change_reason=self._last_reason,
            is_override=self._is_override,
            override_expiry=self._override_expiry,
        )

    def set_override(self, value: float, reason: str, expiry: datetime, operator: str | None = None) -> None:
        old = self._high
        self._high = max(self._high_floor, min(self._high_ceiling, value))
        self._is_override = True
        self._override_expiry = expiry
        self._last_reason = reason
        self._last_updated = datetime.utcnow()
        self._history.append(ThresholdChangeEvent(
            threshold_type="HIGH",
            old_value=old,
            new_value=self._high,
            reason=reason,
            operator=operator,
            expiry=expiry,
            created_at=self._last_updated,
        ))
        logger.info("Threshold override set: %.3f (reason=%s, expiry=%s)", self._high, reason, expiry)

    def tick(self, queue_depth: int = 0) -> None:
        """Adaptive logic: adjust threshold based on queue depth; revert expired overrides."""
        now = datetime.utcnow()

        # Revert expired override
        if self._is_override and self._override_expiry and now >= self._override_expiry:
            old = self._high
            self._is_override = False
            self._override_expiry = None
            self._last_reason = "override_expired_reversion"
            self._last_updated = now
            self._history.append(ThresholdChangeEvent(
                threshold_type="HIGH",
                old_value=old,
                new_value=self._high,
                reason="override_expired_reversion",
                operator=None,
                expiry=None,
                created_at=now,
            ))
            logger.info("Threshold override expired; reverted to %.3f", self._high)
            return

        if self._is_override:
            return  # Don't adapt while override is active

        old = self._high
        if queue_depth > self._max_queue:
            # Raise threshold to reduce alert volume
            self._high = min(self._high_ceiling, self._high + 0.02)
            reason = f"queue_depth={queue_depth} > max={self._max_queue}"
        elif queue_depth < self._min_queue:
            # Lower threshold toward model-optimal
            self._high = max(self._high_floor, self._high - 0.01)
            reason = f"queue_depth={queue_depth} < min={self._min_queue}"
        else:
            return  # No change needed

        if self._high != old:
            self._last_reason = reason
            self._last_updated = now
            self._history.append(ThresholdChangeEvent(
                threshold_type="HIGH",
                old_value=old,
                new_value=self._high,
                reason=reason,
                operator=None,
                expiry=None,
                created_at=now,
            ))

    def simulate(self, proposed: float) -> ThresholdSimulation:
        """Estimate alert volume/recall/precision for a proposed threshold."""
        # Stub: production would query last 7 days of predictions
        estimated_volume = max(0, int((1.0 - proposed) * 1000))
        estimated_recall = max(0.0, min(1.0, 1.0 - proposed))
        estimated_precision = max(0.0, min(1.0, proposed))
        return ThresholdSimulation(
            proposed_threshold=proposed,
            estimated_alert_volume=estimated_volume,
            estimated_recall=round(estimated_recall, 3),
            estimated_precision=round(estimated_precision, 3),
            computed_at=datetime.utcnow(),
        )

    def get_history(self, limit: int = 30) -> list[ThresholdChangeEvent]:
        return self._history[-limit:]
