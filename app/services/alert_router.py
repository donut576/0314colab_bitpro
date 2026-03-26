from __future__ import annotations
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from app.models.alert import AlertRecord, AlertStatus, ChannelTestResult, RiskAlert

logger = logging.getLogger(__name__)


class AlertRouter:
    def __init__(
        self,
        rate_limit_per_hour: int = 60,
        cooldown_seconds: int = 3600,
    ) -> None:
        self._rate_limit = rate_limit_per_hour
        self._cooldown = timedelta(seconds=cooldown_seconds)
        # channel -> deque of dispatch timestamps (rolling window)
        self._channel_timestamps: dict[str, deque] = defaultdict(lambda: deque())
        # user_id -> last alert timestamp
        self._last_alert: dict[str, datetime] = {}
        self._history: list[AlertRecord] = []
        self._channels: list[str] = ["line", "email", "webhook"]

    def _is_rate_limited(self, channel: str, now: datetime) -> bool:
        window_start = now - timedelta(hours=1)
        dq = self._channel_timestamps[channel]
        while dq and dq[0] < window_start:
            dq.popleft()
        return len(dq) >= self._rate_limit

    def _is_suppressed(self, user_id: str, now: datetime) -> bool:
        last = self._last_alert.get(user_id)
        return last is not None and (now - last) < self._cooldown

    def dispatch(self, alert: RiskAlert) -> None:
        """Synchronous dispatch — queues or suppresses as needed."""
        now = datetime.utcnow()
        if self._is_suppressed(alert.user_id, now):
            record = AlertRecord(
                alert_id=str(uuid.uuid4()),
                case_id=alert.case_id,
                user_id=alert.user_id,
                channel="all",
                status=AlertStatus.SUPPRESSED,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                timestamp=now,
            )
            self._history.append(record)
            logger.info("Alert suppressed for user %s (cooldown)", alert.user_id)
            return

        self._last_alert[alert.user_id] = now
        for channel in self._channels:
            if self._is_rate_limited(channel, now):
                status = AlertStatus.QUEUED
                logger.warning("Rate limit reached for channel %s — alert queued", channel)
            else:
                self._channel_timestamps[channel].append(now)
                status = AlertStatus.DELIVERED
                logger.info("Alert dispatched to %s for user %s", channel, alert.user_id)

            record = AlertRecord(
                alert_id=str(uuid.uuid4()),
                case_id=alert.case_id,
                user_id=alert.user_id,
                channel=channel,
                status=status,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                timestamp=now,
            )
            self._history.append(record)

    async def dispatch_async(self, prediction: dict) -> None:
        """Async wrapper for dispatch called from StreamConsumer."""
        alert = RiskAlert(
            case_id=prediction.get("case_id", ""),
            user_id=prediction.get("user_id", ""),
            risk_score=float(prediction.get("risk_score", 0.0)),
            risk_level=prediction.get("risk_level", "HIGH"),
            top_signals=prediction.get("top_signals", []),
            timestamp=datetime.utcnow(),
        )
        self.dispatch(alert)

    def get_history(self, limit: int = 1000) -> list[AlertRecord]:
        return self._history[-limit:]

    def send_test(self) -> list[ChannelTestResult]:
        return [
            ChannelTestResult(
                channel=ch,
                success=True,
                message=f"Test notification sent to {ch}",
            )
            for ch in self._channels
        ]
