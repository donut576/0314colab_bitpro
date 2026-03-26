"""StreamConsumer — real-time transaction event processing pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.services.audit_logger import AuditLogger
    from app.services.ensemble_scorer import EnsembleScorer
    from app.services.feature_store import FeatureStore
    from app.services.predictor import XGBPredictor

logger = logging.getLogger(__name__)


class StreamHealth(BaseModel):
    broker_connected: bool
    consumer_lag: int
    events_per_second: float
    broker_type: str


class StreamConsumer:
    def __init__(
        self,
        broker_type: str,  # "kafka" or "kinesis"
        feature_store: "FeatureStore",
        predictor: "XGBPredictor",
        ensemble_scorer: "EnsembleScorer",
        audit_logger: "AuditLogger",
        alert_router: object,
        risk_threshold: float = 0.7,
    ) -> None:
        self._broker_type = broker_type
        self._feature_store = feature_store
        self._predictor = predictor
        self._ensemble_scorer = ensemble_scorer
        self._audit_logger = audit_logger
        self._alert_router = alert_router
        self._risk_threshold = risk_threshold
        self._running = False
        self._broker_connected = False
        self._buffer: deque = deque(maxlen=10000)
        self._events_processed = 0
        self._start_time: float = 0.0
        self._consumer_lag = 0

    def start(self) -> None:
        self._running = True
        self._start_time = time.monotonic()
        logger.info("StreamConsumer started (broker=%s)", self._broker_type)

    def stop(self) -> None:
        self._running = False
        logger.info("StreamConsumer stopped")

    def get_health(self) -> StreamHealth:
        elapsed = time.monotonic() - self._start_time if self._start_time else 1.0
        eps = self._events_processed / max(elapsed, 1.0)
        return StreamHealth(
            broker_connected=self._broker_connected,
            consumer_lag=self._consumer_lag,
            events_per_second=round(eps, 2),
            broker_type=self._broker_type,
        )

    async def process_event(self, event: dict) -> dict:
        """Process a single transaction event end-to-end."""
        user_id = event.get("user_id", "")
        feature_degraded = False

        try:
            feature_vector = await self._feature_store.get(user_id)
            if feature_vector.cold_start:
                feature_degraded = True
        except Exception:
            feature_degraded = True
            feature_vector = None

        try:
            xgb_score = self._predictor.predict_single(
                user_id, feature_vector.features if feature_vector else {}
            )
            if not isinstance(xgb_score, float):
                xgb_score = 0.5
        except Exception:
            xgb_score = 0.5

        ensemble_score = self._ensemble_scorer.combine(xgb_score=xgb_score)

        record = {
            "user_id": user_id,
            "risk_score": ensemble_score,
            "feature_degraded": feature_degraded,
            "model_version": "1.0",
        }

        try:
            await self._audit_logger.log_prediction(record)
        except Exception as exc:
            logger.warning("Audit log failed: %s", exc)

        if ensemble_score > self._risk_threshold:
            try:
                asyncio.create_task(self._alert_router.dispatch_async(record))
            except Exception as exc:
                logger.warning("Alert dispatch failed: %s", exc)

        self._events_processed += 1
        return record
