from __future__ import annotations
import logging
import os
from datetime import datetime
from app.models.sequence import AnomalousEvent, BehavioralProfile, SequenceScore

logger = logging.getLogger(__name__)


class SequenceScorer:
    MODEL_VERSION = "seq-1.0"

    def __init__(self) -> None:
        self._lookback_days = int(os.getenv("SEQ_LOOKBACK_DAYS", "90"))
        self._min_transactions = int(os.getenv("SEQ_MIN_TRANSACTIONS", "5"))
        self._model_arch = os.getenv("SEQ_MODEL_ARCH", "lstm")
        # user_id -> list of transaction dicts
        self._user_sequences: dict[str, list[dict]] = {}

    def add_transaction(self, user_id: str, tx: dict) -> None:
        """Append a transaction event to the user's sequence."""
        if user_id not in self._user_sequences:
            self._user_sequences[user_id] = []
        self._user_sequences[user_id].append(tx)

    def score(self, user_id: str) -> SequenceScore:
        """Score a user's behavioral sequence for anomalies."""
        txs = self._user_sequences.get(user_id, [])
        if len(txs) < self._min_transactions:
            return SequenceScore(
                user_id=user_id,
                sequence_anomaly_score=None,
                top_anomalous_events=[],
                model_version=self.MODEL_VERSION,
                insufficient_history=True,
            )

        # Stub: production would run LSTM/Transformer inference here
        # Score is based on recency-weighted variance of amounts as a proxy
        amounts = [float(t.get("amount", 0.0)) for t in txs[-self._lookback_days:]]
        if len(amounts) > 1:
            mean = sum(amounts) / len(amounts)
            variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
            max_amount = max(amounts) if amounts else 1.0
            anomaly_score = min(1.0, variance / (max_amount ** 2 + 1e-9))
        else:
            anomaly_score = 0.0

        # Build top-3 anomalous events (highest amounts as proxy)
        sorted_txs = sorted(txs, key=lambda t: float(t.get("amount", 0.0)), reverse=True)[:3]
        top_events = [
            AnomalousEvent(
                timestamp=datetime.fromisoformat(t["timestamp"]) if isinstance(t.get("timestamp"), str) else datetime.utcnow(),
                channel=t.get("channel", "unknown"),
                amount=float(t.get("amount", 0.0)),
                anomaly_score=anomaly_score,
            )
            for t in sorted_txs
        ]

        return SequenceScore(
            user_id=user_id,
            sequence_anomaly_score=round(anomaly_score, 6),
            top_anomalous_events=top_events,
            model_version=self.MODEL_VERSION,
            insufficient_history=False,
        )

    def get_profile(self, user_id: str) -> BehavioralProfile:
        txs = self._user_sequences.get(user_id, [])
        recent = sorted(txs, key=lambda t: t.get("timestamp", ""), reverse=True)[:10]
        recent_events = [
            AnomalousEvent(
                timestamp=datetime.fromisoformat(t["timestamp"]) if isinstance(t.get("timestamp"), str) else datetime.utcnow(),
                channel=t.get("channel", "unknown"),
                amount=float(t.get("amount", 0.0)),
                anomaly_score=0.0,
            )
            for t in recent
        ]
        return BehavioralProfile(
            user_id=user_id,
            transaction_count=len(txs),
            lookback_days=self._lookback_days,
            last_scored=datetime.utcnow() if txs else None,
            recent_events=recent_events,
        )

    def retrain(self) -> None:
        """Stub: production would retrain LSTM/Transformer on latest sequences."""
        logger.info("SequenceScorer retrain triggered (arch=%s)", self._model_arch)
