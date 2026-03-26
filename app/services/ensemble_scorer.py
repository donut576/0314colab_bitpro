"""EnsembleScorer — weighted combination of model scores."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class EnsembleScorer:
    def __init__(self) -> None:
        self._w_xgb = float(os.getenv("ENSEMBLE_WEIGHT_XGB", "0.5"))
        self._w_graph = float(os.getenv("ENSEMBLE_WEIGHT_GRAPH", "0.3"))
        self._w_seq = float(os.getenv("ENSEMBLE_WEIGHT_SEQ", "0.2"))

    def combine(
        self,
        xgb_score: float | None = None,
        graph_score: float | None = None,
        seq_score: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Weighted average of available scores, clamped to [0, 1]."""
        if weights:
            w_xgb = weights.get("xgb", self._w_xgb)
            w_graph = weights.get("graph", self._w_graph)
            w_seq = weights.get("seq", self._w_seq)
        else:
            w_xgb, w_graph, w_seq = self._w_xgb, self._w_graph, self._w_seq

        total_weight = 0.0
        weighted_sum = 0.0

        if xgb_score is not None:
            weighted_sum += xgb_score * w_xgb
            total_weight += w_xgb
        else:
            logger.warning("xgb_score missing from ensemble")

        if graph_score is not None:
            weighted_sum += graph_score * w_graph
            total_weight += w_graph
        else:
            logger.warning("graph_score missing from ensemble")

        if seq_score is not None:
            weighted_sum += seq_score * w_seq
            total_weight += w_seq
        else:
            logger.warning("seq_score missing from ensemble")

        if total_weight == 0.0:
            return 0.0

        result = weighted_sum / total_weight
        return max(0.0, min(1.0, result))
