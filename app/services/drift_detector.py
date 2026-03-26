"""DriftDetector — PSI-based data drift detection."""

from __future__ import annotations


class DriftDetector:
    """Placeholder — implemented in Task 6."""

    def __init__(self, model_loader) -> None:
        self._model_loader = model_loader

    def compute_psi(self, feature: str, current_values) -> float:
        raise NotImplementedError("DriftDetector.compute_psi not yet implemented")

    def compute_batch_drift(self, batch_df) -> object:
        raise NotImplementedError("DriftDetector.compute_batch_drift not yet implemented")
