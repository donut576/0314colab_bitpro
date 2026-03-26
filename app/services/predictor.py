"""XGBPredictor — wraps the loaded XGBoost model for batch/single inference."""

from __future__ import annotations


class XGBPredictor:
    """Placeholder — implemented in Task 4."""

    def __init__(self, model_loader) -> None:
        self._model_loader = model_loader

    def predict_batch(self, features) -> list:
        raise NotImplementedError("XGBPredictor.predict_batch not yet implemented")

    def predict_single(self, user_id: str, features: dict) -> object:
        raise NotImplementedError("XGBPredictor.predict_single not yet implemented")
