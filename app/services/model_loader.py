"""ModelLoader — loads XGBoost artifact and training stats from S3 at startup."""

from __future__ import annotations


class ModelLoader:
    """Placeholder — implemented in Task 3."""

    def __init__(self) -> None:
        self._model = None
        self._metadata: dict = {}
        self._training_stats: dict = {}
        self.loaded: bool = False

    def load_from_s3(self, s3_uri: str) -> None:  # noqa: ARG002
        raise NotImplementedError("ModelLoader.load_from_s3 not yet implemented")

    def get_model(self):
        return self._model

    def get_metadata(self) -> dict:
        return self._metadata

    def get_training_stats(self) -> dict:
        return self._training_stats
