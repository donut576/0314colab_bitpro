"""SHAPExplainer — per-user and global SHAP explanations."""

from __future__ import annotations


class SHAPExplainer:
    """Placeholder — implemented in Task 5."""

    def __init__(self, model_loader) -> None:
        self._model_loader = model_loader
        self.shap_available: bool = False

    def explain_user(self, user_id: str, features, top_n: int = 10) -> list:
        raise NotImplementedError("SHAPExplainer.explain_user not yet implemented")

    def get_global_summary_png(self, sample_df) -> bytes:
        raise NotImplementedError("SHAPExplainer.get_global_summary_png not yet implemented")
