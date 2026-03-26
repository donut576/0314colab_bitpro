"""Application configuration loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Existing ──────────────────────────────────────────────────────────
    model_s3_uri: str = "s3://aml-models/model_registry/latest"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/aml"
    default_mode: Literal["full", "no_leak", "safe"] = "safe"
    max_batch_size: int = 1000
    shap_top_n: int = 10
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.2
    audit_retention_days: int = 90
    aws_region: str = "ap-northeast-1"

    # ── Streaming (REQ-A1) ────────────────────────────────────────────────
    stream_broker_type: Literal["kafka", "kinesis"] = "kafka"
    kafka_bootstrap_servers: str = "localhost:9092"
    kinesis_stream_name: str = "aml-transactions"

    # ── Feature Store (REQ-A9) ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Graph Engine (REQ-A2) ─────────────────────────────────────────────
    graph_hidden_dim: int = 64
    graph_high_fraction_threshold: float = 0.3

    # ── Sequence Model (REQ-A4) ───────────────────────────────────────────
    seq_lookback_days: int = 90
    seq_min_transactions: int = 5
    seq_model_arch: Literal["lstm", "transformer"] = "lstm"
    seq_retrain_schedule: str = "weekly"

    # ── Ensemble Weights (REQ-A2.6) ───────────────────────────────────────
    ensemble_weight_xgb: float = 0.5
    ensemble_weight_graph: float = 0.3
    ensemble_weight_seq: float = 0.2

    # ── Adaptive Thresholds (REQ-A5) ──────────────────────────────────────
    threshold_max_queue: int = 500
    threshold_min_queue: int = 50
    threshold_high_floor: float = 0.5
    threshold_high_ceiling: float = 0.95

    # ── AI Copilot (REQ-A7) ───────────────────────────────────────────────
    copilot_llm_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ── Alert Router (REQ-A8) ─────────────────────────────────────────────
    line_notify_token: str = ""
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_webhook_url: str = ""
    alert_rate_limit_per_hour: int = 60
    alert_cooldown_seconds: int = 3600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
