"""Application configuration loaded from environment variables via pydantic-settings."""

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

    # Model artifact
    model_s3_uri: str = "s3://aml-models/model_registry/latest"

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/aml"

    # API behaviour
    default_mode: Literal["full", "no_leak", "safe"] = "safe"
    max_batch_size: int = 1000
    shap_top_n: int = 10

    # Drift thresholds (PSI)
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.2

    # Audit retention
    audit_retention_days: int = 90

    # AWS (optional — boto3 also reads env vars directly)
    aws_region: str = "ap-northeast-1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
