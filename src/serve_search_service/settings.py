"""Runtime settings for the data processing service."""

from enum import StrEnum
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


# Find .env file relative to this module
_env_file = Path(__file__).parent.parent.parent / ".env"


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SERVE_SEARCH_",
        env_file=str(_env_file) if _env_file.exists() else None,
        extra="ignore"
    )

    deployment_environment: str = "development"
    service_version: str = "0.1.0"
    log_directory: Path = Path("logs")
    runtime_mode: RuntimeMode = RuntimeMode.DEVELOPMENT
    tenant_resolver: str = "development-header"
    dev_tenant_id: str = "local-development-tenant"

    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Embedding service (serve-data-processing-service)
    embedding_url: str = "http://localhost:8000"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"

    @field_validator("qdrant_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def reject_development_identity_in_production(self) -> "ServiceSettings":
        if (
            self.runtime_mode is RuntimeMode.PRODUCTION
            and self.tenant_resolver == "development-header"
        ):
            raise ValueError("Production requires a verified identity adapter.")
        return self
