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

    # Dense embedding backend.
    embedding_url: str = "http://10.6.14.127:9003"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_api_key: str | None = None
    embedding_dimensions: int = 4096
    embedding_timeout_seconds: float = 60.0
    embedding_max_retries: int = 2
    embedding_retry_backoff_seconds: float = 1.5

    # Sparse embedding (BM25, computed in-process)
    sparse_embedding_model: str = "bm25"
    sparse_embedding_k1: float = 1.5
    sparse_embedding_b: float = 0.75
    sparse_embedding_avg_len: float = 256.0

    # Search / rerank defaults (overridable per-request via search_options).
    default_collection: str = "EISO_COLLECTION"
    default_vector_weight: float = 0.7
    default_keyword_weight: float = 0.3
    default_rrf_k: int = 5
    default_candidate_multiplier: int = 5
    default_fusion_mode: str = "client"
    # Dynamic routing for document-code / số-hiệu queries (router_regex feature).
    default_auto_route: bool = True
    default_identifier_bonus: float = 0.5
    default_score_threshold: float =0.2
    
    # Rerank defaults (overridable per-request via rerank option).
    default_rerank_strategy: str = "hybrid"
    default_rerank_mean_m: int = 3
    default_rerank_rrf_k: int = 60
    default_rerank_top_chunks: int = 3

    default_citation: bool = True
    citation_idf_tau: float = 2.5
    citation_top_keywords: int = 6
    citation_min_cohesion: float = 0.6
    citation_window_chars: int = 400
    citation_companion_ratio: float = 0.7
    citation_max_chunks: int = 5
    citation_semantic_gate: float = 0.5
    citation_max_phrase_tokens: int = 6
    citation_nested_ratio: float = 0.9

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
