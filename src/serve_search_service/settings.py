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
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_api_key: str | None = None
    embedding_dimensions: int = 2560
    embedding_timeout_seconds: float = 60.0
    embedding_max_retries: int = 2
    embedding_retry_backoff_seconds: float = 1.5
    # Qwen3-Embedding applies an instruction on the QUERY side only (documents
    # are embedded raw at ingest time). Without it, query vectors are off and
    # dense retrieval degrades badly for natural-language queries. Empty string
    # disables it (query embedded raw). Overridable via env
    # SERVE_SEARCH_DEFAULT_QUERY_INSTRUCTION and per-request.
    # Per Qwen3 guidance, instructions are written in English even for
    # non-English corpora (training instructions were mostly English), and
    # tailored to the task/domain. Overridable via env + per-request.
    default_query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

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
    default_fusion_mode: str = "batch_client_rrf"
    max_candidate_limit: int = 500  # Hard limit on candidate retrieval
    # Dynamic routing for document-code / số-hiệu queries (router_regex feature).
    default_auto_route: bool = True
    default_identifier_bonus: float = 0.5
    # Extra bonus added (on top of identifier_bonus) when the query carries a
    # version (e.g. "v4.0") AND the document's docVersion matches it. Prioritizes
    # the exact-version document within the same docMaVanBan family without
    # turning the version into part of the identifier.
    default_version_bonus: float = 0.4
    # Master switch for the version-matching flow . When false, the query version
    # (e.g. "v4.0") is still extracted but no version bonus is applied.
    default_version_match: bool = True
    # Với query MÃ/SỐ HIỆU: sắp lại doc đã match theo (khớp 100% > một phần),
    # (isShow='Y' chính > 'N' phụ), (mã tự nhiên PL01<PL02<PL10). Overridable.
    default_identifier_primary_sort: bool = True
    # Title branch: search chunk 0 (title/metadata) in parallel and return the
    # documents whose title covers the query in a separate title_result field.
    default_title_search: bool = True
    default_title_top_k: int = 20
    default_title_rrf_k: int = 10
    default_title_min_coverage: float = 0.5
    default_score_threshold: float = 0.2
    # Regex patterns for identifier shape detection (re.IGNORECASE applied automatically).
    # Override via env to extend/narrow what counts as a doc-code or sohieu.
    # Default values mirror the hardcoded patterns in router_regex.py.
    # SERVE_SEARCH_DEFAULT_CODE_SHAPE_PATTERN
    default_code_shape_pattern: str = (
        r"^[a-z]{1,4}\d{0,3}[a-z]?(?:/[a-z]{1,4}\d{0,3}[a-z]?){1,5}$"
    )
    # SERVE_SEARCH_DEFAULT_SOHIEU_SHAPE_PATTERN
    default_sohieu_shape_pattern: str = (
        r"^\d{1,5}/\d{4}(?:/[a-z0-9]{1,12}){0,6}$"
    )

    # --- Identifier detection: natural-language / version / family patterns ---
    # (re.IGNORECASE applied automatically). Mirror the hardcoded regex in
    # router_regex.py. Override via env SERVE_SEARCH_DEFAULT_* without touching
    # code; the NL/version ones are also overridable per-request (search_options).
    #
    # Query dài hơn ngưỡng này (số TỪ) không được coi là identifier (câu tự nhiên).
    default_max_whole_words: int = 8
    # Đuôi câu hỏi tiếng Việt bỏ khi tách identifier ("... là mã của văn bản nào").
    default_trailing_nl_pattern: str = (
        r"\s*(?:"
        r"la\s*ma\s*cua\s*van\s*ban\s*(?:nao|gi)?"
        r"|la\s*(?:gi|van\s*ban\s*(?:nao|gi)?|tai\s*lieu\s*(?:nao|gi)?)?"
        r"|noi\s*ve\s*gi"
        r"|la\s*cua\s*ai"
        r"|the\s*nao"
        r"|o\s*dau"
        r")\s*$"
    )
    # Đuôi "... là mã của văn bản" dùng để nhận biết câu hỏi tra cứu.
    default_lookup_tail_pattern: str = (
        r"\s+la\s*(?:"
        r"ma\s*cua\s*van\s*ban"
        r"|van\s*ban"
        r"|gi"
        r"|tai\s*lieu"
        r")"
        r"(?:\s*(?:nao|gi))?\s*$"
    )
    # Hậu tố version tường minh: v1, v2, v1.0, v1.1 ...
    default_version_explicit_pattern: str = (
        r"(?:^|\s+)(v\d+(?:[._/-]\d+){0,4})\.?\s*$"
    )
    # Hậu tố version theo từ khoá: "version 4", "ver 4.0", "phien ban 4", "ban 4".
    default_version_keyword_pattern: str = (
        r"\s+(?:version|ver|phien\s*ban|ban)\s*[:\-]?\s*"
        r"(\d+(?:[._/-]\d+){0,4})\.?\s*$"
    )
    # Hậu tố version dạng số trần: " 2.0", " 3.1" (bắt buộc khoảng trắng trước).
    default_version_bare_pattern: str = (
        r"\s+(\d+(?:[._/-]\d+)+)\.?\s*$"
    )
    # Tiền tố họ "PLxx" ở đầu mã (pl01, pl36). Dùng khi so khớp family.
    default_pl_prefix_pattern: str = r"pl\d{1,3}"
    # Ứng viên tiền tố họ: 1-4 ký tự chữ/số (độ dài kiểm tra riêng ở caller).
    default_family_prefix_pattern: str = r"[a-z0-9]+"

    # Rerank defaults (overridable per-request via rerank option).
    default_rerank_strategy: str = "hybrid"
    default_rerank_mean_m: int = 3
    default_rerank_rrf_k: int = 60
    default_rerank_top_chunks: int = 3

    # Model reranker (Qwen3-Reranker via vLLM /v1/rerank). Reorders fused chunk
    # candidates before document grouping; on/off overridable per-request via
    # search_options.model_rerank.
    default_model_rerank: bool = True
    reranker_url: str = "http://10.6.14.127:9002"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_api_key: str | None = None
    reranker_timeout_seconds: float = 30.0
    reranker_max_retries: int = 2
    reranker_retry_backoff_seconds: float = 1.0
    reranker_batch_size: int = 32
    # Wrap query/document in the official Qwen3-Reranker prompt. Set false if
    # the serving side already applies the template.
    reranker_qwen3_template: bool = True
    default_model_rerank_top_n: int = 50
    default_rerank_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

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
    citation_secondary_tau: float = 1.0

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
