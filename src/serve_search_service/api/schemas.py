"""Client-facing DTOs for search API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..application.search import MAX_CANDIDATE_LIMIT
from ..settings import ServiceSettings

# Load once at import so field defaults come from env (SERVE_SEARCH_*), while
# remaining overridable per-request. String-typed fields keep string defaults.
_settings = ServiceSettings()


# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────


class FilterCondition(BaseModel):


    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, description="Payload field name to filter on")
    value: str | list[str] = Field(
        min_length=1,
        description="Value or list of values to match (OR within a condition)",
    )


class UdfField(BaseModel):
    """A user-defined field in the MAS payload format."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, description="User-defined field name")
    value: Any = Field(description="User-defined field value")


# ─────────────────────────────────────────────
# Search Options
# ─────────────────────────────────────────────


class SearchOptions(BaseModel):
    """Search tuning parameters."""

    model_config = ConfigDict(extra="forbid")

    vector_weight: str = Field(
        default_factory=lambda: str(_settings.default_vector_weight),
        description="Dense vector weight in RRF fusion (0.0 to 1.0)",
    )
    keyword_weight: str = Field(
        default_factory=lambda: str(_settings.default_keyword_weight),
        description="Sparse (BM25) weight in RRF fusion (0.0 to 1.0)",
    )
    rrf_k: str = Field(
        default_factory=lambda: str(_settings.default_rrf_k),
        description="RRF constant k (1 to 200)",
    )
    candidate_multiplier: str = Field(
        default_factory=lambda: str(_settings.default_candidate_multiplier),
        description="Multiply limit by this to get candidate pool size (2 to 20)",
    )
    fusion_mode: Literal["client", "server_rrf", "batch_client_rrf"] = Field(
        default_factory=lambda: _settings.default_fusion_mode,  # type: ignore[arg-type]
        description="Fusion strategy for hybrid search",
    )
    auto_route: str = Field(
        default_factory=lambda: str(_settings.default_auto_route).lower(),
        description="Dynamic routing for document-code queries ('true'/'false')",
    )
    identifier_bonus: str = Field(
        default_factory=lambda: str(_settings.default_identifier_bonus),
        description=(
            "Score added to a document (after rerank) when its docMaVanBan/docSoHieu "
            "exactly matches the detected query identifier ."
        ),
    )
    score_threshold: str | None = Field(
        default_factory=lambda: (
            None if _settings.default_score_threshold is None
            else str(_settings.default_score_threshold)
        ),
        description=(
            "Drop documents whose final rerank score is below this threshold "
            "(0.0 to 1.0). Applies only when rerank is used. None disables filtering."
        ),
    )

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold_string(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"score_threshold must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {value}")
        return v

    @field_validator("vector_weight", "keyword_weight")
    @classmethod
    def validate_weight_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError:
            raise ValueError(f"Weight must be a valid number string, got '{v}'")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {value}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"rrf_k must be a valid integer string, got '{v}'")
        if not (1 <= value <= 200):
            raise ValueError(f"rrf_k must be between 1 and 200, got {value}")
        return v

    @field_validator("candidate_multiplier")
    @classmethod
    def validate_candidate_multiplier_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"candidate_multiplier must be a valid integer string, got '{v}'")
        if not (2 <= value <= 20):
            raise ValueError(f"candidate_multiplier must be between 2 and 20, got {value}")
        return v

    @field_validator("auto_route")
    @classmethod
    def validate_auto_route_string(cls, v: str) -> str:
        if v.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                f"auto_route must be a boolean string (true/false), got '{v}'"
            )
        return v

    def auto_route_enabled(self) -> bool:
        """Parse auto_route string to bool for internal use."""
        return self.auto_route.strip().lower() in {"true", "1", "yes"}

    @field_validator("identifier_bonus")
    @classmethod
    def validate_identifier_bonus_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"identifier_bonus must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 5.0):
            raise ValueError(f"identifier_bonus must be between 0.0 and 5.0, got {value}")
        return v

    @model_validator(mode="after")
    def validate_weights_not_both_zero(self) -> "SearchOptions":
        vector_w = float(self.vector_weight)
        keyword_w = float(self.keyword_weight)
        if vector_w == 0 and keyword_w == 0:
            raise ValueError(
                "At least one of vector_weight or keyword_weight must be > 0."
            )
        return self


# ─────────────────────────────────────────────
# Rerank Options
# ─────────────────────────────────────────────


class RerankOptions(BaseModel):
    """Document-level reranking configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["maxp", "top_m_mean", "hit_count", "rrf", "hybrid"] = Field(
        default="hybrid",
        description="Rerank strategy: maxp | top_m_mean | hit_count | rrf | hybrid",
    )
    weights: str | None = Field(
        default=None,
        description=(
            "Hybrid strategy weights as string: '(w1, w2, w3, w4)' for (maxp, top_m_mean, hit_count, rrf). "
            "Must sum to 1.0. Only used when strategy='hybrid'."
        ),
    )
    mean_m: str = Field(
        default="3", description="Top M chunks for top_m_mean strategy"
    )
    rrf_k: str = Field(
        default="60", description="RRF k constant for rrf/hybrid strategy"
    )
    document_id_field: str = Field(
        default="document_id",
        description="Payload field name that identifies the parent document",
    )
    top_chunks: str = Field(
        default="3",
        description="Number of top chunks used internally for scoring",
    )

    @staticmethod
    def _parse_weights(weights: str) -> tuple[float, ...]:
        """Parse weights string to tuple.
        
        Accepts formats like:
        - "(0.25, 0.25, 0.25, 0.25)"
        - "0.25, 0.25, 0.25, 0.25"
        """
        weights_str = weights.strip()
        if weights_str.startswith("(") and weights_str.endswith(")"):
            weights_str = weights_str[1:-1]
        return tuple(float(w.strip()) for w in weights_str.split(","))

    @field_validator("mean_m")
    @classmethod
    def validate_mean_m_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"mean_m must be a valid integer string, got '{v}'")
        if value < 1:
            raise ValueError(f"mean_m must be >= 1, got {value}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"rrf_k must be a valid integer string, got '{v}'")
        if value <= 0:
            raise ValueError(f"rrf_k must be > 0, got {value}")
        return v

    @field_validator("top_chunks")
    @classmethod
    def validate_top_chunks_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"top_chunks must be a valid integer string, got '{v}'")
        if not (1 <= value <= 20):
            raise ValueError(f"top_chunks must be between 1 and 20, got {value}")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "RerankOptions":
        if self.strategy != "hybrid" and self.weights is not None:
            raise ValueError("weights is only supported when strategy='hybrid'.")
        if self.weights is not None:
            try:
                weight_values = self._parse_weights(self.weights)
            except ValueError:
                raise ValueError(f"weights must be comma-separated floats, got '{self.weights}'")
            
            if len(weight_values) != 4:
                raise ValueError(f"weights must have exactly 4 values, got {len(weight_values)}")
            
            if any(w < 0 for w in weight_values):
                raise ValueError("All weights must be non-negative.")
            if abs(sum(weight_values) - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {sum(weight_values):.6f}."
                )
        return self

    def get_weights_tuple(self) -> tuple[float, ...] | None:
        """Get weights as tuple for service consumption."""
        if not self.weights:
            return None
        return self._parse_weights(self.weights)


# ─────────────────────────────────────────────
# Search Request
# ─────────────────────────────────────────────

class HybridSearchRequest(BaseModel):

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(min_length=1, description="Query text (question / search keywords)")
    collection: str = Field(
        default_factory=lambda: _settings.default_collection,
        min_length=1,
        description="Qdrant collection name (default from SERVE_SEARCH_DEFAULT_COLLECTION)",
    )
    limit: str = Field(default="10", description="Max number of documents to return (1-100)")

    embedding_url: str | None = Field(
        default=None,
        min_length=1,
        description="Override the dense embedding gateway base URL (default: from env)",
    )
    embedding_model: str | None = Field(
        default=None,
        min_length=1,
        description="Override the dense embedding model (default: from env)",
    )
    embedding_api_key: str | None = Field(
        default=None,
        min_length=1,
        description="Override the API key sent to the embedding gateway (default: from env)",
    )

    search_options: SearchOptions | None = Field(
        default=None,
        description="Search tuning parameters (weights, thresholds, etc.)",
        alias="searchOptions",
    )
    filter: list[FilterCondition] | None = Field(
        default=None,
        description=(
            'Payload filter conditions (AND logic between conditions). '
            'Example: [{"key": "group_ids", "value": ["hr"]}]'
        ),
    )
    rerank: RerankOptions | None = Field(
        default=None,
        description="Document-level reranking options",
    )
    udf: list[UdfField] | None = Field(
        default=None,
        description="User-defined fields in MAS format: [{key, value}, ...]",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be blank.")
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"limit must be a valid integer string, got '{v}'")
        if not (1 <= value <= 100):
            raise ValueError(f"limit must be between 1 and 100, got {value}")
        return v

    @model_validator(mode="after")
    def validate_candidate_pool(self) -> "HybridSearchRequest":
        # The candidate pool actually retrieved is
        #   min(limit * candidate_multiplier, MAX_CANDIDATE_LIMIT).
        # Reject (instead of silently capping) when the requested pool exceeds
        # the hard cap, so the caller is not misled about how many candidates
        # are used. Both fields validate individually first, so int() is safe.
        opts = self.search_options or SearchOptions()
        pool = int(self.limit) * int(opts.candidate_multiplier)
        if pool > MAX_CANDIDATE_LIMIT:
            raise ValueError(
                f"limit * candidate_multiplier = {pool} exceeds the candidate pool "
                f"cap of {MAX_CANDIDATE_LIMIT}; reduce limit or candidate_multiplier."
            )
        return self


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────


class DocumentResultItem(BaseModel):
    """A document-level result."""

    document_id: str
    rank_document: int = Field(description="Document rank (1-based)")


class HybridSearchResponse(BaseModel):

    status: int = 200
    result: list[DocumentResultItem] = Field(default_factory=list)
