"""Client-facing DTOs for search API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────


class FilterCondition(BaseModel):
    """A single filter condition for Qdrant payload filtering.

    Examples:
        {"key": "group_ids", "value": ["hr", "aiit"]}
        {"key": "doc_status", "value": ["active"]}
        {"key": "document_id", "value": "doc-001"}
    """

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
        default="0.7", description="Dense vector weight in RRF fusion (0.0 to 1.0)"
    )
    keyword_weight: str = Field(
        default="0.3", description="Sparse (BM25) weight in RRF fusion (0.0 to 1.0)"
    )
    rrf_k: str = Field(
        default="60", description="RRF constant k (1 to 200)"
    )
    candidate_multiplier: str = Field(
        default="5",
        description="Multiply limit by this to get candidate pool size (2 to 20)",
    )
    fusion_mode: Literal["client", "server_rrf", "batch_client_rrf"] = Field(
        default="client",
        description=(
            "Fusion strategy for hybrid search:\n"
            "- 'client' (default): dense + sparse in 2 parallel Qdrant calls, then "
            "weighted RRF in-process (supports vector_weight/keyword_weight/rrf_k).\n"
            "- 'server_rrf': single Qdrant Query API call with server-side RRF fusion "
            "(weights ignored, requires Qdrant >= 1.10).\n"
            "- 'batch_client_rrf': single Qdrant batch request retrieves dense + sparse "
            "candidates, then weighted RRF in-process (keeps vector_weight/keyword_weight/"
            "rrf_k, 1 round-trip, requires Qdrant >= 1.10)."
        ),
    )

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

DEFAULT_COLLECTION = "EISO_COLLECTION"


class HybridSearchRequest(BaseModel):
    """Hybrid search request.

    Example:
        {
            "query": "recruitment process for staff",
            "collection": "eiso_collection",
            "limit": 20,
            "search_options": {...},
            "filter": [{"key": "group_ids", "value": ["hr"]}],
            "rerank": {"strategy": "hybrid", "top_chunks": 3}
        }
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(min_length=1, description="Query text (question / search keywords)")
    collection: str = Field(
        default=DEFAULT_COLLECTION,
        min_length=1,
        description="Qdrant collection name (default: EISO_COLLECTION)",
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


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────


class DocumentResultItem(BaseModel):
    """A document-level result."""

    document_id: str
    rank_document: int = Field(description="Document rank (1-based)")


class HybridSearchResponse(BaseModel):
    """Search response.

    {
        "status": 200,
        "result": [
            {"document_id": "doc-001", "rank_document": 1},
            {"document_id": "doc-002", "rank_document": 2}
        ]
    }
    """

    status: int = 200
    result: list[DocumentResultItem] = Field(default_factory=list)
