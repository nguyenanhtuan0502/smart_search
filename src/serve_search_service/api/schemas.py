"""Client-facing DTOs for search API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ─────────────────────────────────────────────
# Rerank Options
# ─────────────────────────────────────────────


class RerankOptions(BaseModel):
    """Document-level reranking configuration.

    When provided, the API aggregates chunk results into document-level
    results and returns them sorted by document relevance score.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["maxp", "top_m_mean", "hit_count", "rrf", "hybrid"] = Field(
        default="hybrid",
        description="Rerank strategy: maxp | top_m_mean | hit_count | rrf | hybrid",
    )
    weights: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "Hybrid strategy weights: (maxp, top_m_mean, hit_count, rrf). "
            "Must sum to 1.0. Only used when strategy='hybrid'."
        ),
    )
    mean_m: int = Field(default=3, ge=1, description="Top M chunks for top_m_mean strategy")
    rrf_k: int = Field(default=60, ge=1, description="RRF k constant for rrf/hybrid strategy")
    document_id_field: str = Field(
        default="document_id",
        description="Payload field name that identifies the parent document",
    )
    top_chunks: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top chunks to include per document in reranked result",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "RerankOptions":
        """Validate weights when strategy is hybrid."""
        if self.weights is not None:
            if any(w < 0 for w in self.weights):
                raise ValueError("All weights must be non-negative.")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {sum(self.weights):.6f}."
                )
        elif self.strategy == "hybrid":
            # Use default weights — no validation needed
            pass
        return self


# ─────────────────────────────────────────────
# Search Request
# ─────────────────────────────────────────────


class HybridSearchRequest(BaseModel):
    """Hybrid search — nhận query text, service tự embed rồi search.

    Luồng: query → Embedding API → dense + sparse vectors → Qdrant → RRF fusion → result.
    Khi có `rerank`: result chuyển sang document-level format.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Query text (câu hỏi / từ khóa tìm kiếm)")
    collection: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(
        default=None,
        description="Minimum score threshold (applies to dense search only)",
    )
    filter: dict[str, Any] | None = None
    vector_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Dense weight in RRF"
    )
    keyword_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Sparse weight in RRF"
    )
    rrf_k: int = Field(default=60, ge=1, le=200)
    candidate_multiplier: int = Field(default=5, ge=2, le=20)
    with_payload: bool = True

    @model_validator(mode="after")
    def validate_weights_not_both_zero(self) -> "HybridSearchRequest":
        """At least one of vector_weight or keyword_weight must be > 0."""
        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("At least one of vector_weight or keyword_weight must be greater than 0.")
        return self
    rerank: RerankOptions | None = Field(
        default=None,
        description=(
            "Document-level reranking. When provided, result switches to "
            "document-level format sorted by document relevance score."
        ),
    )


# ─────────────────────────────────────────────
# Search Response — Chunk-level (no rerank)
# ─────────────────────────────────────────────


class ChunkResultItem(BaseModel):
    """A single chunk result from search (RRF fusion or single-mode)."""

    id: Any
    payload: dict[str, Any] | None = None
    score: float = Field(description="RRF score (hybrid) or similarity/keyword score (single-mode)")
    score_type: Literal["rrf", "similarity", "keyword"] = Field(
        default="rrf",
        description="Type of score: 'rrf' for fused, 'similarity' for dense, 'keyword' for sparse",
    )
    vector_rank: int | None = None
    keyword_rank: int | None = None


# ─────────────────────────────────────────────
# Search Response — Document-level (with rerank)
# ─────────────────────────────────────────────


class DocumentChunkItem(BaseModel):
    """A chunk belonging to a reranked document."""

    chunk_id: str
    payload: dict[str, Any] | None = None
    score: float = Field(description="RRF score or similarity/keyword score of this chunk")
    score_type: Literal["rrf", "similarity", "keyword"] = "rrf"
    vector_rank: int | None = None
    keyword_rank: int | None = None


class DocumentResultItem(BaseModel):
    """A document-level result after reranking (when rerank IS requested)."""

    document_id: str
    score: float = Field(description="Document relevance score from rerank strategy")
    chunks: list[DocumentChunkItem] = Field(
        description="Top chunks for this document, sorted by score desc"
    )


# ─────────────────────────────────────────────
# Unified Response
# ─────────────────────────────────────────────


class HybridSearchResponse(BaseModel):
    """Hybrid search response.

    mode values:
    - "hybrid": both dense + sparse succeeded, chunk-level RRF result
    - "vector_only": dense-only search (sparse not provided), chunk-level result
    - "keyword_only": sparse-only search (dense not provided), chunk-level result
    - "vector_only_fallback": sparse search failed, fallback to dense results
    - "keyword_only_fallback": dense search failed, fallback to sparse results
    - "hybrid_reranked": rerank applied, document-level result
    """

    result: list[ChunkResultItem] | list[DocumentResultItem] = Field(default_factory=list)
    vector_count: int = 0
    keyword_count: int = 0
    fused_count: int = 0
    mode: Literal[
        "hybrid",
        "vector_only",
        "keyword_only",
        "vector_only_fallback",
        "keyword_only_fallback",
        "hybrid_reranked",
    ] = "hybrid"
