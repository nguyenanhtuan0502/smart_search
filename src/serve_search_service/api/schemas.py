"""Client-facing DTOs for search API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    rrf_k: int = Field(default=60, ge=0, description="RRF k constant for rrf/hybrid strategy")
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
    """A single chunk result from RRF fusion (when rerank is NOT requested)."""

    id: Any
    payload: dict[str, Any] | None = None
    rrf_score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


# ─────────────────────────────────────────────
# Search Response — Document-level (with rerank)
# ─────────────────────────────────────────────


class DocumentChunkItem(BaseModel):
    """A chunk belonging to a reranked document."""

    chunk_id: str
    payload: dict[str, Any] | None = None
    rrf_score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


class DocumentResultItem(BaseModel):
    """A document-level result after reranking (when rerank IS requested)."""

    document_id: str
    score: float = Field(description="Document relevance score from rerank strategy")
    chunks: list[DocumentChunkItem] = Field(
        description="Top chunks for this document, sorted by rrf_score desc"
    )


# ─────────────────────────────────────────────
# Unified Response
# ─────────────────────────────────────────────


class HybridSearchResponse(BaseModel):
    """Hybrid search response.

    mode values:
    - "hybrid": both dense + sparse, chunk-level result
    - "vector_only": dense-only search, chunk-level result
    - "keyword_only": sparse-only search, chunk-level result
    - "hybrid_reranked": rerank applied, document-level result
    """

    result: list[ChunkResultItem] | list[DocumentResultItem] = Field(default_factory=list)
    vector_count: int = 0
    keyword_count: int = 0
    fused_count: int = 0
    mode: Literal["hybrid", "vector_only", "keyword_only", "hybrid_reranked"] = "hybrid"
