"""Client-facing DTOs for search API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────

class HybridSearchRequest(BaseModel):
    """Hybrid search — caller sends pre-embedded vectors."""

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(min_length=1)
    dense_vector: list[float] = Field(min_length=1, description="Dense vector")
    sparse_vector: dict[str, Any] | None = Field(
        default=None,
        description='Sparse vector: {"indices": [...], "values": [...]}. Null = dense-only.',
    )
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = None
    filter: dict[str, Any] | None = None
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=200)
    candidate_multiplier: int = Field(default=5, ge=2, le=20)
    with_payload: bool = True


class HybridResultItem(BaseModel):
    id: Any
    payload: dict[str, Any] | None = None
    rrf_score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


class HybridSearchResponse(BaseModel):
    result: list[HybridResultItem] = Field(default_factory=list)
    vector_count: int = 0
    keyword_count: int = 0
    fused_count: int = 0
    mode: str = "hybrid"
