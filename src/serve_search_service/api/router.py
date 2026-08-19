"""Public HTTP routes for the search service."""

import time
from typing import Any

from fastapi import APIRouter, Request, status

from ..adapters.call_embedding import EmbeddingClient, EmbeddingError
from ..application.rerank_service import RerankService
from ..application.search import HybridSearchService
from ..logging import TraceIdAdapter, get_logger
from .errors import ApiError
from .schemas import HybridSearchRequest, HybridSearchResponse

_logger: TraceIdAdapter = get_logger(__name__)

router = APIRouter()

# Singleton rerank service (stateless, safe to share)
_rerank_service = RerankService()


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@router.get("/health/live", tags=["platform"])
async def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready", tags=["platform"])
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/health/startup", tags=["platform"])
async def startup() -> dict[str, str]:
    return {"status": "started"}


@router.get("/v1", tags=["platform"])
async def api_root(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {
        "service": "serve-search-service",
        "version": settings.service_version,
        "api_version": "v1",
    }


# ─────────────────────────────────────────────
# Search API
# ─────────────────────────────────────────────

@router.post("/v1/search/hybrid", response_model=HybridSearchResponse, tags=["search"])
async def api_hybrid_search(payload: HybridSearchRequest, request: Request) -> dict[str, Any]:
    """Hybrid search — nhận query text, tự embed rồi search Qdrant.

    Luồng: query text → Embedding API → dense + sparse vectors → Qdrant → RRF fusion.
    Khi có `rerank`: result chuyển sang document-level format.
    """
    search_service: HybridSearchService = request.app.state.search_service
    embedding_client: EmbeddingClient = request.app.state.embedding_client
    total_start = time.perf_counter()

    _logger.info(
        "POST /v1/search/hybrid | collection=%s, query='%s', limit=%d, rerank=%s",
        payload.collection,
        payload.query[:80] + "..." if len(payload.query) > 80 else payload.query,
        payload.limit,
        payload.rerank.strategy if payload.rerank else None,
    )

    # ── Step 1: Embed query text ─────────────────────────────
    embed_start = time.perf_counter()
    try:
        embed_result = await embedding_client.embed_query(payload.query.strip())
    except EmbeddingError as e:
        _logger.error("Embedding failed for query='%s': %s", payload.query[:50], e)
        raise ApiError(
            status=502,
            code="embedding_failed",
            detail=str(e),
            retryable=True,
        ) from e
    embed_ms = (time.perf_counter() - embed_start) * 1000
    _logger.info("Embedding completed in %.0fms", embed_ms)

    # ── Step 2: Hybrid search (dense + sparse → RRF fusion) ──
    search_start = time.perf_counter()
    result = await search_service.hybrid_search(
        collection=payload.collection,
        dense_vector=embed_result["embedding"],
        sparse_vector=embed_result["sparse_embedding"],
        limit=payload.limit,
        score_threshold=payload.score_threshold,
        payload_filter=payload.filter,
        vector_weight=payload.vector_weight,
        keyword_weight=payload.keyword_weight,
        rrf_k=payload.rrf_k,
        candidate_multiplier=payload.candidate_multiplier,
        with_payload=payload.with_payload,
    )
    search_ms = (time.perf_counter() - search_start) * 1000

    if result.get("error"):
        _logger.error(
            "Search failed: status=%s, detail=%s",
            result.get("status"), result.get("detail"),
        )
        raise ApiError(
            status=result.get("status", 503),
            code="search_failed",
            detail=result.get("detail", "Search failed"),
            retryable=True,
        )

    _logger.info(
        "Search completed in %.0fms | mode=%s, vector_count=%d, keyword_count=%d, fused_count=%d",
        search_ms,
        result.get("mode"),
        result.get("vector_count", 0),
        result.get("keyword_count", 0),
        result.get("fused_count", 0),
    )

    # ── Step 3: Apply document-level reranking if requested ───
    if payload.rerank and result.get("result"):
        rerank_start = time.perf_counter()
        try:
            document_results = _rerank_service.apply(
                result["result"],
                strategy=payload.rerank.strategy,
                document_id_field=payload.rerank.document_id_field,
                top_chunks=payload.rerank.top_chunks,
                weights=payload.rerank.weights,
                mean_m=payload.rerank.mean_m,
                rrf_k=payload.rerank.rrf_k,
            )
            result["result"] = document_results
            result["mode"] = "hybrid_reranked"
            result["fused_count"] = len(document_results)
        except ValueError as e:
            _logger.error("Rerank failed: %s", e)
            raise ApiError(
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="rerank_error",
                detail=f"Rerank failed: {e}",
            ) from e
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        _logger.info(
            "Rerank completed in %.0fms | strategy=%s, documents=%d",
            rerank_ms, payload.rerank.strategy, len(document_results),
        )

    # ── Summary log ───────────────────────────────────────────
    total_ms = (time.perf_counter() - total_start) * 1000
    _logger.info(
        "Request completed in %.0fms | embed=%.0fms, search=%.0fms, mode=%s, results=%d",
        total_ms, embed_ms, search_ms,
        result.get("mode"), result.get("fused_count", 0),
    )

    return result
