"""Public HTTP routes for the search service."""

import time
from typing import Any

from fastapi import APIRouter, Request, status

from ..adapters.call_embedding import EmbeddingClient, EmbeddingError
from ..application.rerank_service import RerankService
from ..application.search import HybridSearchService
from ..logging import TraceIdAdapter, get_logger
from .errors import ApiError
from .schemas import FilterCondition, HybridSearchRequest, HybridSearchResponse

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
# Helpers
# ─────────────────────────────────────────────

def _build_qdrant_filter(
    filter_conditions: list[FilterCondition] | None,
) -> dict[str, Any] | None:
    """Convert FilterCondition list to Qdrant filter format.

    Input:  [{"key": "group_ids", "value": ["hr", "aiit"]}, ...]
    Output: {"must": [{"key": "group_ids", "match": {"any": ["hr", "aiit"]}}, ...]}
    
    None → no filter
    [] → no filter
    """
    if not filter_conditions:
        return None

    must_clauses: list[dict[str, Any]] = []

    for condition in filter_conditions:
        key = condition.key
        value = condition.value

        if isinstance(value, list):
            must_clauses.append({"key": key, "match": {"any": value}})
        else:
            must_clauses.append({"key": key, "match": {"value": value}})

    return {"must": must_clauses} if must_clauses else None


def _group_chunks_to_documents(
    chunk_results: list[dict[str, Any]],
    document_id_field: str = "document_id",
) -> list[dict[str, Any]]:
    """Group chunk-level results into document-level results.

    Score = max chunk score per document, used for sorting.
    Chunks missing document_id or with invalid score are skipped with a warning.
    """
    if not chunk_results:
        return []

    doc_scores: dict[str, float] = {}

    for item in chunk_results:
        payload = item.get("payload") or {}
        doc_id = payload.get(document_id_field)
        if doc_id is None:
            _logger.warning(
                "Chunk %s missing '%s' in payload, skipping.",
                item.get("id", "?"), document_id_field,
            )
            continue
        doc_id = str(doc_id)
        
        # Validate score is a number
        score = item.get("score")
        if not isinstance(score, (int, float)):
            _logger.warning(
                "Chunk %s has invalid score=%r, skipping.",
                item.get("id", "?"),
                score,
            )
            continue
        score = float(score)
        
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    # Sort by score desc
    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])

    return [
        {"document_id": doc_id, "rank_document": rank}
        for rank, (doc_id, _) in enumerate(sorted_docs, 1)
    ]


# ─────────────────────────────────────────────
# Search API
# ─────────────────────────────────────────────

@router.post("/v1/search/hybrid", response_model=HybridSearchResponse, tags=["search"])
async def api_hybrid_search(payload: HybridSearchRequest, request: Request) -> dict[str, Any]:
    """Hybrid search — nhận query text, tự embed rồi search Qdrant.

    Response:
        {"status": 200, "result": [{"document_id": "...", "rank_document": 1}, ...]}
    """
    search_service: HybridSearchService = request.app.state.search_service
    embedding_client: EmbeddingClient = request.app.state.embedding_client
    total_start = time.perf_counter()

    opts = payload.search_options

    _logger.info(
        "POST /v1/search/hybrid | collection=%s, query_len=%d, limit=%d, rerank=%s",
        payload.collection,
        len(payload.query),
        int(payload.limit),
        payload.rerank.strategy if payload.rerank else None,
    )

    # ── Step 1: Embed query text ─────────────────────────────
    embed_start = time.perf_counter()
    try:
        embed_result = await embedding_client.embed_query(payload.query)
    except EmbeddingError as e:
        _logger.error("Embedding failed: %s", e.detail)
        raise ApiError(
            status=e.status,
            code="embedding_failed",
            detail=e.detail,
            retryable=e.retryable,
        ) from e
    embed_ms = (time.perf_counter() - embed_start) * 1000
    _logger.info("Embedding completed in %.0fms", embed_ms)

    # ── Step 2: Build Qdrant filter ──────────────────────────
    qdrant_filter = _build_qdrant_filter(payload.filter)

    # ── Step 3: Hybrid search (dense + sparse → RRF fusion) ──
    search_start = time.perf_counter()
    result = await search_service.hybrid_search(
        collection=payload.collection,
        dense_vector=embed_result["embedding"],
        sparse_vector=embed_result["sparse_embedding"],
        limit=int(payload.limit),
        payload_filter=qdrant_filter,
        vector_weight=float(opts.vector_weight),
        keyword_weight=float(opts.keyword_weight),
        rrf_k=int(opts.rrf_k),
        candidate_multiplier=int(opts.candidate_multiplier),
        with_payload=True,  # Always need payload internally for document grouping
    )
    search_ms = (time.perf_counter() - search_start) * 1000

    if result.get("error"):
        _logger.error(
            "Search failed: status=%s, detail=%s",
            result.get("status"), result.get("detail"),
        )
        # Determine if retryable based on status code
        retryable = result.get("status") in (503, 504)  # Service unavailable or gateway timeout
        raise ApiError(
            status=result.get("status", 503),
            code="search_failed",
            detail=result.get("detail", "Search failed"),
            retryable=retryable,
        )

    _logger.info(
        "Search completed in %.0fms | mode=%s, vector_count=%d, keyword_count=%d, fused_count=%d",
        search_ms,
        result.get("mode"),
        result.get("vector_count", 0),
        result.get("keyword_count", 0),
        result.get("fused_count", 0),
    )

    # ── Step 4: Build document-level results ─────────────────
    chunk_results = result.get("result", [])
    doc_id_field = payload.rerank.document_id_field if payload.rerank else "document_id"

    if payload.rerank and chunk_results:
        # With rerank: use rerank service for better document scoring
        rerank_start = time.perf_counter()
        try:
            reranked = _rerank_service.apply(
                chunk_results,
                strategy=payload.rerank.strategy,
                document_id_field=payload.rerank.document_id_field,
                top_chunks=int(payload.rerank.top_chunks),
                weights=payload.rerank.get_weights_tuple(),
                mean_m=int(payload.rerank.mean_m),
                rrf_k=int(payload.rerank.rrf_k),
            )
            document_results = [
                {"document_id": doc["document_id"], "rank_document": rank}
                for rank, doc in enumerate(reranked, 1)
            ]
        except ValueError as e:
            _logger.error("Rerank failed: %s", e)
            raise ApiError(
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="rerank_error",
                detail=f"Rerank failed: {e}",
                retryable=False,
            ) from e
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        _logger.info(
            "Rerank completed in %.0fms | strategy=%s, documents=%d",
            rerank_ms, payload.rerank.strategy, len(document_results),
        )
    else:
        # Without rerank: group by document_id, rank by max chunk score
        document_results = _group_chunks_to_documents(
            chunk_results,
            document_id_field=doc_id_field,
        )

    # ── Step 5: Apply limit (số document trả về tối đa) ─────
    document_results = document_results[:int(payload.limit)]

    # ── Summary log ───────────────────────────────────────────
    total_ms = (time.perf_counter() - total_start) * 1000
    _logger.info(
        "Request completed in %.0fms | embed=%.0fms, search=%.0fms, documents=%d",
        total_ms, embed_ms, search_ms, len(document_results),
    )

    return {
        "status": 200,
        "result": document_results,
    }
