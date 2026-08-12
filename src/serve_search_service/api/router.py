"""Public HTTP routes for the search service."""

from typing import Any

from fastapi import APIRouter, Request, status

from ..application.search import HybridSearchService
from ..logging import TraceIdAdapter, get_logger
from .errors import ApiError
from .schemas import HybridSearchRequest, HybridSearchResponse

_logger: TraceIdAdapter = get_logger(__name__)

router = APIRouter()


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
    """Hybrid search (Dense + Sparse RRF).

    Caller handles embedding externally — sends dense_vector + sparse_vector.
    This service queries Qdrant and fuses results.
    """
    search_service: HybridSearchService = request.app.state.search_service

    _logger.info(
        "POST /v1/search/hybrid - collection=%s, limit=%d, has_sparse=%s",
        payload.collection,
        payload.limit,
        payload.sparse_vector is not None,
    )

    result = await search_service.hybrid_search(
        collection=payload.collection,
        dense_vector=payload.dense_vector,
        sparse_vector=payload.sparse_vector,
        limit=payload.limit,
        score_threshold=payload.score_threshold,
        payload_filter=payload.filter,
        vector_weight=payload.vector_weight,
        keyword_weight=payload.keyword_weight,
        rrf_k=payload.rrf_k,
        candidate_multiplier=payload.candidate_multiplier,
        with_payload=payload.with_payload,
    )

    if result.get("error"):
        raise ApiError(
            status=result.get("status", 503),
            code="search_failed",
            detail=result.get("detail", "Search failed"),
            retryable=True,
        )

    return result
