"""Public HTTP routes for the search service."""

import time
from typing import Any

from fastapi import APIRouter, Request, status

from ..application.embedding import EmbeddingService
from ..application.rerank_service import RerankService
from ..application.router_regex import detect, normalize_identifier
from ..application.search import HybridSearchService
from ..domain.embedding import EmbedderUnavailableError, EmbeddingResult
from ..logging import TraceIdAdapter, get_logger
from .errors import ApiError
from .schemas import (
    FilterCondition,
    HybridSearchRequest,
    HybridSearchResponse,
    SearchOptions,
)

_logger: TraceIdAdapter = get_logger(__name__)

router = APIRouter()

# Singleton rerank service (stateless, safe to share)
_rerank_service = RerankService()


def _extract_vectors(
    result: EmbeddingResult,
) -> tuple[list[float], dict[str, list] | None]:
    """Pull the dense vector and (optional) sparse vector out of a single-query
    embedding result, shaped the way HybridSearchService expects them."""
    item = result.items[0]
    dense = list(item.vector)
    sparse: dict[str, list] | None = None
    if item.sparse is not None and item.sparse.indices:
        sparse = {
            "indices": list(item.sparse.indices),
            "values": list(item.sparse.values),
        }
    return dense, sparse


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


def _documents_matching_identifier(
    chunk_results: list[dict[str, Any]],
    query_identifier: str,
    document_id_field: str = "document_id",
) -> set[str]:
    """Return document ids whose docMaVanBan or docSoHieu (normalized) equals the
    query identifier. Used to award the identifier bonus after rerank.

    Purely payload-based (docMaVanBan/docSoHieu are present on every chunk), so
    it needs no extra Qdrant call.
    """
    if not query_identifier:
        return set()

    matched: set[str] = set()
    for item in chunk_results:
        payload = item.get("payload") or {}
        doc_id = payload.get(document_id_field)
        if doc_id is None:
            continue
        ma = normalize_identifier(payload.get("docMaVanBan"))
        sohieu = normalize_identifier(payload.get("docSoHieu"))
        if query_identifier in (ma, sohieu):
            matched.add(str(doc_id))
    return matched


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
    embedding_service: EmbeddingService = request.app.state.embedding_service
    total_start = time.perf_counter()

    opts = payload.search_options or SearchOptions()

    # Detect before embedding so identifier queries can use their canonical,
    # separator-preserving form for sparse retrieval.
    vector_weight = float(opts.vector_weight)
    keyword_weight = float(opts.keyword_weight)
    identifier = ""
    auto_route = opts.auto_route_enabled()
    detection = detect(payload.query) if auto_route else None
    embedding_query = payload.query.strip()
    is_identifier_query = bool(detection and detection.is_doc_code and detection.collapsed)
    if is_identifier_query:
        identifier = normalize_identifier(detection.collapsed)
        embedding_query = detection.collapsed
        vector_weight, keyword_weight = 0.0, 1.0

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
        embed_result = await embedding_service.create_embeddings(
            texts=[embedding_query],
            input_type="query",
            include_dense=not is_identifier_query,
            model=payload.embedding_model,
            base_url=payload.embedding_url,
            api_key=payload.embedding_api_key,
        )
    except EmbedderUnavailableError as e:
        _logger.error("Embedding failed for query='%s': %s", payload.query[:50], e)
        raise ApiError(
            status=502,
            code="embedding_failed",
            detail=str(e),
            retryable=True,
        ) from e
    dense_vector, sparse_vector = _extract_vectors(embed_result)
    if is_identifier_query:
        dense_vector = None
    embed_ms = (time.perf_counter() - embed_start) * 1000
    _logger.info("Embedding completed in %.0fms", embed_ms)

    # ── Step 2: Build Qdrant filter ──────────────────────────
    qdrant_filter = _build_qdrant_filter(payload.filter)

    # ── Step 2.5: Dynamic routing for document-code / số-hiệu queries ──
    # If auto_route is on and the query is an identifier lookup, force sparse-only
    # retrieval (identifiers are lexical, not semantic) and remember the
    # normalized identifier so we can bonus exact-match documents after rerank.
    if is_identifier_query:
        _logger.info(
            "Doc-code query detected (kind=%s, id=%s) → routing sparse-only",
            detection.kind, identifier,
        )

    # ── Step 3: Hybrid search (dense + sparse → RRF fusion) ──
    # `limit` is the number of DOCUMENTS the client wants back (cut in Step 5).
    # For fusion we keep a wider chunk pool (fuse_limit) so rerank and the
    # identifier bonus can see chunks beyond the first `limit`; the search
    # service caps this to the actual candidate pool.
    fuse_limit = int(payload.limit) * int(opts.candidate_multiplier)
    search_start = time.perf_counter()
    result = await search_service.hybrid_search(
        collection=payload.collection,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=int(payload.limit),
        payload_filter=qdrant_filter,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        rrf_k=int(opts.rrf_k),
        candidate_multiplier=int(opts.candidate_multiplier),
        with_payload=True,  # Always need payload internally for document grouping
        fuse_limit=fuse_limit,
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

    # Documents whose identifier exactly matches the query (empty when not a
    # code query or auto_route off). Their score gets the identifier bonus.
    hybrid_rerank_enabled = bool(
        payload.rerank and payload.rerank.strategy == "hybrid"
    )
    identifier_bonus = (
        float(opts.identifier_bonus)
        if auto_route and hybrid_rerank_enabled
        else 0.0
    )
    matched_docs = (
        _documents_matching_identifier(chunk_results, identifier, doc_id_field)
        if (identifier and identifier_bonus > 0)
        else set()
    )

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
        except ValueError as e:
            _logger.error("Rerank failed: %s", e)
            raise ApiError(
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="rerank_error",
                detail=f"Rerank failed: {e}",
                retryable=False,
            ) from e

        # Apply identifier bonus to the final (post-rerank) document score, then
        # re-sort. Rerank scores are in [0,1]; bonus >= 1.0 guarantees an
        # exact-identifier document ranks first, while ties among matched docs
        # keep their relative rerank order.
        if matched_docs:
            for doc in reranked:
                if doc["document_id"] in matched_docs:
                    doc["score"] = (doc.get("score", 0.0) or 0.0) + identifier_bonus
            reranked.sort(key=lambda d: d.get("score", 0.0), reverse=True)

        # Optional score threshold: drop every document below the cutoff. Rerank
        # scores are min-max normalized to [0,1]. Identifier-matched documents
        # already carry the bonus, so they normally stay above the cutoff; if one
        # still falls below it is dropped too (the threshold is absolute).
        n_before = len(reranked)
        if hybrid_rerank_enabled and opts.score_threshold is not None:
            threshold = float(opts.score_threshold)
            reranked = [
                doc for doc in reranked
                if (doc.get("score", 0.0) or 0.0) >= threshold
            ]

        document_results = [
            {"document_id": doc["document_id"], "rank_document": rank}
            for rank, doc in enumerate(reranked, 1)
        ]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        _logger.info(
            "Rerank completed in %.0fms | strategy=%s, documents=%d (filtered %d→%d), "
            "id_bonus_docs=%d",
            rerank_ms, payload.rerank.strategy, len(document_results),
            n_before, len(reranked), len(matched_docs),
        )
    else:
        # Without rerank: group by document_id, rank by max chunk score.
        # score_threshold is not applied here: raw fusion scores are not
        # normalized to [0,1], so a 0-1 cutoff would be meaningless.
        if hybrid_rerank_enabled and opts.score_threshold is not None:
            _logger.warning(
                "score_threshold=%s ignored: only applies when rerank is enabled "
                "(raw fusion scores are not normalized to [0,1]).",
                opts.score_threshold,
            )
        document_results = _group_chunks_to_documents(
            chunk_results,
            document_id_field=doc_id_field,
        )
        # Promote identifier-matched documents to the front (no score to add to
        # here, so we move them ahead, preserving their relative order).
        if matched_docs:
            matched = [d for d in document_results if d["document_id"] in matched_docs]
            rest = [d for d in document_results if d["document_id"] not in matched_docs]
            document_results = [
                {"document_id": d["document_id"], "rank_document": rank}
                for rank, d in enumerate(matched + rest, 1)
            ]

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