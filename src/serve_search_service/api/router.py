"""Public HTTP routes for the search service."""

import asyncio
import time
from collections import Counter
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request, status

from ..adapters.qdrant import QdrantClient
from ..application.citation import (
    CitationConfig,
    build_citations,
    idf_from_probe,
    query_token_indices,
)
from ..application.embedding import EmbeddingService
from ..application.rerank_service import RerankService
from ..application.router_regex import detect, normalize_identifier
from ..application.search import HybridSearchService
from ..domain.embedding import EmbedderUnavailableError, EmbeddingResult
from ..logging import TraceIdAdapter, get_logger
from .errors import ApiError
from .schemas import (
    CitationOptions,
    FilterCondition,
    HybridSearchRequest,
    HybridSearchResponse,
    SearchOptions,
)

_logger: TraceIdAdapter = get_logger(__name__)

router = APIRouter()

# Singleton rerank service (stateless, safe to share)
_rerank_service = RerankService()

_background_probes: set[asyncio.Task[dict[int, float]]] = set()


async def _probe_idf(
    qdrant_client: QdrantClient, collection: str, indices: list[int]
) -> dict[int, float]:
    if not indices:
        return {}
    try:
        return await qdrant_client.probe_sparse_idf(collection, indices)
    except Exception:
        _logger.exception("IDF probe raised, falling back to flat weights.")
        return {}


def _extract_vectors(
    result: EmbeddingResult,
) -> tuple[list[float] | None, dict[str, list] | None]:
    """Pull the dense vector and (optional) sparse vector out of a single-query
    embedding result, shaped the way HybridSearchService expects them.
    
    Dense vector may be None if include_dense=False (identifier query).
    """
    item = result.items[0]
    dense = list(item.vector) if item.vector is not None else None
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
    must_not_conditions: list[FilterCondition] | None = None,
) -> dict[str, Any] | None:
    """Convert FilterCondition lists to Qdrant filter format.

    Args:
        filter_conditions: Conditions to include (AND logic in must clause)
        must_not_conditions: Conditions to exclude (must_not clause, user-provided only)

    Input:  [{"key": "group_ids", "value": ["hr", "aiit"]}, ...]
    Output: {"must": [...], "must_not": [...]}
    
    Filter behavior:
    - match_type='value': Exact Qdrant pre-filter via must_not clause
    - match_type='text': Application-level post-filter using literal substring matching
      after Qdrant search (see _filter_excluded_patterns)
    
    Text post-filter is intentional so patterns like "v/v" can be matched literally
    without depending on Qdrant tokenizer configuration.
    
    None → no filter
    [] → no filter
    """
    if not filter_conditions:
        filter_conditions = []

    must_clauses: list[dict[str, Any]] = []

    for condition in filter_conditions:
        key = condition.key
        value = condition.value

        if isinstance(value, list):
            must_clauses.append({"key": key, "match": {"any": value}})
        else:
            must_clauses.append({"key": key, "match": {"value": value}})

    # Build must_not clauses from user-provided conditions
    # In Qdrant: must_not = [A, B] means NOT A AND NOT B (exclude if any match)
    must_not_clauses: list[dict[str, Any]] = []
    
    # Add user-provided must_not conditions (only if provided)
    if must_not_conditions:
        for condition in must_not_conditions:
            key = condition.key
            value = condition.value
            match_type = getattr(condition, "match_type", "value")
            
            # match_type='text' is handled by application-level post-filtering
            # after Qdrant search. Only send 'value' (exact) to Qdrant.
            if match_type == "text":
                continue
            
            if isinstance(value, list):
                must_not_clauses.append({"key": key, "match": {"any": value}})
            else:
                must_not_clauses.append({"key": key, "match": {"value": value}})

    # Return filter structure
    filter_dict: dict[str, Any] = {}
    if must_clauses:
        filter_dict["must"] = must_clauses
    if must_not_clauses:
        filter_dict["must_not"] = must_not_clauses

    return filter_dict if filter_dict else None


def _filter_excluded_patterns(
    chunk_results: list[dict[str, Any]],
    must_not_conditions: list[FilterCondition] | None,
) -> list[dict[str, Any]]:
    """Post-filter chunks using literal substring matching for text patterns.

    match_type='text' is handled at application level (not Qdrant pre-filter)
    because Qdrant text matching is tokenizer-based, not raw substring/contains.
    
    This ensures patterns like "v/v" are matched literally in chunk text.
    """
    if not must_not_conditions or not chunk_results:
        return chunk_results
    
    # Extract only text-type conditions
    text_conditions = [
        cond
        for cond in must_not_conditions
        if getattr(cond, "match_type", "value") == "text"
    ]
    
    if not text_conditions:
        return chunk_results
    
    filtered: list[dict[str, Any]] = []
    
    for chunk in chunk_results:
        payload = chunk.get("payload") or {}
        excluded = False
        
        for cond in text_conditions:
            field_value = payload.get(cond.key)
            
            if field_value is None:
                continue
            
            patterns = (
                cond.value
                if isinstance(cond.value, list)
                else [cond.value]
            )
            
            if isinstance(field_value, list):
                text_content = " ".join(str(v) for v in field_value).casefold()
            else:
                text_content = str(field_value).casefold()
            
            if any(
                str(pattern).casefold() in text_content
                for pattern in patterns
            ):
                excluded = True
                break
        
        if not excluded:
            filtered.append(chunk)
    
    _logger.info(
        "Post-filter text patterns: before=%d, after=%d",
        len(chunk_results),
        len(filtered),
    )
    
    return filtered


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

    idf_task: asyncio.Task[dict[int, float]] | None = None
    if opts.citation_enabled():
        query_tokens, query_indices = query_token_indices(payload.query)
        idf_task = asyncio.create_task(
            _probe_idf(request.app.state.qdrant_client, payload.collection, query_indices)
        )
        _background_probes.add(idf_task)
        idf_task.add_done_callback(_background_probes.discard)
    else:
        _logger.info("Citation disabled for this request (search_options.citation=false)")

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
    # Merge filters from 2 sources with AND logic:
    # 1. Top-level filter (user/external)
    # 2. Nested filter_conditions in search_options (if provided)
    filter_conditions: list[FilterCondition] = []
    
    # 1. Add top-level filter
    if payload.filter:
        filter_conditions.extend(payload.filter)
    
    # 2. Add nested filter_conditions from search_options (only if provided by user)
    if opts.filter_conditions:
        filter_conditions.extend(opts.filter_conditions)
    
    qdrant_filter = _build_qdrant_filter(filter_conditions, opts.must_not_conditions) if filter_conditions else _build_qdrant_filter([], opts.must_not_conditions)

    # ── Step 2.5: Dynamic routing for document-code / số-hiệu queries ──
    # If auto_route is on and the query is an identifier lookup, force sparse-only
    # retrieval (identifiers are lexical, not semantic) and remember the
    # normalized identifier so we can bonus exact-match documents after rerank.
    if is_identifier_query:
        _logger.info(
            "Doc-code query detected (kind=%s, id=%s) → routing sparse-only",
            detection.kind, identifier,
        )


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
        fusion_mode=opts.fusion_mode,
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

    # Log chunk score range for debugging
    chunk_results = result.get("items", [])
    if chunk_results:
        scores = [item.get("score", 0) for item in chunk_results if isinstance(item.get("score"), (int, float))]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            _logger.info(
                "Chunk scores: max=%.4f, min=%.4f, count=%d",
                max_score, min_score, len(scores)
            )

    # ── Step 4: Build document-level results ─────────────────
    chunk_results = result.get("result", [])
    
    # Apply literal substring exclusions after Qdrant search.
    chunk_results = _filter_excluded_patterns(
        chunk_results,
        opts.must_not_conditions,
    )
    
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
        # re-sort. Rerank scores are in [0,1]; bonus > 1.0 guarantees an
        # exact-identifier document ranks first. If bonus = 1.0, matched docs
        # may tie with highest-scored non-matched docs (tied docs keep relative order).
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

    citation_ms = 0.0
    if idf_task is not None and not document_results:
        _logger.info("Citation skipped: no documents to cite")
    if idf_task is not None and document_results:
        citation_start = time.perf_counter()
        try:
            cite = payload.citation_options or CitationOptions()
            probed = await idf_task
            idf_wait_ms = (time.perf_counter() - citation_start) * 1000
            config = CitationConfig(
                tau=float(cite.idf_tau),
                top_m=int(cite.top_keywords),
                min_cohesion=float(cite.min_cohesion),
                window_chars=int(cite.window_chars),
                companion_ratio=float(cite.companion_ratio),
                max_chunks=int(cite.max_chunks),
                semantic_gate=float(cite.semantic_gate),
                max_phrase_tokens=int(cite.max_phrase_tokens),
                nested_ratio=float(cite.nested_ratio),
            )
            citations = build_citations(
                document_results,
                chunk_results,
                payload.query,
                idf_from_probe(query_tokens, query_indices, probed),
                document_id_field=doc_id_field,
                config=config,
            )
            for doc in document_results:
                block = citations.get(doc["document_id"])
                if block:
                    doc["citation_tier"] = block["tier"]
                    doc["citations"] = block["citations"]
            citation_ms = (time.perf_counter() - citation_start) * 1000
            tiers = Counter(block["tier"] for block in citations.values())
            snippets = [c for block in citations.values() for c in block["citations"]]
            _logger.info(
                "Citations built in %.0fms (idf wait %.0fms) | documents=%d, tiers=%s, "
                "snippets=%d, highlights=%d, probed_tokens=%d/%d | config=%s",
                citation_ms,
                idf_wait_ms,
                len(document_results),
                dict(tiers),
                len(snippets),
                sum(len(c["highlights"]) for c in snippets),
                len(probed),
                len(query_indices),
                asdict(config),
            )
        except Exception:
            citation_ms = (time.perf_counter() - citation_start) * 1000
            _logger.exception("Citation building failed, returning results without it.")

    # ── Summary log ───────────────────────────────────────────
    total_ms = (time.perf_counter() - total_start) * 1000
    _logger.info(
        "Request completed in %.0fms | embed=%.0fms, search=%.0fms, citation=%.0fms, "
        "documents=%d",
        total_ms, embed_ms, search_ms, citation_ms, len(document_results),
    )

    return {
        "status": 200,
        "result": document_results,
    }