"""Public HTTP routes for the search service."""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
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
from ..application.router_regex import (
    detect,
    identifier_confusable_typo_variants_path,
    identifier_extra_retrieval_queries,
    identifier_match_kind,
    identifier_typo_variants_path,
    normalize_identifier,
    normalize_identifier_path,
    version_match,
)
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


# ============================================================
# Dedicated debug file
# One TXT file is created for each service process.
# ============================================================

_DEBUG_LOG_DIR = Path.cwd() / "logs"
_DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

_DEBUG_LOG_FILE = (
    _DEBUG_LOG_DIR
    / f"search_debug_{datetime.now():%Y%m%d_%H%M%S}.txt"
)

_debug_file_logger = logging.getLogger(
    f"{__name__}.top_results"
)

_debug_file_logger.setLevel(logging.INFO)
_debug_file_logger.propagate = False

if not _debug_file_logger.handlers:
    _handler = logging.FileHandler(
        _DEBUG_LOG_FILE,
        mode="a",
        encoding="utf-8",
    )
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s")
    )
    _debug_file_logger.addHandler(_handler)


router = APIRouter()

_logger.info(
    "Top-result debug log file: %s",
    _DEBUG_LOG_FILE,
)


# Singleton rerank service (stateless, safe to share)
_rerank_service = RerankService()

_background_probes: set[
    asyncio.Task[dict[int, float]]
] = set()


# ============================================================
# Citation / IDF
# ============================================================

async def _probe_idf(
    qdrant_client: QdrantClient,
    collection: str,
    indices: list[int],
) -> dict[int, float]:
    if not indices:
        return {}

    try:
        return await qdrant_client.probe_sparse_idf(
            collection,
            indices,
        )
    except Exception:
        _logger.exception(
            "IDF probe raised, falling back to flat weights."
        )
        return {}


# ============================================================
# Embedding helpers
# ============================================================

def _extract_vectors(
    result: EmbeddingResult,
) -> tuple[list[float] | None, dict[str, list] | None]:
    """Pull dense and sparse vectors from a single-query embedding result."""

    item = result.items[0]

    dense = (
        list(item.vector)
        if item.vector is not None
        else None
    )

    sparse: dict[str, list] | None = None

    if item.sparse is not None and item.sparse.indices:
        sparse = {
            "indices": list(item.sparse.indices),
            "values": list(item.sparse.values),
        }

    return dense, sparse


# ============================================================
# Health
# ============================================================

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
async def api_root(
    request: Request,
) -> dict[str, str]:
    settings = request.app.state.settings

    return {
        "service": "serve-search-service",
        "version": settings.service_version,
        "api_version": "v1",
    }


# ============================================================
# Helpers
# ============================================================

def _build_qdrant_filter(
    filter_conditions: list[FilterCondition] | None,
    must_not_conditions: list[FilterCondition] | None = None,
) -> dict[str, Any] | None:
    """Convert FilterCondition lists to Qdrant filter format.

    Args:
        filter_conditions:
            Conditions to include (AND logic in must clause).

        must_not_conditions:
            Conditions to exclude.

    Text conditions are handled by application-level post-filtering.
    """

    if not filter_conditions:
        filter_conditions = []

    must_clauses: list[dict[str, Any]] = []

    for condition in filter_conditions:
        key = condition.key
        value = condition.value

        if isinstance(value, list):
            must_clauses.append(
                {
                    "key": key,
                    "match": {
                        "any": value,
                    },
                }
            )
        else:
            must_clauses.append(
                {
                    "key": key,
                    "match": {
                        "value": value,
                    },
                }
            )

    # Build must_not clauses
    must_not_clauses: list[dict[str, Any]] = []

    if must_not_conditions:
        for condition in must_not_conditions:
            key = condition.key
            value = condition.value

            match_type = getattr(
                condition,
                "match_type",
                "value",
            )

            # Text matching is handled after Qdrant search.
            if match_type == "text":
                continue

            if isinstance(value, list):
                must_not_clauses.append(
                    {
                        "key": key,
                        "match": {
                            "any": value,
                        },
                    }
                )
            else:
                must_not_clauses.append(
                    {
                        "key": key,
                        "match": {
                            "value": value,
                        },
                    }
                )

    filter_dict: dict[str, Any] = {}

    if must_clauses:
        filter_dict["must"] = must_clauses

    if must_not_clauses:
        filter_dict["must_not"] = must_not_clauses

    return (
        filter_dict
        if filter_dict
        else None
    )


def _filter_excluded_patterns(
    chunk_results: list[dict[str, Any]],
    must_not_conditions: list[FilterCondition] | None,
) -> list[dict[str, Any]]:
    """Post-filter chunks using literal substring matching."""

    if not must_not_conditions or not chunk_results:
        return chunk_results

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
                text_content = " ".join(
                    str(v)
                    for v in field_value
                ).casefold()
            else:
                text_content = str(
                    field_value
                ).casefold()

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


# ============================================================
# Identifier exact match - CHUNK LEVEL
# ============================================================

def _find_identifier_matches(
    chunk_results: list[dict[str, Any]],
    query_identifier: str,
    document_id_field: str = "document_id",
    query_path: str | None = None,
) -> list[dict[str, Any]]:

    matches: list[dict[str, Any]] = []

    if not query_identifier and not query_path:
        return matches

    query_compact = normalize_identifier(query_identifier)
    query_path_norm = normalize_identifier_path(
        query_path or query_identifier
    )
    typo_variants = identifier_typo_variants_path(
        query_path_norm
    ) | identifier_confusable_typo_variants_path(
        query_path_norm
    )
    typo_compact = {
        normalize_identifier(value)
        for value in typo_variants
    }

    _debug_file_logger.info(
        "========== IDENTIFIER CHUNK MATCH CHECK START =========="
    )
    _debug_file_logger.info(
        "query_identifier=%r",
        query_compact,
    )
    _debug_file_logger.info(
        "query_path=%r",
        query_path_norm,
    )
    _debug_file_logger.info(
        "docMaVanBan_typo_variants=%s",
        sorted(typo_compact),
    )
    _debug_file_logger.info(
        "document_id_field=%s",
        document_id_field,
    )
    _debug_file_logger.info(
        "candidate_chunk_count=%d",
        len(chunk_results),
    )

    for rank, item in enumerate(
        chunk_results,
        start=1,
    ):
        payload = item.get("payload") or {}
        doc_id = payload.get(document_id_field)

        if doc_id is None:
            _debug_file_logger.info(
                "CHUNK CHECK SKIP | "
                "rank=%d | chunk_id=%s | "
                "reason=missing_document_id",
                rank,
                item.get("id", "?"),
            )
            continue

        ma_raw = payload.get("docMaVanBan")
        sohieu_raw = payload.get("docSoHieu")

        ma = normalize_identifier(ma_raw)
        sohieu = normalize_identifier(sohieu_raw)
        chunk_index = payload.get("chunk_index")

        chunk_text = (
            payload.get("chunk_text")
            or payload.get("chunkText")
            or payload.get("text")
            or payload.get("content")
            or ""
        )

        _debug_file_logger.info(
            "CHUNK CHECK | "
            "rank=%d | chunk_id=%s | document_id=%s | "
            "chunk_index=%s | docMaVanBan=%r -> %r | "
            "docSoHieu=%r -> %r",
            rank,
            item.get("id", "?"),
            doc_id,
            chunk_index,
            ma_raw,
            ma,
            sohieu_raw,
            sohieu,
        )

        # Decide the match kind using the pure regex helper. Orchestration
        # (payload extraction, logging, bonus) stays here in router.py.
        matched_field, raw_value_field, match_kind = identifier_match_kind(
            query_compact,
            query_path_norm,
            ma_raw,
            sohieu_raw,
            typo_compact,
        )

        if raw_value_field == "ma":
            raw_value: Any = ma_raw
            normalized_value: str | None = ma
        elif raw_value_field == "sohieu":
            raw_value = sohieu_raw
            normalized_value = sohieu
        else:
            raw_value = None
            normalized_value = None

        if matched_field:
            match_info = {
                "rank": rank,
                "chunk_id": item.get("id", ""),
                "document_id": str(doc_id),
                "chunk_index": chunk_index,
                "matched_field": matched_field,
                "raw_value": raw_value,
                "normalized_value": normalized_value,
                "score": item.get("score", 0.0),
                "chunk_text": chunk_text,
                "payload": payload,
            }
            matches.append(match_info)

            _debug_file_logger.info(
                "%s | rank=%d | chunk_id=%s | document_id=%s | "
                "chunk_index=%s | field=%s | value=%r | normalized=%r | "
                "score=%s | chunk_text=%r",
                match_kind,
                rank,
                item.get("id", "?"),
                doc_id,
                chunk_index,
                matched_field,
                raw_value,
                normalized_value,
                item.get("score"),
                chunk_text,
            )
        else:
            _debug_file_logger.info(
                "IDENTIFIER NO MATCH | "
                "rank=%d | chunk_id=%s | document_id=%s | "
                "chunk_index=%s",
                rank,
                item.get("id", "?"),
                doc_id,
                chunk_index,
            )

    _debug_file_logger.info(
        "========== IDENTIFIER CHUNK MATCH CHECK END =========="
    )
    _debug_file_logger.info(
        "matched_chunk_count=%d",
        len(matches),
    )
    _debug_file_logger.info(
        "matched_doc_count=%d",
        len({str(match["document_id"]) for match in matches}),
    )

    for match in matches:
        _debug_file_logger.info(
            "MATCHED CHUNK SUMMARY | "
            "document_id=%s | chunk_id=%s | chunk_index=%s | "
            "field=%s | value=%r",
            match["document_id"],
            match["chunk_id"],
            match["chunk_index"],
            match["matched_field"],
            match["raw_value"],
        )

    return matches


def _merge_chunk_results(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    max_count: int,
) -> list[dict[str, Any]]:
    """Union two result sets; keep the highest score for duplicate chunks."""
    merged: dict[str, dict[str, Any]] = {}

    for item in [*primary, *fallback]:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue

        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue

        previous = merged.get(item_id)
        if previous is None or score > float(previous.get("score", 0.0)):
            merged[item_id] = item

    return sorted(
        merged.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:max_count]

def _documents_matching_identifier(
    chunk_results: list[dict[str, Any]],
    query_identifier: str,
    document_id_field: str = "document_id",
) -> set[str]:
    """
    Backward-compatible wrapper.

    Still returns a set of document IDs so existing
    rerank / promotion logic remains unchanged.
    """

    matches = _find_identifier_matches(
        chunk_results,
        query_identifier,
        document_id_field,
    )

    return {
        str(match["document_id"])
        for match in matches
    }


# ============================================================
# Group chunks -> documents
# ============================================================

def _group_chunks_to_documents(
    chunk_results: list[dict[str, Any]],
    document_id_field: str = "document_id",
) -> list[dict[str, Any]]:
    """Group chunk-level results into document-level results."""

    if not chunk_results:
        return []

    doc_scores: dict[str, float] = {}

    for item in chunk_results:
        payload = item.get("payload") or {}
        doc_id = payload.get(document_id_field)

        if doc_id is None:
            _logger.warning(
                "Chunk %s missing '%s' in payload, skipping.",
                item.get("id", "?"),
                document_id_field,
            )
            continue

        doc_id = str(doc_id)

        score = item.get("score")

        if not isinstance(
            score,
            (int, float),
        ):
            _logger.warning(
                "Chunk %s has invalid score=%r, skipping.",
                item.get("id", "?"),
                score,
            )
            continue

        score = float(score)

        if (
            doc_id not in doc_scores
            or score > doc_scores[doc_id]
        ):
            doc_scores[doc_id] = score

    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: -x[1],
    )

    return [
        {
            "document_id": doc_id,
            "rank_document": rank,
        }
        for rank, (doc_id, _) in enumerate(
            sorted_docs,
            1,
        )
    ]


# ============================================================
# Search API
# ============================================================

@router.post(
    "/v1/search/hybrid",
    response_model=HybridSearchResponse,
    tags=["search"],
)
async def api_hybrid_search(
    payload: HybridSearchRequest,
    request: Request,
) -> dict[str, Any]:
    """Hybrid search — nhận query text, tự embed rồi search Qdrant."""

    search_service: HybridSearchService = (
        request.app.state.search_service
    )

    embedding_service: EmbeddingService = (
        request.app.state.embedding_service
    )

    total_start = time.perf_counter()

    opts = payload.search_options or SearchOptions()

    # ========================================================
    # Citation / IDF
    # ========================================================

    idf_task: asyncio.Task[
        dict[int, float]
    ] | None = None

    if opts.citation_enabled():
        query_tokens, query_indices = query_token_indices(
            payload.query
        )

        idf_task = asyncio.create_task(
            _probe_idf(
                request.app.state.qdrant_client,
                payload.collection,
                query_indices,
            )
        )

        _background_probes.add(idf_task)

        idf_task.add_done_callback(
            _background_probes.discard
        )

    else:
        _logger.info(
            "Citation disabled for this request "
            "(search_options.citation=false)"
        )

    # ========================================================
    # Step 0: Router Regex / Identifier Detection
    # ========================================================

    vector_weight = float(
        opts.vector_weight
    )

    keyword_weight = float(
        opts.keyword_weight
    )

    identifier = ""

    auto_route = opts.auto_route_enabled()

    # Detect before embedding so identifier queries can use
    # canonical separator-preserving form.
    detection = (
        detect(
            payload.query,
            code_shape_pattern=opts.code_shape_pattern,
            sohieu_shape_pattern=opts.sohieu_shape_pattern,
        )
        if auto_route
        else None
    )

    # ========================================================
    # DEBUG: ROUTER REGEX
    # ========================================================

    _debug_file_logger.info(
        "============================================================"
    )

    _debug_file_logger.info(
        "QUERY=%s",
        payload.query,
    )

    _debug_file_logger.info(
        "========== ROUTER REGEX =========="
    )

    _debug_file_logger.info(
        "auto_route=%s",
        auto_route,
    )

    if detection is None:
        _debug_file_logger.info(
            "detection=None"
        )

    else:
        _debug_file_logger.info(
            "is_doc_code=%s",
            detection.is_doc_code,
        )

        _debug_file_logger.info(
            "kind=%s",
            detection.kind,
        )

        _debug_file_logger.info(
            "candidate=%r",
            detection.candidate,
        )

        _debug_file_logger.info(
            "collapsed=%r",
            detection.collapsed,
        )

        _debug_file_logger.info(
            "version=%r",
            detection.version,
        )

        _debug_file_logger.info(
            "reason=%s",
            detection.reason,
        )

    embedding_query = payload.query.strip()

    is_identifier_query = bool(
        detection
        and detection.is_doc_code
        and detection.collapsed
    )

    _debug_file_logger.info(
        "is_identifier_query=%s",
        is_identifier_query,
    )

    # ========================================================
    # Identifier routing
    # ========================================================

    if is_identifier_query:
        identifier = normalize_identifier(
            detection.collapsed
        )

        # PA2: keep the original user text for embedding/sparse retrieval.
        # Identifier routing already forces sparse-only, but we do not rewrite
        # the user's query here.
        embedding_query = payload.query.strip()

        vector_weight = 0.0
        keyword_weight = 1.0

        _debug_file_logger.info(
            "========== IDENTIFIER ROUTING =========="
        )

        _debug_file_logger.info(
            "identifier=%r",
            identifier,
        )

        _debug_file_logger.info(
            "embedding_query=%r",
            embedding_query,
        )

        _debug_file_logger.info(
            "vector_weight=%s",
            vector_weight,
        )

        _debug_file_logger.info(
            "keyword_weight=%s",
            keyword_weight,
        )

    else:
        _debug_file_logger.info(
            "========== NORMAL HYBRID ROUTING =========="
        )

        _debug_file_logger.info(
            "vector_weight=%s",
            vector_weight,
        )

        _debug_file_logger.info(
            "keyword_weight=%s",
            keyword_weight,
        )

    # ========================================================
    # Request log
    # ========================================================

    _logger.info(
        "POST /v1/search/hybrid | "
        "collection=%s, query_len=%d, limit=%d, rerank=%s",
        payload.collection,
        len(payload.query),
        int(payload.limit),
        payload.rerank.strategy
        if payload.rerank
        else None,
    )

    # ========================================================
    # Step 1: Embed query text
    # ========================================================

    embed_start = time.perf_counter()

    # Qwen3 instruction for the dense query side. Documents are embedded raw at
    # ingest, so applying the instruction only to the query matches how the
    # model was trained. Skipped for identifier queries (sparse-only).
    query_instruction = (
        (opts.query_instruction or "").strip() or None
        if not is_identifier_query
        else None
    )

    _debug_file_logger.info(
        "========== DENSE QUERY EMBEDDING =========="
    )
    _debug_file_logger.info(
        "embedding_query=%r",
        embedding_query,
    )
    _debug_file_logger.info(
        "query_instruction=%r",
        query_instruction,
    )
    _debug_file_logger.info(
        "formatted_query=%r",
        f"Instruct: {query_instruction}\nQuery:{embedding_query}"
        if query_instruction
        else embedding_query,
    )

    try:
        embed_result = (
            await embedding_service.create_embeddings(
                texts=[embedding_query],
                input_type="query",
                include_dense=not is_identifier_query,
                model=payload.embedding_model,
                base_url=payload.embedding_url,
                api_key=payload.embedding_api_key,
                instruction=query_instruction,
            )
        )

    except EmbedderUnavailableError as e:
        _logger.error(
            "Embedding failed for query='%s' (instruction='%s'): %s",
            payload.query[:50],
            (query_instruction or "")[:50],
            e,
        )

        raise ApiError(
            status=502,
            code="embedding_failed",
            detail=str(e),
            retryable=True,
        ) from e

    dense_vector, sparse_vector = _extract_vectors(
        embed_result
    )

    if is_identifier_query:
        dense_vector = None

    embed_ms = (
        time.perf_counter()
        - embed_start
    ) * 1000

    _logger.info(
        "Embedding completed in %.0fms",
        embed_ms,
    )

    # ========================================================
    # Step 2: Build Qdrant filter
    # ========================================================

    filter_conditions: list[FilterCondition] = []

    # Top-level filter
    if payload.filter:
        filter_conditions.extend(
            payload.filter
        )

    # Nested filter_conditions
    if opts.filter_conditions:
        filter_conditions.extend(
            opts.filter_conditions
        )

    qdrant_filter = (
        _build_qdrant_filter(
            filter_conditions,
            opts.must_not_conditions,
        )
        if filter_conditions
        else _build_qdrant_filter(
            [],
            opts.must_not_conditions,
        )
    )

    # ========================================================
    # Step 2.5: Dynamic routing
    # ========================================================

    if is_identifier_query:
        _logger.info(
            "Doc-code query detected "
            "(kind=%s, id=%s) → routing sparse-only",
            detection.kind,
            identifier,
        )

        _debug_file_logger.info(
            "========== SPARSE-ONLY ROUTING CONFIRMED =========="
        )

        _debug_file_logger.info(
            "kind=%s",
            detection.kind,
        )

        _debug_file_logger.info(
            "identifier=%r",
            identifier,
        )

        _debug_file_logger.info(
            "dense_enabled=False"
        )

        _debug_file_logger.info(
            "sparse_enabled=True"
        )

    # ========================================================
    # Step 3: Search Qdrant
    # ========================================================

    fuse_limit = (
        int(payload.limit)
        * int(opts.candidate_multiplier)
    )

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
        candidate_multiplier=int(
            opts.candidate_multiplier
        ),
        fusion_mode=opts.fusion_mode,
        with_payload=True,
        fuse_limit=fuse_limit,
        max_candidate_limit=(
            request.app.state.settings.max_candidate_limit
        ),
    )

    search_ms = (
        time.perf_counter()
        - search_start
    ) * 1000

    if result.get("error"):
        _logger.error(
            "Search failed: status=%s, detail=%s",
            result.get("status"),
            result.get("detail"),
        )

        retryable = result.get("status") in (
            503,
            504,
        )

        raise ApiError(
            status=result.get("status", 503),
            code="search_failed",
            detail=result.get(
                "detail",
                "Search failed",
            ),
            retryable=retryable,
        )

    # ========================================================
    # Identifier retrieval fallbacks (sparse-only extra queries)
    #
    # These rewrites stay on the detected identifier. They do not
    # change natural-language hybrid search.
    # ========================================================
    if is_identifier_query:
        extra_queries = identifier_extra_retrieval_queries(
            detection,
            payload.query,
        )

        if extra_queries:
            _debug_file_logger.info(
                "========== IDENTIFIER RETRIEVAL FALLBACK =========="
            )
            _debug_file_logger.info(
                "original_identifier=%r | extra_queries=%s",
                identifier,
                extra_queries,
            )

            merged_chunks = result.get("result", [])

            try:
                for extra_query in extra_queries:
                    fallback_embed = await embedding_service.create_embeddings(
                        texts=[extra_query],
                        input_type="query",
                        include_dense=False,
                        model=payload.embedding_model,
                        base_url=payload.embedding_url,
                        api_key=payload.embedding_api_key,
                    )

                    _, fallback_sparse = _extract_vectors(fallback_embed)
                    if not fallback_sparse:
                        continue

                    fallback_result = await search_service.hybrid_search(
                        collection=payload.collection,
                        dense_vector=None,
                        sparse_vector=fallback_sparse,
                        limit=int(payload.limit),
                        payload_filter=qdrant_filter,
                        vector_weight=0.0,
                        keyword_weight=1.0,
                        rrf_k=int(opts.rrf_k),
                        candidate_multiplier=int(opts.candidate_multiplier),
                        fusion_mode=opts.fusion_mode,
                        with_payload=True,
                        fuse_limit=fuse_limit,
                        max_candidate_limit=(
                            request.app.state.settings.max_candidate_limit
                        ),
                    )

                    if fallback_result.get("error"):
                        _debug_file_logger.warning(
                            "Identifier fallback search failed query=%r: %s",
                            extra_query,
                            fallback_result.get("detail"),
                        )
                        continue

                    fallback_chunks = fallback_result.get("result", [])
                    before_count = len(merged_chunks)
                    merged_chunks = _merge_chunk_results(
                        merged_chunks,
                        fallback_chunks,
                        fuse_limit,
                    )
                    _debug_file_logger.info(
                        "fallback_query=%r | extra_hits=%d | merged=%d (was %d)",
                        extra_query,
                        len(fallback_chunks),
                        len(merged_chunks),
                        before_count,
                    )

                result["result"] = merged_chunks
                result["fused_count"] = len(merged_chunks)

            except Exception:
                _logger.exception(
                    "Identifier retrieval fallback failed; "
                    "keeping primary search results."
                )

    _logger.info(
        "Search completed in %.0fms | "
        "mode=%s, vector_count=%d, "
        "keyword_count=%d, fused_count=%d",
        search_ms,
        result.get("mode"),
        result.get("vector_count", 0),
        result.get("keyword_count", 0),
        result.get("fused_count", 0),
    )

    # ========================================================
    # Step 4: Chunk results
    # ========================================================

    chunk_results = result.get(
        "result",
        [],
    )

    chunk_results = _filter_excluded_patterns(
        chunk_results,
        opts.must_not_conditions,
    )

    # ========================================================
    # Document / rerank configuration
    # ========================================================

    doc_id_field = (
        payload.rerank.document_id_field
        if payload.rerank
        else "document_id"
    )

    hybrid_rerank_enabled = bool(
        payload.rerank
        and payload.rerank.strategy == "hybrid"
    )

    identifier_bonus = (
        float(opts.identifier_bonus)
        if auto_route and hybrid_rerank_enabled
        else 0.0
    )

    # ========================================================
    # IDENTIFIER EXACT MATCH - CHUNK LEVEL
    #
    # This must happen BEFORE rerank because chunk payload
    # exists here.
    # ========================================================

    identifier_matches: list[dict[str, Any]] = []

    if (
        identifier
        and identifier_bonus > 0
    ):
        identifier_matches = _find_identifier_matches(
            chunk_results,
            identifier,
            doc_id_field,
            query_path=(
                detection.collapsed
                if detection is not None
                else identifier
            ),
        )

    matched_docs = {
        str(match["document_id"])
        for match in identifier_matches
    }

    # ========================================================
    # VERSION MATCH - DOCUMENT LEVEL
    #
    # Second matching signal (independent of identifier). When the query
    # carries a version (e.g. "v4.0") we flag the identifier-matched docs
    # whose payload docVersion matches it. These get an extra version bonus
    # on top of identifier_bonus so the exact-version document is prioritized
    # within the same docMaVanBan family. We never fold the version into the
    # identifier itself, so other versions remain retrievable.
    # ========================================================

    version_bonus = (
        float(opts.version_bonus)
        if auto_route and hybrid_rerank_enabled and opts.version_match_enabled()
        else 0.0
    )

    query_version = (
        detection.version
        if detection is not None
        else None
    )

    version_matched_docs: set[str] = set()

    if (
        query_version
        and version_bonus > 0
        and matched_docs
    ):
        for item in chunk_results:
            chunk_payload = item.get("payload") or {}
            doc_id = str(chunk_payload.get(doc_id_field, ""))
            if doc_id not in matched_docs:
                continue
            if version_match(query_version, chunk_payload.get("docVersion")):
                version_matched_docs.add(doc_id)

        _debug_file_logger.info(
            "========== VERSION MATCH =========="
        )
        _debug_file_logger.info(
            "query_version=%r | version_bonus=%.6f",
            query_version,
            version_bonus,
        )
        _debug_file_logger.info(
            "version_matched_docs=%s",
            sorted(version_matched_docs),
        )

    # ========================================================
    # DEBUG: ALL CHUNKS BEFORE RERANK
    # ========================================================

    _debug_file_logger.info(
        "========== ALL CHUNKS BEFORE RERANK =========="
    )

    _debug_file_logger.info(
        "chunk_count=%d",
        len(chunk_results),
    )

    for rank, item in enumerate(
        chunk_results,
        1,
    ):
        chunk_payload = item.get("payload") or {}

        chunk_id = item.get("id")
        document_id = chunk_payload.get(
            doc_id_field
        )

        chunk_index = chunk_payload.get(
            "chunk_index"
        )

        chunk_text = (
            chunk_payload.get("chunk_text")
            or chunk_payload.get("chunkText")
            or chunk_payload.get("text")
            or chunk_payload.get("content")
            or ""
        )

        matching_chunks = [
            match
            for match in identifier_matches
            if str(match["chunk_id"]) == str(chunk_id)
        ]

        is_identifier_exact_match = bool(
            matching_chunks
        )

        matched_field = (
            matching_chunks[0]["matched_field"]
            if matching_chunks
            else ""
        )

        match_marker = (
            "IDENTIFIER_EXACT_MATCH"
            if is_identifier_exact_match
            else "NO_IDENTIFIER_MATCH"
        )

        _debug_file_logger.info(
            "#%d chunk_id=%s score=%.6f "
            "document_id=%s chunk_index=%s "
            "docMaVanBan=%s docSoHieu=%s "
            "identifier_match=%s "
            "matched_field=%s "
            "chunk_text=%r",
            rank,
            chunk_id,
            float(item.get("score", 0.0)),
            document_id,
            chunk_index,
            chunk_payload.get("docMaVanBan"),
            chunk_payload.get("docSoHieu"),
            match_marker,
            matched_field,
            chunk_text,
        )

    # ========================================================
    # DEBUG: EXACT MATCH SUMMARY
    # ========================================================

    _debug_file_logger.info(
        "========== EXACT MATCH =========="
    )

    _debug_file_logger.info(
        "identifier=%r",
        identifier,
    )

    _debug_file_logger.info(
        "identifier_bonus=%.6f",
        identifier_bonus,
    )

    _debug_file_logger.info(
        "exact_match_enabled=%s",
        bool(
            identifier
            and identifier_bonus > 0
        ),
    )

    _debug_file_logger.info(
        "matched_chunk_count=%d",
        len(identifier_matches),
    )

    _debug_file_logger.info(
        "matched_doc_count=%d",
        len(matched_docs),
    )

    _debug_file_logger.info(
        "matched_docs=%s",
        sorted(matched_docs),
    )

    if identifier_matches:
        _debug_file_logger.info(
            "EXACT MATCH RESULT: MATCH FOUND"
        )

        _debug_file_logger.info(
            "========== EXACT MATCH CHUNKS =========="
        )

        for match in identifier_matches:
            _debug_file_logger.info(
                "EXACT MATCH CHUNK | "
                "rank=%d | "
                "chunk_id=%s | "
                "document_id=%s | "
                "chunk_index=%s | "
                "field=%s | "
                "value=%r | "
                "normalized=%r | "
                "score=%s | "
                "chunk_text=%r",
                match["rank"],
                match["chunk_id"],
                match["document_id"],
                match["chunk_index"],
                match["matched_field"],
                match["raw_value"],
                match["normalized_value"],
                match["score"],
                match["chunk_text"],
            )

    elif (
        identifier
        and identifier_bonus > 0
    ):
        _debug_file_logger.info(
            "EXACT MATCH RESULT: NO MATCH FOUND"
        )

    else:
        _debug_file_logger.info(
            "EXACT MATCH RESULT: NOT APPLICABLE"
        )

    # ========================================================
    # Rerank
    # ========================================================

    if payload.rerank and chunk_results:
        rerank_start = time.perf_counter()

        try:
            reranked = _rerank_service.apply(
                chunk_results,
                strategy=payload.rerank.strategy,
                document_id_field=(
                    payload.rerank.document_id_field
                ),
                top_chunks=int(
                    payload.rerank.top_chunks
                ),
                weights=(
                    payload.rerank.get_weights_tuple()
                ),
                mean_m=int(
                    payload.rerank.mean_m
                ),
                rrf_k=int(
                    payload.rerank.rrf_k
                ),
            )

        except ValueError as e:
            _logger.error(
                "Rerank failed: %s",
                e,
            )

            raise ApiError(
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="rerank_error",
                detail=f"Rerank failed: {e}",
                retryable=False,
            ) from e

        # ====================================================
        # Apply identifier bonus
        # ====================================================

        if matched_docs:
            _debug_file_logger.info(
                "========== APPLY IDENTIFIER BONUS =========="
            )

            _debug_file_logger.info(
                "bonus_target_documents=%s",
                sorted(matched_docs),
            )

            for doc in reranked:
                document_id = str(
                    doc.get(
                        "document_id",
                        "",
                    )
                )

                if document_id in matched_docs:
                    old_score = (
                        doc.get(
                            "score",
                            0.0,
                        )
                        or 0.0
                    )

                    # Level 1: identifier match. Level 2: identifier + version.
                    applied_version_bonus = (
                        version_bonus
                        if document_id in version_matched_docs
                        else 0.0
                    )

                    new_score = (
                        old_score
                        + identifier_bonus
                        + applied_version_bonus
                    )

                    source_matches = [
                        match
                        for match in identifier_matches
                        if str(match["document_id"])
                        == document_id
                    ]

                    _debug_file_logger.info(
                        "IDENTIFIER BONUS APPLIED | "
                        "document_id=%s | "
                        "score_before=%.6f | "
                        "identifier_bonus=%.6f | "
                        "version_bonus=%.6f | "
                        "version_match=%s | "
                        "score_after=%.6f | "
                        "source_exact_match_chunks=%d",
                        document_id,
                        old_score,
                        identifier_bonus,
                        applied_version_bonus,
                        document_id in version_matched_docs,
                        new_score,
                        len(source_matches),
                    )

                    for match in source_matches:
                        _debug_file_logger.info(
                            "BONUS SOURCE CHUNK | "
                            "document_id=%s | "
                            "chunk_id=%s | "
                            "chunk_index=%s | "
                            "matched_field=%s | "
                            "value=%r",
                            document_id,
                            match["chunk_id"],
                            match["chunk_index"],
                            match["matched_field"],
                            match["raw_value"],
                        )

                    doc["score"] = new_score

            reranked.sort(
                key=lambda d: d.get(
                    "score",
                    0.0,
                ),
                reverse=True,
            )

        # ====================================================
        # DEBUG: AFTER RERANK
        # ====================================================

        _debug_file_logger.info(
            "========== ALL DOCUMENTS AFTER RERANK =========="
        )

        _debug_file_logger.info(
            "document_count=%d",
            len(reranked),
        )

        for rank, doc in enumerate(
            reranked,
            1,
        ):
            document_id = str(
                doc.get(
                    "document_id",
                    "",
                )
            )

            is_bonus_document = (
                document_id in matched_docs
            )

            _debug_file_logger.info(
                "#%d document_id=%s score=%.6f "
                "identifier_bonus=%s",
                rank,
                document_id,
                float(
                    doc.get(
                        "score",
                        0.0,
                    )
                ),
                is_bonus_document,
            )

        # ====================================================
        # Score threshold
        # ====================================================

        n_before = len(reranked)

        if (
            hybrid_rerank_enabled
            and opts.score_threshold is not None
        ):
            threshold = float(
                opts.score_threshold
            )

            reranked = [
                doc
                for doc in reranked
                if (
                    doc.get(
                        "score",
                        0.0,
                    )
                    or 0.0
                ) >= threshold
            ]

        document_results = [
            {
                "document_id": doc["document_id"],
                "rank_document": rank,
            }
            for rank, doc in enumerate(
                reranked,
                1,
            )
        ]

        rerank_ms = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        _logger.info(
            "Rerank completed in %.0fms | "
            "strategy=%s, documents=%d "
            "(filtered %d→%d), "
            "id_bonus_docs=%d",
            rerank_ms,
            payload.rerank.strategy,
            len(document_results),
            n_before,
            len(reranked),
            len(matched_docs),
        )

    else:
        # ====================================================
        # No rerank
        # ====================================================

        if (
            hybrid_rerank_enabled
            and opts.score_threshold is not None
        ):
            _logger.warning(
                "score_threshold=%s ignored: "
                "only applies when rerank is enabled "
                "(raw fusion scores are not normalized "
                "to [0,1]).",
                opts.score_threshold,
            )

        document_results = (
            _group_chunks_to_documents(
                chunk_results,
                document_id_field=doc_id_field,
            )
        )

        # Promote exact identifier documents
        if matched_docs:
            matched = [
                d
                for d in document_results
                if d["document_id"]
                in matched_docs
            ]

            rest = [
                d
                for d in document_results
                if d["document_id"]
                not in matched_docs
            ]

            document_results = [
                {
                    "document_id": d["document_id"],
                    "rank_document": rank,
                }
                for rank, d in enumerate(
                    matched + rest,
                    1,
                )
            ]

    # ========================================================
    # DEBUG: FINAL DOCUMENT ORDER BEFORE LIMIT
    # ========================================================

    _debug_file_logger.info(
        "========== FINAL DOCUMENT ORDER BEFORE LIMIT =========="
    )

    _final_debug_docs = (
        reranked
        if payload.rerank and chunk_results
        else document_results
    )

    _debug_file_logger.info(
        "document_count=%d",
        len(_final_debug_docs),
    )

    for rank, doc in enumerate(
        _final_debug_docs,
        1,
    ):
        document_id = str(
            doc.get(
                "document_id",
                "",
            )
        )

        _debug_file_logger.info(
            "#%d document_id=%s final_score=%s "
            "identifier_exact_match=%s",
            rank,
            document_id,
            doc.get(
                "score",
                "n/a",
            ),
            document_id in matched_docs,
        )

    # ========================================================
    # Step 5: Apply limit
    # ========================================================

    document_results = document_results[
        : int(payload.limit)
    ]

    # ========================================================
    # Citation
    # ========================================================

    citation_ms = 0.0

    if (
        idf_task is not None
        and not document_results
    ):
        _logger.info(
            "Citation skipped: no documents to cite"
        )

    if (
        idf_task is not None
        and document_results
    ):
        citation_start = time.perf_counter()

        try:
            cite = (
                payload.citation_options
                or CitationOptions()
            )

            probed = await idf_task

            idf_wait_ms = (
                time.perf_counter()
                - citation_start
            ) * 1000

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
                secondary_tau=float(cite.secondary_tau),
            )

            citations = build_citations(
                document_results,
                chunk_results,
                payload.query,
                idf_from_probe(
                    query_tokens,
                    query_indices,
                    probed,
                ),
                document_id_field=doc_id_field,
                config=config,
            )

            for doc in document_results:
                block = citations.get(
                    doc["document_id"]
                )

                if block:
                    doc["citation_tier"] = (
                        block["tier"]
                    )

                    doc["citations"] = (
                        block["citations"]
                    )

            citation_ms = (
                time.perf_counter()
                - citation_start
            ) * 1000

            tiers = Counter(
                block["tier"]
                for block in citations.values()
            )

            snippets = [
                c
                for block in citations.values()
                for c in block["citations"]
            ]

            _logger.info(
                "Citations built in %.0fms "
                "(idf wait %.0fms) | "
                "documents=%d, tiers=%s, "
                "snippets=%d, highlights=%d, "
                "probed_tokens=%d/%d | config=%s",
                citation_ms,
                idf_wait_ms,
                len(document_results),
                dict(tiers),
                len(snippets),
                sum(
                    len(c["highlights"])
                    for c in snippets
                ),
                len(probed),
                len(query_indices),
                asdict(config),
            )

        except Exception:
            citation_ms = (
                time.perf_counter()
                - citation_start
            ) * 1000

            _logger.exception(
                "Citation building failed, "
                "returning results without it."
            )

    # ========================================================
    # Summary
    # ========================================================

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    _logger.info(
        "Request completed in %.0fms | "
        "embed=%.0fms, search=%.0fms, "
        "citation=%.0fms, documents=%d",
        total_ms,
        embed_ms,
        search_ms,
        citation_ms,
        len(document_results),
    )

    return {
        "status": 200,
        "result": document_results,
    }
