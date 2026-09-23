"""Search application service — orchestrate dense + sparse + RRF fusion."""

import asyncio
from typing import Any

from ..adapters.qdrant import QdrantClient
from ..logging import TraceIdAdapter, get_logger

_logger: TraceIdAdapter = get_logger(__name__)

# Hard limit on candidate retrieval to avoid overloading Qdrant.
MAX_CANDIDATE_LIMIT = 100

# Fusion strategies accepted by hybrid_search. The public DTO enforces this
# via Literal[...], but the service validates too so direct (non-HTTP) callers
# get a clear error instead of silently falling through to client-mode.
VALID_FUSION_MODES = frozenset({"client", "server_rrf", "batch_client_rrf"})


def rrf_fuse(
    dense_points: list[dict[str, Any]],
    sparse_points: list[dict[str, Any]],
    limit: int = 10,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion.

    Raises:
        ValueError: If weights are negative, both zero, or rrf_k <= 0.
    """
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if vector_weight == 0 and keyword_weight == 0:
        raise ValueError("At least one weight must be greater than 0.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    total = vector_weight + keyword_weight
    vector_weight /= total
    keyword_weight /= total

    scores: dict[str, dict[str, Any]] = {}

    for rank, p in enumerate(dense_points, start=1):
        pid = str(p["id"])
        scores[pid] = {
            "id": p["id"],
            "payload": p.get("payload"),
            "score": vector_weight / (rrf_k + rank),
            "score_type": "rrf",
            "vector_rank": rank,
            "keyword_rank": None,
            "dense_score": p.get("score"),
        }

    for rank, p in enumerate(sparse_points, start=1):
        pid = str(p["id"])
        if pid in scores:
            scores[pid]["score"] += keyword_weight / (rrf_k + rank)
            scores[pid]["keyword_rank"] = rank
        else:
            scores[pid] = {
                "id": p["id"],
                "payload": p.get("payload"),
                "score": keyword_weight / (rrf_k + rank),
                "score_type": "rrf",
                "vector_rank": None,
                "keyword_rank": rank,
                "dense_score": None,
            }

    fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return fused[:limit]


def _format_single_results(
    points: list[dict[str, Any]],
    *,
    limit: int,
    is_dense: bool,
    score_type: str = "similarity",
) -> list[dict[str, Any]]:
    """Format results from a single-mode search (dense or sparse)."""
    return [
        {
            "id": p["id"],
            "payload": p.get("payload"),
            "score": p.get("score", 0),
            "score_type": score_type,
            "vector_rank": rank if is_dense else None,
            "keyword_rank": rank if not is_dense else None,
            "dense_score": p.get("score", 0) if is_dense else None,
        }
        for rank, p in enumerate(points[:limit], start=1)
    ]


def _format_fused_results(
    points: list[dict[str, Any]],
    *,
    limit: int,
    score_type: str = "rrf",
) -> list[dict[str, Any]]:
    """Format results that are already fused server-side (e.g. Qdrant RRF).

    Unlike single-mode results, per-branch ranks are unknown here: the points
    are a single fused list, so ``vector_rank``/``keyword_rank`` are left as
    None rather than faked from the fused position.
    """
    return [
        {
            "id": p["id"],
            "payload": p.get("payload"),
            "score": p.get("score", 0),
            "score_type": score_type,
            "vector_rank": None,
            "keyword_rank": None,
            "dense_score": None,
        }
        for p in points[:limit]
    ]


class HybridSearchService:
    """Orchestrate hybrid search: dense + sparse → RRF fusion.

    Supports 3 modes:
    - dense-only:   only dense_vector provided
    - keyword-only: only sparse_vector provided
    - hybrid:       both vectors provided → parallel search + RRF fusion
    """

    def __init__(self, qdrant: QdrantClient) -> None:
        self._qdrant = qdrant

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[str, Any] | None = None,
        limit: int = 10,
        payload_filter: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
        candidate_multiplier: int = 5,
        fusion_mode: str = "client",
        with_payload: bool = True,
        fuse_limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute search based on available vectors and return fused results.

        ``limit`` bounds how many results this method returns. ``fuse_limit``
        (optional) is how many chunks to keep after fusion so a downstream
        reranker can see a wider pool: when set (e.g. to the candidate pool
        size), fusion keeps ``fuse_limit`` chunks instead of ``limit``, and the
        final document-level cut to ``limit`` happens in the caller. When
        ``fuse_limit`` is None the method keeps the legacy behavior (== limit).
        """

        # ── Input validation ────────────────────────────────────
        if dense_vector is None and sparse_vector is None:
            return {
                "error": True,
                "status": 422,
                "detail": "At least one of dense_vector or sparse_vector must be provided.",
            }

        if limit <= 0:
            return {
                "error": True,
                "status": 422,
                "detail": "limit must be greater than 0.",
            }

        if limit > MAX_CANDIDATE_LIMIT:
            return {
                "error": True,
                "status": 422,
                "detail": f"limit must not exceed {MAX_CANDIDATE_LIMIT}.",
            }

        # Service-level sanity bound only. The public DTO (SearchOptions)
        # enforces the stricter 2..20 range; this guards direct/non-HTTP
        # callers against a nonsensical multiplier.
        if candidate_multiplier < 1:
            return {
                "error": True,
                "status": 422,
                "detail": "candidate_multiplier must be greater than 0.",
            }

        if vector_weight < 0 or keyword_weight < 0:
            return {
                "error": True,
                "status": 422,
                "detail": "Weights must be non-negative.",
            }

        if vector_weight == 0 and keyword_weight == 0:
            return {
                "error": True,
                "status": 422,
                "detail": "At least one weight must be greater than 0.",
            }

        if rrf_k <= 0:
            return {
                "error": True,
                "status": 422,
                "detail": "rrf_k must be greater than 0.",
            }

        if fusion_mode not in VALID_FUSION_MODES:
            return {
                "error": True,
                "status": 422,
                "detail": (
                    f"Unsupported fusion_mode: {fusion_mode!r}. "
                    f"Must be one of {sorted(VALID_FUSION_MODES)}."
                ),
            }

        candidate_limit = min(
            max(limit, limit * candidate_multiplier),
            MAX_CANDIDATE_LIMIT,
        )

        # How many results to keep after fusion. Defaults to `limit` (legacy);
        # when the caller asks for a wider rerank pool via `fuse_limit`, keep up
        # to that many (never more than the candidate pool actually retrieved).
        out_limit = min(fuse_limit, candidate_limit) if fuse_limit else limit

        # ── Mode 1: Dense-only ──────────────────────────────────
        if dense_vector is not None and sparse_vector is None:
            _logger.info("Search mode: dense-only, collection=%s, limit=%d", collection, limit)
            result = await self._qdrant.dense_search(
                collection=collection,
                vector=dense_vector,
                limit=out_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                _logger.error("Dense-only search failed: %s", result.get("detail"))
                return result

            points = result.get("result", [])
            return {
                "result": _format_single_results(
                    points, limit=out_limit, is_dense=True, score_type="similarity",
                ),
                "vector_count": len(points),
                "keyword_count": 0,
                "fused_count": min(len(points), out_limit),
                "mode": "vector_only",
            }

        # ── Mode 2: Keyword-only (sparse-only) ─────────────────
        if sparse_vector is not None and dense_vector is None:
            _logger.info("Search mode: keyword-only, collection=%s, limit=%d", collection, limit)
            result = await self._qdrant.sparse_search(
                collection=collection,
                sparse_vector=sparse_vector,
                limit=out_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                _logger.error("Sparse-only search failed: %s", result.get("detail"))
                return result

            points = result.get("result", [])
            return {
                "result": _format_single_results(
                    points, limit=out_limit, is_dense=False, score_type="keyword",
                ),
                "vector_count": 0,
                "keyword_count": len(points),
                "fused_count": min(len(points), out_limit),
                "mode": "keyword_only",
            }

        # ── Mode 3a: Hybrid single-call (server-side RRF fusion) ───
        # Both vectors present + server fusion requested → 1 round-trip to
        # Qdrant Query API. Server-side RRF ignores vector/keyword weights.
        if fusion_mode == "server_rrf":
            # Both vectors are guaranteed non-None here (dense/sparse-only
            # modes returned above); narrow the type for the type checker.
            assert dense_vector is not None and sparse_vector is not None
            _logger.info(
                "Search mode: hybrid-single-call, collection=%s, limit=%d, "
                "candidate_limit=%d, fusion=rrf",
                collection, limit, candidate_limit,
            )
            result = await self._qdrant.query_hybrid(
                collection=collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                fusion="rrf",
                # Client-side RRF uses 1-based ranks (k + rank), while
                # Qdrant uses 0-based positions (k + position). Offset k so
                # both modes produce the same denominator for each rank.
                rrf_k=rrf_k + 1,
                limit=out_limit,
                prefetch_limit=candidate_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                _logger.error("Hybrid single-call search failed: %s", result.get("detail"))
                return result

            points = result.get("result", [])
            return {
                "result": _format_fused_results(
                    points, limit=out_limit, score_type="rrf",
                ),
                # Server-side fusion returns a single fused list; per-branch
                # candidate counts are not exposed by Qdrant here.
                "vector_count": None,
                "keyword_count": None,
                "fused_count": min(len(points), out_limit),
                "mode": "hybrid_single_call_rrf",
            }

        # ── Mode 3b: Hybrid batch retrieval + client-side weighted RRF ───
        # Retrieve dense + sparse candidates in ONE batch request, then fuse
        # in-process with the configured weights/rrf_k (best of both: 1
        # round-trip, full weighted-fusion control).
        #
        # Batch retrieval is atomic from the application's perspective. If the
        # batch request fails we return the error instead of falling back to
        # separate dense/sparse requests: a batch failure is treated as a
        # request/infrastructure error, not a signal to silently switch
        # behavior. (Contrast with the 'client' mode below, which does degrade
        # gracefully because it issues two independent requests.)
        if fusion_mode == "batch_client_rrf":
            assert dense_vector is not None and sparse_vector is not None
            _logger.info(
                "Search mode: hybrid-batch, collection=%s, limit=%d, candidate_limit=%d",
                collection, limit, candidate_limit,
            )
            batch = await self._qdrant.query_batch(
                collection=collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=candidate_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if batch.get("error"):
                _logger.error("Hybrid batch search failed: %s", batch.get("detail"))
                return batch

            dense_points = batch.get("dense", [])
            sparse_points = batch.get("sparse", [])

            _logger.info(
                "Hybrid batch search: dense_count=%d, sparse_count=%d",
                len(dense_points), len(sparse_points),
            )

            fused = rrf_fuse(
                dense_points=dense_points,
                sparse_points=sparse_points,
                limit=out_limit,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                rrf_k=rrf_k,
            )
            return {
                "result": fused,
                "vector_count": len(dense_points),
                "keyword_count": len(sparse_points),
                "fused_count": len(fused),
                "mode": "hybrid_batch",
            }

        # ── Mode 3c: Hybrid "client" (default) ─────────────────
        # Reached when fusion_mode == "client" (the two server modes above
        # returned early). Two independent Qdrant requests in parallel, then
        # weighted RRF in-process. This is the only hybrid mode that degrades
        # gracefully (one branch can fail and the other still returns results).
        _logger.info(
            "Search mode: hybrid-client, collection=%s, limit=%d, candidate_limit=%d",
            collection, limit, candidate_limit,
        )

        dense_result, sparse_result = await asyncio.gather(
            self._qdrant.dense_search(
                collection=collection,
                vector=dense_vector,
                limit=candidate_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            ),
            self._qdrant.sparse_search(
                collection=collection,
                sparse_vector=sparse_vector,
                limit=candidate_limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            ),
            return_exceptions=True,
        )

        # Handle unexpected exceptions from gather
        if isinstance(dense_result, Exception):
            _logger.error("Dense search raised exception: %s", dense_result)
            dense_result = {"error": True, "status": 503, "detail": str(dense_result)}
        if isinstance(sparse_result, Exception):
            _logger.error("Sparse search raised exception: %s", sparse_result)
            sparse_result = {"error": True, "status": 503, "detail": str(sparse_result)}

        dense_ok = not dense_result.get("error")
        sparse_ok = not sparse_result.get("error")

        # ── Both failed ─────────────────────────────────────────
        if not dense_ok and not sparse_ok:
            dense_status = dense_result.get("status", 503)
            sparse_status = sparse_result.get("status", 503)
            
            # Prefer 504 if either side timed out, otherwise use 503
            status = 504 if 504 in (dense_status, sparse_status) else 503
            
            _logger.error(
                "Both dense and sparse search failed: dense_status=%d, sparse_status=%d",
                dense_status,
                sparse_status,
            )
            return {
                "error": True,
                "status": status,
                "detail": "Both dense and sparse search failed"
            }

        # ── Graceful degradation based on error state ───────────
        if not dense_ok and sparse_ok:
            _logger.warning("Dense search failed, falling back to sparse-only")
            sparse_points = sparse_result.get("result", [])
            return {
                "result": _format_single_results(
                    sparse_points, limit=out_limit, is_dense=False, score_type="keyword",
                ),
                "vector_count": 0,
                "keyword_count": len(sparse_points),
                "fused_count": min(len(sparse_points), out_limit),
                "mode": "keyword_only_fallback",
            }

        if not sparse_ok and dense_ok:
            _logger.warning("Sparse search failed, falling back to dense-only")
            dense_points = dense_result.get("result", [])
            return {
                "result": _format_single_results(
                    dense_points, limit=out_limit, is_dense=True, score_type="similarity",
                ),
                "vector_count": len(dense_points),
                "keyword_count": 0,
                "fused_count": min(len(dense_points), out_limit),
                "mode": "vector_only_fallback",
            }

        # ── Both succeeded → RRF Fusion (even if one side is empty) ──
        dense_points = dense_result.get("result", [])
        sparse_points = sparse_result.get("result", [])

        _logger.info(
            "Hybrid search: dense_count=%d, sparse_count=%d",
            len(dense_points), len(sparse_points),
        )

        fused = rrf_fuse(
            dense_points=dense_points,
            sparse_points=sparse_points,
            limit=out_limit,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            rrf_k=rrf_k,
        )

        return {
            "result": fused,
            "vector_count": len(dense_points),
            "keyword_count": len(sparse_points),
            "fused_count": len(fused),
            "mode": "hybrid",
        }
