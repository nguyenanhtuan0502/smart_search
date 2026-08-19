"""Search application service — orchestrate dense + sparse + RRF fusion."""

import asyncio
from typing import Any

from ..adapters.qdrant import QdrantClient
from ..logging import TraceIdAdapter, get_logger

_logger: TraceIdAdapter = get_logger(__name__)

# Hard limit on candidate retrieval to avoid overloading Qdrant.
MAX_CANDIDATE_LIMIT = 100


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
        }
        for rank, p in enumerate(points[:limit], start=1)
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
        score_threshold: float | None = None,
        payload_filter: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
        candidate_multiplier: int = 5,
        with_payload: bool = True,
    ) -> dict[str, Any]:
        """Execute search based on available vectors and return fused results."""

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

        if candidate_multiplier <= 0:
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

        candidate_limit = min(
            max(limit, limit * candidate_multiplier),
            MAX_CANDIDATE_LIMIT,
        )

        # ── Mode 1: Dense-only ──────────────────────────────────
        if dense_vector is not None and sparse_vector is None:
            _logger.info("Search mode: dense-only, collection=%s, limit=%d", collection, limit)
            result = await self._qdrant.dense_search(
                collection=collection,
                vector=dense_vector,
                limit=limit,
                score_threshold=score_threshold,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                _logger.error("Dense-only search failed: %s", result.get("detail"))
                return result

            points = result.get("result", [])
            return {
                "result": _format_single_results(
                    points, limit=limit, is_dense=True, score_type="similarity",
                ),
                "vector_count": len(points),
                "keyword_count": 0,
                "fused_count": min(len(points), limit),
                "mode": "vector_only",
            }

        # ── Mode 2: Keyword-only (sparse-only) ─────────────────
        if sparse_vector is not None and dense_vector is None:
            _logger.info("Search mode: keyword-only, collection=%s, limit=%d", collection, limit)
            result = await self._qdrant.sparse_search(
                collection=collection,
                sparse_vector=sparse_vector,
                limit=limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                _logger.error("Sparse-only search failed: %s", result.get("detail"))
                return result

            points = result.get("result", [])
            return {
                "result": _format_single_results(
                    points, limit=limit, is_dense=False, score_type="keyword",
                ),
                "vector_count": 0,
                "keyword_count": len(points),
                "fused_count": min(len(points), limit),
                "mode": "keyword_only",
            }

        # ── Mode 3: Hybrid (dense + sparse in parallel) ────────
        _logger.info(
            "Search mode: hybrid, collection=%s, limit=%d, candidate_limit=%d",
            collection, limit, candidate_limit,
        )

        dense_result, sparse_result = await asyncio.gather(
            self._qdrant.dense_search(
                collection=collection,
                vector=dense_vector,
                limit=candidate_limit,
                score_threshold=score_threshold,
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
        if isinstance(dense_result, BaseException):
            _logger.error("Dense search raised exception: %s", dense_result)
            dense_result = {"error": True, "status": 503, "detail": str(dense_result)}
        if isinstance(sparse_result, BaseException):
            _logger.error("Sparse search raised exception: %s", sparse_result)
            sparse_result = {"error": True, "status": 503, "detail": str(sparse_result)}

        dense_ok = not dense_result.get("error")
        sparse_ok = not sparse_result.get("error")

        # ── Both failed ─────────────────────────────────────────
        if not dense_ok and not sparse_ok:
            _logger.error("Both dense and sparse search failed")
            return {"error": True, "status": 503, "detail": "Both dense and sparse search failed"}

        # ── Graceful degradation based on error state ───────────
        if not dense_ok and sparse_ok:
            _logger.warning("Dense search failed, falling back to sparse-only")
            sparse_points = sparse_result.get("result", [])
            return {
                "result": _format_single_results(
                    sparse_points, limit=limit, is_dense=False, score_type="keyword",
                ),
                "vector_count": 0,
                "keyword_count": len(sparse_points),
                "fused_count": min(len(sparse_points), limit),
                "mode": "keyword_only_fallback",
            }

        if not sparse_ok and dense_ok:
            _logger.warning("Sparse search failed, falling back to dense-only")
            dense_points = dense_result.get("result", [])
            return {
                "result": _format_single_results(
                    dense_points, limit=limit, is_dense=True, score_type="similarity",
                ),
                "vector_count": len(dense_points),
                "keyword_count": 0,
                "fused_count": min(len(dense_points), limit),
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
            limit=limit,
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
