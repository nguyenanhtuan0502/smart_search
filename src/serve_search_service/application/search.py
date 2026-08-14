"""Search application service — orchestrate dense + sparse + RRF fusion."""

import asyncio
from typing import Any

from ..adapters.qdrant import QdrantClient
from ..logging import TraceIdAdapter, get_logger

_logger: TraceIdAdapter = get_logger(__name__)


def rrf_fuse(
    dense_points: list[dict[str, Any]],
    sparse_points: list[dict[str, Any]],
    limit: int = 10,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion."""
    total = vector_weight + keyword_weight
    if total > 0:
        vector_weight /= total
        keyword_weight /= total

    scores: dict[str, dict[str, Any]] = {}

    for rank, p in enumerate(dense_points, start=1):
        pid = str(p["id"])
        scores[pid] = {
            "id": p["id"],
            "payload": p.get("payload"),
            "rrf_score": vector_weight / (rrf_k + rank),
            "vector_rank": rank,
            "keyword_rank": None,
        }

    for rank, p in enumerate(sparse_points, start=1):
        pid = str(p["id"])
        if pid in scores:
            scores[pid]["rrf_score"] += keyword_weight / (rrf_k + rank)
            scores[pid]["keyword_rank"] = rank
        else:
            scores[pid] = {
                "id": p["id"],
                "payload": p.get("payload"),
                "rrf_score": keyword_weight / (rrf_k + rank),
                "vector_rank": None,
                "keyword_rank": rank,
            }

    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused[:limit]


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
        """Execute search based on available vectors and return fused results.

        Raises ValueError if neither dense_vector nor sparse_vector is provided.
        """
        if dense_vector is None and sparse_vector is None:
            return {
                "error": True,
                "status": 422,
                "detail": "At least one of dense_vector or sparse_vector must be provided.",
            }

        candidate_limit = min(limit * candidate_multiplier, 100)

        # ── Mode 1: Dense-only ──────────────────────────────────
        if dense_vector is not None and sparse_vector is None:
            result = await self._qdrant.dense_search(
                collection=collection,
                vector=dense_vector,
                limit=limit,
                score_threshold=score_threshold,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                return result

            points = result.get("result", [])
            return {
                "result": [
                    {
                        "id": p["id"],
                        "payload": p.get("payload"),
                        "rrf_score": p.get("score", 0),
                        "vector_rank": rank,
                        "keyword_rank": None,
                    }
                    for rank, p in enumerate(points[:limit], start=1)
                ],
                "vector_count": len(points),
                "keyword_count": 0,
                "fused_count": min(len(points), limit),
                "mode": "vector_only",
            }

        # ── Mode 2: Keyword-only (sparse-only) ─────────────────
        if sparse_vector is not None and dense_vector is None:
            result = await self._qdrant.sparse_search(
                collection=collection,
                sparse_vector=sparse_vector,
                limit=limit,
                payload_filter=payload_filter,
                with_payload=with_payload,
            )
            if result.get("error"):
                return result

            points = result.get("result", [])
            return {
                "result": [
                    {
                        "id": p["id"],
                        "payload": p.get("payload"),
                        "rrf_score": p.get("score", 0),
                        "vector_rank": None,
                        "keyword_rank": rank,
                    }
                    for rank, p in enumerate(points[:limit], start=1)
                ],
                "vector_count": 0,
                "keyword_count": len(points),
                "fused_count": min(len(points), limit),
                "mode": "keyword_only",
            }

        # ── Mode 3: Hybrid (dense + sparse in parallel) ────────
        assert dense_vector is not None  # guaranteed by Mode 1/2 checks above
        assert sparse_vector is not None  # guaranteed by Mode 1/2 checks above

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
        )

        dense_ok = not dense_result.get("error")
        sparse_ok = not sparse_result.get("error")

        if not dense_ok and not sparse_ok:
            return {"error": True, "status": 503, "detail": "Both dense and sparse search failed"}

        dense_points = dense_result.get("result", []) if dense_ok else []
        sparse_points = sparse_result.get("result", []) if sparse_ok else []

        # Graceful degradation: one side failed
        if dense_points and not sparse_points:
            return {
                "result": [
                    {
                        "id": p["id"],
                        "payload": p.get("payload"),
                        "rrf_score": p.get("score", 0),
                        "vector_rank": rank,
                        "keyword_rank": None,
                    }
                    for rank, p in enumerate(dense_points[:limit], start=1)
                ],
                "vector_count": len(dense_points),
                "keyword_count": 0,
                "fused_count": min(len(dense_points), limit),
                "mode": "vector_only",
            }

        if sparse_points and not dense_points:
            return {
                "result": [
                    {
                        "id": p["id"],
                        "payload": p.get("payload"),
                        "rrf_score": p.get("score", 0),
                        "vector_rank": None,
                        "keyword_rank": rank,
                    }
                    for rank, p in enumerate(sparse_points[:limit], start=1)
                ],
                "vector_count": 0,
                "keyword_count": len(sparse_points),
                "fused_count": min(len(sparse_points), limit),
                "mode": "keyword_only",
            }

        # RRF Fusion
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
