from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkResult:
    """A single chunk search result with its rank and score."""

    chunk_id: str
    document_id: str
    score: float
    rank: int


Ranking = list[tuple[str, float]]
Grouped = dict[str, list[ChunkResult]]


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────


def _group(results: Iterable[ChunkResult]) -> Grouped:
    grouped: Grouped = defaultdict(list)
    for result in results:
        grouped[result.document_id].append(result)
    return grouped


def _sorted(scores: dict[str, float]) -> Ranking:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _minmax(values: dict[str, float]) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


# ─────────────────────────────────────────────
# Reranking strategies
# ─────────────────────────────────────────────


def _maxp(grouped: Grouped) -> Ranking:
    """Score = max chunk score per document."""
    return _sorted({doc: max(c.score for c in chunks) for doc, chunks in grouped.items()})


def _top_m_mean(grouped: Grouped, m: int) -> Ranking:
    """Score = mean of top M chunk scores per document."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return _sorted({
        doc: fmean(sorted((c.score for c in chunks), reverse=True)[:m])
        for doc, chunks in grouped.items()
    })


def _hit_count(grouped: Grouped) -> Ranking:
    """Score = number of chunks matching per document."""
    return _sorted({doc: float(len(chunks)) for doc, chunks in grouped.items()})


def _rrf(grouped: Grouped, k: int = 60) -> Ranking:
    """Score = sum of 1/(k + rank) for each chunk per document."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return _sorted({doc: sum(1 / (k + c.rank) for c in chunks) for doc, chunks in grouped.items()})


# ─────────────────────────────────────────────
# Public rerank functions
# ─────────────────────────────────────────────


def maxp_rerank(results: list[ChunkResult]) -> Ranking:
    return _maxp(_group(results))


def top_m_mean_rerank(results: list[ChunkResult], m: int) -> Ranking:
    return _top_m_mean(_group(results), m)


def hit_count_rerank(results: list[ChunkResult]) -> Ranking:
    return _hit_count(_group(results))


def rrf_rerank(results: list[ChunkResult], k: int = 60) -> Ranking:
    return _rrf(_group(results), k)


def hybrid_rerank(
    results: list[ChunkResult],
    weights: tuple[float, float, float, float] = (0.40, 0.25, 0.15, 0.20),
    mean_m: int = 3,
    rrf_k: int = 60,
) -> Ranking:
    """Min-max normalize each strategy signal, then weighted combination."""
    if not results:
        return []

    if len(weights) != 4 or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-9:
        raise ValueError("weights must be four non-negative values summing to 1")

    grouped = _group(results)

    component_rankings = (
        _maxp(grouped),
        _top_m_mean(grouped, mean_m),
        _hit_count(grouped),
        _rrf(grouped, rrf_k),
    )
    normalized = [_minmax(dict(ranking)) for ranking in component_rankings]
    document_ids = normalized[0].keys()
    return _sorted({
        doc: sum(weight * values[doc] for weight, values in zip(weights, normalized, strict=True))
        for doc in document_ids
    })


# ─────────────────────────────────────────────
# Rerank Service
# ─────────────────────────────────────────────


class RerankService:
    """Apply document-level reranking on chunk search results.

    Takes chunk-level RRF results and produces document-level rankings
    with top chunks per document.
    """

    def apply(
        self,
        chunk_results: list[dict[str, Any]],
        *,
        strategy: str = "hybrid",
        document_id_field: str = "document_id",
        top_chunks: int = 3,
        weights: tuple[float, float, float, float] | None = None,
        mean_m: int = 3,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        if not chunk_results:
            return []

        if top_chunks < 1:
            raise ValueError("top_chunks must be >= 1")

        # Build ChunkResult list
        chunk_result_items = self._build_chunk_results(chunk_results, document_id_field)

        # Apply strategy
        ranking = self._apply_strategy(
            chunk_result_items,
            strategy=strategy,
            weights=weights,
            mean_m=mean_m,
            rrf_k=rrf_k,
        )

        # Build document-level output
        return self._build_document_results(
            chunk_results, ranking, document_id_field, top_chunks
        )

    def _build_chunk_results(
        self,
        chunk_results: list[dict[str, Any]],
        document_id_field: str,
    ) -> list[ChunkResult]:
        """Convert raw chunk dicts to ChunkResult dataclass instances.

        Raises ValueError if a chunk is missing the document_id_field in its payload.
        """
        items: list[ChunkResult] = []
        for rank, item in enumerate(chunk_results, start=1):
            payload = item.get("payload") or {}
            document_id_value = payload.get(document_id_field)
            if document_id_value is None:
                raise ValueError(
                    f"Missing '{document_id_field}' in chunk payload for chunk "
                    f"{item.get('id', '<unknown>')}. "
                    f"Reranking requires every chunk to have a document identifier."
                )
            items.append(
                ChunkResult(
                    chunk_id=str(item.get("id", "")),
                    document_id=str(document_id_value),
                    score=item.get("score", 0.0),
                    rank=rank,
                )
            )
        return items

    def _apply_strategy(
        self,
        items: list[ChunkResult],
        *,
        strategy: str,
        weights: tuple[float, float, float, float] | None,
        mean_m: int,
        rrf_k: int,
    ) -> Ranking:
        """Dispatch to the correct reranking strategy.

        Raises ValueError for unsupported strategies.
        """
        if strategy == "maxp":
            return maxp_rerank(items)
        if strategy == "top_m_mean":
            return top_m_mean_rerank(items, m=mean_m)
        if strategy == "hit_count":
            return hit_count_rerank(items)
        if strategy == "rrf":
            return rrf_rerank(items, k=rrf_k)
        if strategy == "hybrid":
            w = weights or (0.40, 0.25, 0.15, 0.20)
            return hybrid_rerank(items, weights=w, mean_m=mean_m, rrf_k=rrf_k)
        raise ValueError(f"Unsupported rerank strategy: {strategy!r}")

    def _build_document_results(
        self,
        chunk_results: list[dict[str, Any]],
        ranking: Ranking,
        document_id_field: str,
        top_chunks: int,
    ) -> list[dict[str, Any]]:
        """Build document-level results with top chunks per document."""
        doc_chunks: dict[str, list[dict[str, Any]]] = {}
        for item in chunk_results:
            payload = item.get("payload") or {}
            document_id = str(payload.get(document_id_field, ""))
            doc_chunks.setdefault(document_id, []).append(item)

        document_results: list[dict[str, Any]] = []
        for doc_id, doc_score in ranking:
            chunks = doc_chunks.get(doc_id, [])
            chunks_sorted = sorted(
                chunks, key=lambda c: c.get("score", 0), reverse=True
            )
            document_results.append({
                "document_id": doc_id,
                "score": doc_score,
                "chunks": [
                    {
                        "chunk_id": str(c.get("id", "")),
                        "payload": c.get("payload"),
                        "score": c.get("score", 0.0),
                        "score_type": c.get("score_type", "rrf"),
                        "vector_rank": c.get("vector_rank"),
                        "keyword_rank": c.get("keyword_rank"),
                    }
                    for c in chunks_sorted[:top_chunks]
                ],
            })

        return document_results
