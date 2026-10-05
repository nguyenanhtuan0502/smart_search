import re
import unicodedata
from typing import Any

from ..adapters.embedding_engine import _tokenize

TITLE_FIELD = "docTenVB"
_TITLE_WORD = re.compile(r"\w+", re.UNICODE)
_ACCENT_FALLBACK_WEIGHT = 0.9

# "915/2024", "675/2019/QĐ-TPB.CB": a document number the code router did not
# catch. Amendment titles quote the number they amend, so matching on titles would
# rank the amendments above the document itself.
_DOCUMENT_NUMBER = re.compile(r"\d+\s*/\s*\d{4}")


def is_document_number_query(query: str) -> bool:
    """True when the query cites a document number (số hiệu) like ``030/2022``."""
    return bool(_DOCUMENT_NUMBER.search(query))


def _title_tokens(text: str) -> list[str]:
    return _tokenize(unicodedata.normalize("NFC", text))


def _fold_title_token(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token.replace("đ", "d"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _prepare_title_query(
    query: str, queryIdf: dict[str, float],
) -> tuple[list[str], dict[str, float]]:
    knownWeights = [value for value in queryIdf.values() if value > 0]
    fallbackWeight = max(knownWeights) if knownWeights else 1.0
    queryWeights: dict[str, float] = {}
    for token, value in queryIdf.items():
        parts = _title_tokens(token)
        if not parts:
            continue
        partWeight = (value if value > 0 else fallbackWeight) / len(parts)
        for part in parts:
            queryWeights[part] = max(queryWeights.get(part, 0.0), partWeight)
    queryTokens = _title_tokens(query)
    if queryTokens:
        queryWeights = {
            token: queryWeights.get(token, fallbackWeight) for token in queryTokens
        }
    return queryTokens, queryWeights


def _token_coverage(
    titleTokens: list[str], queryTokens: list[str], queryWeights: dict[str, float],
) -> float:
    if not queryWeights:
        return 0.0
    fallbackWeight = max(queryWeights.values())
    queryPairs = set(zip(queryTokens, queryTokens[1:], strict=False))
    if queryPairs:
        titlePairs = set(zip(titleTokens, titleTokens[1:], strict=False))
        pairWeights = {
            pair: queryWeights.get(pair[0], fallbackWeight)
            + queryWeights.get(pair[1], fallbackWeight)
            for pair in queryPairs
        }
        return sum(
            weight for pair, weight in pairWeights.items() if pair in titlePairs
        ) / sum(pairWeights.values())
    presentTokens = set(titleTokens)
    return sum(
        weight for token, weight in queryWeights.items() if token in presentTokens
    ) / sum(queryWeights.values())


def _prepared_title_coverage(
    title: str, queryTokens: list[str], queryWeights: dict[str, float],
    foldedQueryTokens: list[str], foldedQueryWeights: dict[str, float],
) -> float:
    titleTokens = _title_tokens(title)
    exactCoverage = _token_coverage(titleTokens, queryTokens, queryWeights)
    if exactCoverage == 1.0:
        return exactCoverage
    foldedCoverage = _token_coverage(
        [_fold_title_token(token) for token in titleTokens],
        foldedQueryTokens,
        foldedQueryWeights,
    )
    return max(exactCoverage, _ACCENT_FALLBACK_WEIGHT * foldedCoverage)


def title_coverage(title: str, query_idf: dict[str, float], query: str = "") -> float:
    if not query_idf:
        return 0.0
    queryTokens, queryWeights = _prepare_title_query(query, query_idf)
    return _prepared_title_coverage(
        title, queryTokens, queryWeights,
        [_fold_title_token(token) for token in queryTokens],
        _fold_title_weights(queryWeights),
    )


def _fold_title_weights(queryWeights: dict[str, float]) -> dict[str, float]:
    foldedWeights: dict[str, float] = {}
    for token, weight in queryWeights.items():
        foldedToken = _fold_title_token(token)
        foldedWeights[foldedToken] = foldedWeights.get(foldedToken, 0.0) + weight
    return foldedWeights


def title_scores(
    dense_points: list[dict[str, Any]],
    sparse_points: list[dict[str, Any]],
    query_idf: dict[str, float],
    *,
    query: str = "",
    document_id_field: str = "document_id",
    rrf_k: int = 10,
    min_coverage: float = 0.5,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    titles = title_texts(dense_points, sparse_points, document_id_field=document_id_field)
    for points in (dense_points, sparse_points):
        seenDocuments: set[str] = set()
        for rank, point in enumerate(points, 1):
            payload = point.get("payload") or {}
            doc_id = payload.get(document_id_field)
            if doc_id is None:
                continue
            doc_id = str(doc_id)
            if doc_id in seenDocuments:
                continue
            seenDocuments.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    if not fused or not query_idf:
        return {}

    queryTokens, queryWeights = _prepare_title_query(query, query_idf)
    foldedQueryTokens = [_fold_title_token(token) for token in queryTokens]
    foldedQueryWeights = _fold_title_weights(queryWeights)
    eligibleCoverage = {}
    for doc_id in fused:
        coverage = _prepared_title_coverage(
            titles.get(doc_id, ""),
            queryTokens, queryWeights, foldedQueryTokens, foldedQueryWeights,
        )
        if coverage >= min_coverage and coverage > 0:
            eligibleCoverage[doc_id] = coverage
    if not eligibleCoverage:
        return {}

    top = max(fused[doc_id] for doc_id in eligibleCoverage)
    return {
        doc_id: fused[doc_id] / top * coverage
        for doc_id, coverage in eligibleCoverage.items()
    }


def title_texts(
    *point_lists: list[dict[str, Any]], document_id_field: str = "document_id",
) -> dict[str, str]:
    """Each document's title (NFC), from the first point carrying it."""
    titles: dict[str, str] = {}
    for points in point_lists:
        for point in points:
            payload = point.get("payload") or {}
            doc_id = payload.get(document_id_field)
            title = payload.get(TITLE_FIELD)
            if doc_id is not None and title and str(doc_id) not in titles:
                titles[str(doc_id)] = unicodedata.normalize("NFC", str(title)).strip()
    return titles


def title_highlights(title: str, query: str) -> list[tuple[int, int]]:
    words = [(m.group().lower(), m.start(), m.end()) for m in _TITLE_WORD.finditer(title)]
    queryTokens = _title_tokens(query)
    if not words or not queryTokens:
        return []
    marked = [False] * len(words)
    for fold in (lambda token: token, _fold_title_token):
        tokens = [fold(word) for word, _, _ in words]
        wanted = [fold(token) for token in queryTokens]
        if len(wanted) == 1:
            for i, token in enumerate(tokens):
                marked[i] = marked[i] or token == wanted[0]
            continue
        pairs = set(zip(wanted, wanted[1:], strict=False))
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i + 1]) in pairs:
                marked[i] = marked[i + 1] = True
    spans: list[tuple[int, int]] = []
    for i, (_, start, end) in enumerate(words):
        if not marked[i]:
            continue
        if i > 0 and marked[i - 1]:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def rank_title_documents(
    scores: dict[str, float],
    *,
    limit: int,
    titles: dict[str, str] | None = None,
    query: str = "",
) -> list[dict[str, Any]]:
    """Title-branch documents, best first, as ``title_result`` items.

    With ``titles`` each item also carries its title and the matched spans.
    """
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    items: list[dict[str, Any]] = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        item: dict[str, Any] = {"document_id": doc_id, "rank_document": rank, "title_score": score}
        if titles is not None:
            title = titles.get(doc_id, "")
            item["title"] = title
            item["title_highlights"] = [
                {"start": s, "end": e} for s, e in title_highlights(title, query)
            ]
        items.append(item)
    return items
