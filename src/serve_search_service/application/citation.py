import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..adapters.embedding_engine import _hash_token, _tokenize
from ..logging import TraceIdAdapter, get_logger

_logger: TraceIdAdapter = get_logger(__name__)

MIN_CITATION_CHARS = 40
MIN_CITATION_WORDS = 3

SNAP_RADIUS = 120

_WORD = re.compile(r"\w", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?;:\n]\s|\n")

_MD_FENCE = re.compile(r"`{1,3}")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_EMPHASIS = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_MD_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]{4,}\|?\s*$", re.M)
_DOT_LEADER = re.compile(r"[.…]{4,}")
_HTML_BREAK = re.compile(r"<br\s*/?>", re.I)
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def markdown_to_plaintext(text: str) -> str:
    if not text:
        return ""
    out = _MD_FENCE.sub("", text)
    out = _HTML_BREAK.sub(" ", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_EMPHASIS.sub(r"\1", out)
    out = _MD_BULLET.sub("", out)
    out = _MD_TABLE_RULE.sub("", out)
    out = _DOT_LEADER.sub(" ", out)
    out = _SPACES.sub(" ", out)
    return _BLANK_LINES.sub("\n\n", out).strip()


def is_citable(plaintext: str) -> bool:
    if len(plaintext) < MIN_CITATION_CHARS:
        return False
    return len(_WORD.findall(plaintext)) > 0 and len(_tokenize(plaintext)) >= MIN_CITATION_WORDS


def _gram_pattern(tokens: tuple[str, ...]) -> re.Pattern[str]:
    body = r"\W+".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE | re.UNICODE)


MAX_PHRASE_TOKENS = 6
NESTED_RATIO = 0.9
SECONDARY_TAU = 1.0

STOPWORDS = frozenset(
    """
    tôi tao mình bạn anh chị em chúng ta nó ai
    gì nào đâu sao bao nhiêu vậy
    thì là mà và hay hoặc với của các những có không chưa được bị phải cần nên
    muốn hãy đã đang sẽ vẫn cũng rất quá nữa này kia đó ấy ở tại về để khi nếu
    vì do bởi nhưng rằng thôi à ạ nhé nhỉ ơi hả chứ
    """.split()  # noqa: SIM905
)

STOP_PHRASES = (
    ("như", "thế", "nào"),
    ("làm", "thế", "nào"),
    ("thế", "nào"),
    ("làm", "sao"),
    ("làm", "gì"),
    ("là", "gì"),
    ("tại", "sao"),
    ("vì", "sao"),
    ("bao", "nhiêu"),
    ("bao", "giờ"),
    ("có", "phải"),
)


def _stop_positions(tokens: list[str]) -> set[int]:
    blocked = {i for i, t in enumerate(tokens) if t in STOPWORDS}
    for phrase in STOP_PHRASES:
        size = len(phrase)
        for start in range(len(tokens) - size + 1):
            if tuple(tokens[start : start + size]) == phrase:
                blocked.update(range(start, start + size))
    return blocked


@dataclass(frozen=True)
class Keyword:
    text: str
    tokens: tuple[str, ...]
    idf: float
    pattern: re.Pattern[str] = field(compare=False)
    span: tuple[int, int] = field(default=(0, 0), compare=False)


def select_keywords(
    query: str,
    pool_plaintexts: list[str],
    idf_by_token: dict[str, float],
    *,
    tau: float,
    top_m: int,
    min_cohesion: float,
    max_phrase_tokens: int = MAX_PHRASE_TOKENS,
    nested_ratio: float = NESTED_RATIO,
    df_cache: dict[tuple[str, ...], int] | None = None,
) -> list[Keyword]:
    """``df_cache`` maps a gram to its pool DF. It depends only on the pool, not on
    tau, so callers selecting twice over the same pool can share it."""
    tokens = _tokenize(query)
    if not tokens:
        return []

    blocked = _stop_positions(tokens)
    lowered = [t.lower() for t in pool_plaintexts]
    if df_cache is None:
        df_cache = {}

    def pool_df(gram: tuple[str, ...]) -> int:
        if gram not in df_cache:
            pattern = _gram_pattern(gram)
            df_cache[gram] = sum(1 for text in lowered if pattern.search(text))
        return df_cache[gram]

    candidates: dict[tuple[int, int], tuple[float, float, int]] = {}
    low_cohesion: list[str] = []
    for size in range(1, max_phrase_tokens + 1):
        for start in range(len(tokens) - size + 1):
            stop = start + size
            if any(p in blocked for p in range(start, stop)):
                continue
            gram = tuple(tokens[start:stop])
            score = max(idf_by_token.get(t, 0.0) for t in gram)
            if size == 1:
                candidates[(start, stop)] = (score, 1.0, 0)
                continue
            occurrences = pool_df(gram)
            floor = min(pool_df((t,)) for t in gram)
            cohesion = occurrences / floor if floor else 0.0
            if cohesion >= min_cohesion:
                candidates[(start, stop)] = (score, cohesion, occurrences)
            elif occurrences:
                low_cohesion.append(f"{' '.join(gram)}({cohesion:.2f})")

    order = sorted(
        candidates,
        key=lambda span: (
            span[1] - span[0] == 1,
            -candidates[span][1],
            -candidates[span][0],
            span[0] - span[1],
        ),
    )

    covered: set[int] = set()

    def free(span: tuple[int, int]) -> bool:
        return span in candidates and not covered.intersection(range(*span))

    def maximal(span: tuple[int, int]) -> bool:
        start, stop = span
        occurrences = candidates[span][2]
        return not any(
            free(ext) and candidates[ext][2] >= nested_ratio * occurrences
            for ext in ((start - 1, stop), (start, stop + 1))
        )

    picks: list[tuple[int, int]] = []
    while len(picks) < top_m:
        pick = next(
            (
                span
                for span in order
                if candidates[span][0] >= tau
                and free(span)
                and (span[1] - span[0] == 1 or maximal(span))
            ),
            None,
        )
        if pick is None:
            break
        picks.append(pick)
        covered.update(range(*pick))

    def orphaned(pos: int) -> bool:
        return any(
            0 <= n < len(tokens)
            and n not in blocked
            and n not in covered
            and not any(
                stop - start > 1 and start <= n < stop and not start <= pos < stop
                for start, stop in candidates
            )
            for n in (pos - 1, pos + 1)
        )

    resolved: list[tuple[int, int]] = []
    orphans: list[str] = []
    for pick in picks:
        if pick[1] - pick[0] > 1 or not orphaned(pick[0]):
            resolved.append(pick)
            continue
        covered.discard(pick[0])
        pos = pick[0]
        spans = [
            (start, start + size)
            for size in range(2, max_phrase_tokens + 1)
            for start in range(max(0, pos - size + 1), min(pos, len(tokens) - size) + 1)
        ]
        options = [
            span
            for span in spans
            if not any(p in blocked or p in covered for p in range(*span))
            and pool_df(tuple(tokens[span[0] : span[1]])) > 0
        ]
        if options:
            whole = min(
                options,
                key=lambda span: (span[1] - span[0], -pool_df(tuple(tokens[span[0] : span[1]]))),
            )
            resolved.append(whole)
            covered.update(range(*whole))
            orphans.append(f"{tokens[pos]}->{' '.join(tokens[whole[0] : whole[1]])}")
        else:
            orphans.append(f"{tokens[pos]}->dropped")

    if _logger.isEnabledFor(logging.DEBUG):
        token_info = []
        for pos, token in enumerate(tokens):
            idf = idf_by_token.get(token, 0.0)
            note = "stop" if pos in blocked else f"{idf:.2f}" + ("" if idf >= tau else "<tau")
            token_info.append(f"{token}({note})")
        _logger.debug(
            "Keyword selection | tokens=%s | low_cohesion=%s | orphans=%s",
            " ".join(token_info),
            ", ".join(low_cohesion) or "-",
            ", ".join(orphans) or "-",
        )

    chosen: list[Keyword] = []
    for start, stop in resolved:
        gram = tuple(tokens[start:stop])
        score = max(idf_by_token.get(t, 0.0) for t in gram)
        chosen.append(
            Keyword(
                text=" ".join(gram),
                tokens=gram,
                idf=score,
                pattern=_gram_pattern(gram),
                span=(start, stop),
            )
        )
    return chosen


def secondary_keywords(
    query: str,
    pool_plaintexts: list[str],
    idf_by_token: dict[str, float],
    primary: list[Keyword],
    config: "CitationConfig",
    df_cache: dict[tuple[str, ...], int] | None = None,
) -> list[Keyword]:
    if config.secondary_tau >= config.tau:
        return []
    lower = select_keywords(
        query,
        pool_plaintexts,
        idf_by_token,
        tau=config.secondary_tau,
        top_m=config.top_m,
        min_cohesion=config.min_cohesion,
        max_phrase_tokens=config.max_phrase_tokens,
        nested_ratio=config.nested_ratio,
        df_cache=df_cache,
    )
    grams = {keyword.tokens for keyword in primary}

    def redundant(keyword: Keyword) -> bool:
        if keyword.tokens in grams:
            return True
        start, stop = keyword.span
        for p_start, p_stop in (p.span for p in primary):
            if stop <= p_start or p_stop <= start:
                continue
            if start <= p_start and p_stop <= stop and (start, stop) != (p_start, p_stop):
                continue
            return True
        return False

    return [k for k in lower if len(k.tokens) > 1 and not redundant(k)]


def _merge_spans(text: str, spans: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end, weight in ordered[1:]:
        last = merged[-1]
        gap = text[int(last[1]) : start]
        if start <= last[1] or (len(gap) <= 2 and not _WORD.search(gap)):
            last[1] = max(last[1], end)
            last[2] = max(last[2], weight)
        else:
            merged.append([start, end, weight])
    return [(int(s), int(e), float(w)) for s, e, w in merged]


def find_spans(text: str, keywords: list[Keyword]) -> list[tuple[int, int, float]]:
    hits = [
        (match.start(), match.end(), keyword.idf)
        for keyword in keywords
        for match in keyword.pattern.finditer(text)
    ]
    return _merge_spans(text, hits)


def _snap_start(text: str, position: int) -> int:
    if position <= 0:
        return 0
    window = text[max(0, position - SNAP_RADIUS) : position]
    ends = list(_SENTENCE_END.finditer(window))
    if ends:
        return max(0, position - SNAP_RADIUS) + ends[-1].end()
    while position < len(text) and _WORD.match(text[position]):
        position += 1
    return position


def _snap_end(text: str, position: int) -> int:
    if position >= len(text):
        return len(text)
    match = _SENTENCE_END.search(text, position, min(len(text), position + SNAP_RADIUS))
    if match is not None:
        return match.end()
    while position > 0 and _WORD.match(text[position - 1]):
        position -= 1
    return position


def make_window(
    text: str,
    spans: list[tuple[int, int, float]],
    width: int,
) -> tuple[str, list[tuple[int, int]], bool]:
    if len(text) <= width:
        return text, [(s, e) for s, e, _ in spans], False

    if not spans:
        end = _snap_end(text, width)
        return text[:end].rstrip() + "…", [], True

    lead = width // 8
    best_start, best_weight = 0, -1.0
    for span_start, _span_end, _weight in spans:
        start = max(0, span_start - lead)
        stop = start + width
        weight = sum(w for s, e, w in spans if s >= start and e <= stop)
        if weight > best_weight:
            best_weight, best_start = weight, start

    start = _snap_start(text, best_start)
    end = _snap_end(text, min(len(text), start + width))
    snippet = text[start:end].strip()
    offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))

    kept = [
        (s - offset, e - offset)
        for s, e, _ in spans
        if s >= offset and e <= offset + len(snippet)
    ]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    if prefix:
        kept = [(s + 1, e + 1) for s, e in kept]
    return prefix + snippet + suffix, kept, True


@dataclass(frozen=True)
class CitationConfig:
    tau: float = 2.5
    top_m: int = 6
    min_cohesion: float = 0.6
    window_chars: int = 400
    companion_ratio: float = 0.7
    max_chunks: int = 5
    semantic_gate: float = 0.5
    max_phrase_tokens: int = MAX_PHRASE_TOKENS
    nested_ratio: float = NESTED_RATIO
    secondary_tau: float = SECONDARY_TAU


TIER_LITERAL = "literal"
TIER_SEMANTIC = "semantic"
TIER_WHOLE = "whole"
TIER_TITLE_ONLY = "title_only"


def query_token_indices(query: str) -> tuple[list[str], list[int]]:
    seen: dict[str, int] = {}
    for token in _tokenize(query):
        if token not in seen:
            seen[token] = _hash_token(token)
    return list(seen), list(seen.values())


def idf_from_probe(
    tokens: list[str], indices: list[int], probed: dict[int, float]
) -> dict[str, float]:
    return {
        token: float(probed.get(index, 0.0))
        for token, index in zip(tokens, indices, strict=False)
    }


@dataclass
class _Chunk:
    point_id: str
    document_id: str
    chunk_index: int | None
    plaintext: str
    score: float
    dense_score: float | None
    spans: list[tuple[int, int, float]] = field(default_factory=list)


def _collect(
    chunk_results: list[dict[str, Any]], document_id_field: str
) -> tuple[dict[str, list[_Chunk]], list[str]]:
    by_document: dict[str, list[_Chunk]] = {}
    pool_texts: list[str] = []
    for item in chunk_results:
        payload = item.get("payload") or {}
        doc_id = payload.get(document_id_field)
        if doc_id is None:
            continue
        raw_index = payload.get("chunk_index")
        try:
            chunk_index = int(raw_index) if raw_index is not None else None
        except (TypeError, ValueError):
            chunk_index = None
        if chunk_index == 0:
            continue
        text = markdown_to_plaintext(payload.get("chunk_text") or "")
        if not is_citable(text):
            continue
        score = item.get("score")
        dense = item.get("dense_score")
        by_document.setdefault(str(doc_id), []).append(
            _Chunk(
                point_id=str(item.get("id", "")),
                document_id=str(doc_id),
                chunk_index=chunk_index,
                plaintext=text,
                score=float(score) if isinstance(score, (int, float)) else 0.0,
                dense_score=float(dense) if isinstance(dense, (int, float)) else None,
            )
        )
        pool_texts.append(text)
    return by_document, pool_texts


def _pick(chunks: list[_Chunk], config: CitationConfig) -> list[_Chunk]:
    ranked = sorted(chunks, key=lambda c: -c.score)
    anchor = ranked[0]
    if anchor.dense_score is None:
        return [anchor]
    floor = config.companion_ratio * anchor.dense_score
    companions = [
        c
        for c in ranked[1:]
        if c.dense_score is not None and c.dense_score >= floor
    ]
    selected = [anchor] + companions[: max(0, config.max_chunks - 1)]
    tail = sorted(
        selected[1:], key=lambda c: (c.chunk_index if c.chunk_index is not None else 1 << 30)
    )
    return [anchor] + tail


def _add_secondary(
    text: str, highlights: list[tuple[int, int]], extras: list[Keyword]
) -> list[tuple[int, int]]:
    spans = [(s, e, 1.0) for s, e in highlights] + find_spans(text, extras)
    return [(s, e) for s, e, _ in _merge_spans(text, spans)]


def _describe(chunk: _Chunk) -> str:
    dense = "-" if chunk.dense_score is None else f"{chunk.dense_score:.3f}"
    return (
        f"#{chunk.chunk_index}(score={chunk.score:.3f}, dense={dense}, spans={len(chunk.spans)})"
    )


def _tier(anchor: _Chunk, config: CitationConfig) -> str:
    if anchor.spans:
        return TIER_LITERAL
    if anchor.dense_score is not None and anchor.dense_score >= config.semantic_gate:
        return TIER_SEMANTIC
    return TIER_WHOLE


def build_citations(
    documents: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    query: str,
    idf_by_token: dict[str, float],
    *,
    document_id_field: str = "document_id",
    config: CitationConfig | None = None,
) -> dict[str, dict[str, Any]]:
    config = config or CitationConfig()
    wanted = {str(doc.get("document_id")) for doc in documents}
    by_document, pool_texts = _collect(chunk_results, document_id_field)
    by_document = {k: v for k, v in by_document.items() if k in wanted}

    df_cache: dict[tuple[str, ...], int] = {}
    keywords = select_keywords(
        query,
        pool_texts,
        idf_by_token,
        tau=config.tau,
        top_m=config.top_m,
        min_cohesion=config.min_cohesion,
        max_phrase_tokens=config.max_phrase_tokens,
        nested_ratio=config.nested_ratio,
        df_cache=df_cache,
    )
    extras = secondary_keywords(query, pool_texts, idf_by_token, keywords, config, df_cache)
    _logger.info(
        "Citation keywords: %s | secondary: %s",
        ", ".join(f"{k.text}({k.idf:.2f})" for k in keywords) or "(none above tau)",
        ", ".join(f"{k.text}({k.idf:.2f})" for k in extras) or "-",
    )

    out: dict[str, dict[str, Any]] = {}
    trace: dict[str, str] = {}
    for doc_id in wanted:
        chunks = by_document.get(doc_id) or []
        if not chunks:
            out[doc_id] = {"tier": TIER_TITLE_ONLY, "citations": []}
            trace[doc_id] = f"{doc_id}:{TIER_TITLE_ONLY}"
            continue

        for chunk in chunks:
            chunk.spans = find_spans(chunk.plaintext, keywords)
        matching = [c for c in chunks if c.spans] or chunks

        selected = _pick(matching, config)
        tier = _tier(selected[0], config)
        matched = sum(1 for c in chunks if c.spans)
        trace[doc_id] = (
            f"{doc_id}:{tier} {matched}/{len(chunks)} [{', '.join(_describe(c) for c in selected)}]"
        )

        citations: list[dict[str, Any]] = []
        for position, chunk in enumerate(selected):
            text, highlights, truncated = make_window(
                chunk.plaintext, chunk.spans, config.window_chars
            )
            if extras and tier == TIER_LITERAL:
                highlights = _add_secondary(text, highlights, extras)
            citations.append(
                {
                    "chunk_id": chunk.point_id,
                    "chunk_index": chunk.chunk_index,
                    "text": text,
                    "highlights": [{"start": s, "end": e} for s, e in highlights],
                    "match_count": len(chunk.spans),
                    "truncated": truncated,
                    "is_anchor": position == 0,
                }
            )
        out[doc_id] = {"tier": tier, "citations": citations}
    ranked = [trace[str(doc.get("document_id"))] for doc in documents]
    _logger.debug("Citation docs | %s", " | ".join(ranked) or "-")
    return out
