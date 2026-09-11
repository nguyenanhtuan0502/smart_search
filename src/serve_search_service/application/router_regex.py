
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LABEL_RE = re.compile(
    r"\b(?:ma\s*(?:van\s*ban|hieu|so)?|so\s*hieu(?:\s*van\s*ban)?"
    r"|bieu\s*mau(?:\s*so)?|ky\s*hieu)\b\s*[:\-]?\s*"
)


_SEGMENT = r"[a-z]{1,4}\d{0,2}[a-z]?"
_CODE_SHAPE_RE = re.compile(rf"^{_SEGMENT}(?:/{_SEGMENT}){{1,5}}$")

_SOHIEU_SHAPE_RE = re.compile(r"^\d{1,5}/\d{4}(?:/[a-z0-9]{1,6}){1,4}$")

_TRAILING_RE = re.compile(
    r"\s*(?:la\s*ma\s*cua\s*van\s*ban\s*(?:nao|gi)?"
    r"|la\s*(?:gi|van\s*ban\s*(?:nao|gi)?|tai\s*lieu\s*(?:nao|gi)?)?"
    r"|noi\s*ve\s*gi|la\s*cua\s*ai|the\s*nao|o\s*dau)\s*$"
,    re.IGNORECASE
)

_LOOKUP_TAIL_RE = re.compile(
    r"\s+la\s*(?:ma\s*cua\s*van\s*ban|van\s*ban|gi|tai\s*lieu)"
    r"(?:\s*(?:nao|gi))?\s*$",
    re.IGNORECASE,
)

_MAX_WHOLE_WORDS = 8

# Characters treated as identifier separators; also used to normalize a stored
# code before comparing it with the query identifier.
_SEP_RE = re.compile(r"[\s.,\-/]+")
_NORMALIZE_STRIP_RE = re.compile(r"[\s./\-]+")


@dataclass(frozen=True)
class DocCodeDetection:
    """Result of :func:`detect`."""

    is_doc_code: bool
    kind: str  # "code" | "sohieu" | "none"
    candidate: str | None
    collapsed: str | None
    reason: str


def strip_diacritics(text: str) -> str:
    """Lowercase-friendly diacritics removal (đ/Đ -> d, plus combining marks)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def collapse_separators(text: str) -> str:
    """Collapse any run of separators (space . , - /) into a single '/'."""
    return _SEP_RE.sub("/", text).strip("/")


def _compact_identifier_to_code(candidate: str) -> str | None:
    source = strip_diacritics(candidate).strip()
    text = source.lower()
    if not text or re.search(r"[\s.,/\-]", text) or not any(ch.isdigit() for ch in text):
        return None

    matches: list[str] = []

    def dfs(start: int, parts: list[str]) -> None:
        if start == len(text):
            code = "/".join(parts)
            if parts and _has_digit_segment(code) and bool(_CODE_SHAPE_RE.fullmatch(code)):
                matches.append(code)
            return

        max_len = min(len(text) - start, 7)
        for length in range(2, max_len + 1):
            seg = text[start:start + length]
            if not re.fullmatch(_SEGMENT, seg):
                continue
            dfs(start + length, parts + [seg])

    dfs(0, [])
    if not matches:
        return None

    def seg_score(seg: str) -> int:
        if re.search(r"\d", seg):
            # Prefer the common ``PREFIX + digits`` boundary. A trailing
            # letter is still valid (for example ``BM01e``), but should not
            # consume the first letter of the next segment.
            return 20 + len(seg) if re.search(r"\d$", seg) else 10 + len(seg)
        if len(seg) in (2, 3):
            return 1
        return -3

    def score(code: str) -> int:
        total = 0
        offset = 0
        for seg in code.split("/"):
            total += seg_score(seg)
            source_seg = source[offset:offset + len(seg)]
            source_prefix_letters = "".join(ch for ch in source_seg[:-1] if ch.isalpha())
            if (
                re.search(r"\d[a-z]$", seg)
                and source_seg[-1:].islower()
                and source_prefix_letters.isupper()
            ):
                total += 20
            offset += len(seg)
        return total

    return max(matches, key=score)


def _has_digit_segment(collapsed: str) -> bool:
    return any(re.search(r"\d", seg) for seg in collapsed.split("/") if seg)


def _expand_mixed_identifier(raw: str) -> str:
    """Expand compact pieces while preserving separators inside a candidate."""
    parts: list[str] = []
    for token in re.split(r"[\s.,/\-]+", raw):
        if not token:
            continue
        parts.append(_compact_identifier_to_code(token) or strip_diacritics(token.lower()))
    return "/".join(parts)


def _extract_identifier_from_sentence(text: str) -> str | None:

    has_lookup_tail = bool(_LOOKUP_TAIL_RE.search(text))
    clean = _TRAILING_RE.sub("", text).strip()
    if not clean:
        return None

    tokens = re.findall(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*", clean)
    if not tokens:
        return None

    best_candidate: str | None = None
    best_score: tuple[int, int, int] | None = None

    for start in range(len(tokens)):
        if not any(ch.isdigit() for ch in tokens[start]):
            continue

        for end in range(start + 1, min(start + 6, len(tokens)) + 1):
            if end < len(tokens) and not has_lookup_tail:
                continue
            next_token = tokens[end - 1]
            if end > start + 1 and not any(ch.isdigit() for ch in next_token) and len(next_token) > 4:
                break
            raw = " ".join(tokens[start:end])
            candidate = _compact_identifier_to_code(raw) or _expand_mixed_identifier(raw)
            if not candidate:
                continue

            if _CODE_SHAPE_RE.fullmatch(candidate) and _has_digit_segment(candidate):
                score = (candidate.count("/"), len(candidate), len(raw.replace(" ", "")))
                if best_score is None or score > best_score:
                    best_candidate = candidate
                    best_score = score

    return best_candidate


def normalize_identifier(value: str | None) -> str:

    if not value:
        return ""
    return _NORMALIZE_STRIP_RE.sub("", strip_diacritics(str(value).lower()))


def detect(query: str) -> DocCodeDetection:
    """Detect whether ``query`` is a document-code / số-hiệu lookup."""
    clean_query = strip_diacritics(query).strip()
    norm = clean_query.lower()

    extracted = _extract_identifier_from_sentence(clean_query)
    if extracted:
        candidate = extracted
        via = "embedded_id"
    else:
        match = _LABEL_RE.search(norm)
        if match:
            candidate = norm[match.end():].strip()
            via = "label_match"
        else:
            if len(norm.split()) > _MAX_WHOLE_WORDS:
                return DocCodeDetection(
                    is_doc_code=False, kind="none", candidate=None, collapsed=None,
                    reason="too_many_words",
                )
            candidate = norm
            via = "whole_short"
    candidate = _TRAILING_RE.sub("", candidate).strip()

    collapsed = collapse_separators(candidate)
    compact_collapsed = _compact_identifier_to_code(candidate)
    if compact_collapsed:
        collapsed = compact_collapsed

    code_ok = bool(_CODE_SHAPE_RE.fullmatch(collapsed))
    sohieu_ok = bool(_SOHIEU_SHAPE_RE.fullmatch(collapsed))
    digit_ok = _has_digit_segment(collapsed)

    if sohieu_ok:
        return DocCodeDetection(
            is_doc_code=True, kind="sohieu", candidate=candidate, collapsed=collapsed,
            reason=f"{via}; sohieu_shape",
        )
    if code_ok and digit_ok:
        return DocCodeDetection(
            is_doc_code=True, kind="code", candidate=candidate, collapsed=collapsed,
            reason=f"{via}; code_shape",
        )

    return DocCodeDetection(
        is_doc_code=False, kind="none", candidate=candidate, collapsed=collapsed,
        reason=f"{via}; no_shape",
    )


def is_doc_code(query: str) -> bool:
    """Convenience wrapper: ``True`` if ``query`` looks like an identifier lookup."""
    return detect(query).is_doc_code


def query_identifier(query: str) -> str:
    """Return the normalized identifier for a code query, or '' if not a code."""
    result = detect(query)
    if not result.is_doc_code or not result.collapsed:
        return ""
    return normalize_identifier(result.collapsed)
