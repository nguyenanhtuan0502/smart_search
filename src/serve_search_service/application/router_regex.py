from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..settings import ServiceSettings

_settings = ServiceSettings()


# ============================================================
# LABELS
# ============================================================

_LABEL_RE = re.compile(
    r"\b(?:"
    r"ma\s*(?:van\s*ban|hieu|so)?"
    r"|so\s*hieu(?:\s*van\s*ban)?"
    r"|so(?=\s*\d)"
    r"|bieu\s*mau(?:\s*so)?"
    r"|ky\s*hieu"
    r")\b\s*[:\-]?\s*",
    re.IGNORECASE,
)


# ============================================================
# SHAPES
# ============================================================

# _CODE_SHAPE_RE and _SO_HIEU_SHAPE_RE are compiled from settings so they can
# be overridden via env vars without changing source code:
#   SERVE_SEARCH_DEFAULT_CODE_SHAPE_PATTERN
#   SERVE_SEARCH_DEFAULT_SOHIEU_SHAPE_PATTERN

_CODE_SHAPE_RE = re.compile(
    _settings.default_code_shape_pattern,
    re.IGNORECASE,
)

# One segment of a compact identifier, e.g. "qd", "bm07", "sp31a".
_SEGMENT = r"[a-z]{1,4}\d{0,2}[a-z]?"

_SO_HIEU_SHAPE_RE = re.compile(
    _settings.default_sohieu_shape_pattern,
    re.IGNORECASE,
)


# ============================================================
# IDENTIFIER-MATCH / RETRIEVAL SEGMENT PATTERNS
# ============================================================
# Named patterns shared by the identifier matching + retrieval helpers below.
# Previously these were inline literals duplicated across api/router.py.
# Centralizing them here keeps all identifier regex in one place.

# A code segment that MUST contain digits, e.g. "sp13", "bm27", "hd01".
# Used by family Rule 4 and middle-segment variant generation.
_SEG_DIGIT = r"[a-z]{1,4}\d{1,3}[a-z]?"
_SEG_DIGIT_RE = re.compile(_SEG_DIGIT, re.IGNORECASE)
# Same, anchored at end (middle-segment fullmatch originally used a `$`).
_SEG_DIGIT_END_RE = re.compile(_SEG_DIGIT + r"$", re.IGNORECASE)

# A code segment where digits are OPTIONAL, e.g. "cn", "td", "sp13".
# Used when validating typo / confusable variants.
_SEG_OPT_DIGIT = r"[a-z]{1,4}\d{0,3}[a-z]?"
_SEG_OPT_DIGIT_RE = re.compile(_SEG_OPT_DIGIT, re.IGNORECASE)

# A pure-letter segment, e.g. "cn", "td", "bm".
_SEG_LETTERS = r"[a-z]{1,4}"
_SEG_LETTERS_RE = re.compile(_SEG_LETTERS, re.IGNORECASE)

# Leading "PLxx" family prefix, e.g. "pl01", "pl36".
_PL_PREFIX_RE = re.compile(r"pl\d{1,3}", re.IGNORECASE)

# Family prefix candidate: 1-4 chars of letters and/or digits.
# (Length is checked separately by the caller.)
_FAMILY_PREFIX_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Số hiệu shape components: <1..5 digits>/<4-digit year>/<suffix segments>.
_SOHIEU_NUM_RE = re.compile(r"\d{1,5}")
_SOHIEU_YEAR_RE = re.compile(r"\d{4}")
_SOHIEU_SUFFIX_SEG_RE = re.compile(r"[a-z0-9]{1,12}", re.IGNORECASE)

# A single digit anywhere in a segment.
_HAS_DIGIT_RE = re.compile(r"\d")

# Separators collapsed into "/" when normalizing an identifier *path*
# (whitespace, dot, comma, underscore, hyphen). Preserves segment boundaries.
_PATH_SEP_RE = re.compile(r"[\s.,_-]+")
_PATH_MULTI_SLASH_RE = re.compile(r"/+")


# ============================================================
# TRAILING NATURAL LANGUAGE
# ============================================================

_TRAILING_RE = re.compile(
    r"\s*(?:"
    r"la\s*ma\s*cua\s*van\s*ban\s*(?:nao|gi)?"
    r"|la\s*(?:gi|van\s*ban\s*(?:nao|gi)?|tai\s*lieu\s*(?:nao|gi)?)?"
    r"|noi\s*ve\s*gi"
    r"|la\s*cua\s*ai"
    r"|the\s*nao"
    r"|o\s*dau"
    r")\s*$",
    re.IGNORECASE,
)


_LOOKUP_TAIL_RE = re.compile(
    r"\s+la\s*(?:"
    r"ma\s*cua\s*van\s*ban"
    r"|van\s*ban"
    r"|gi"
    r"|tai\s*lieu"
    r")"
    r"(?:\s*(?:nao|gi))?\s*$",
    re.IGNORECASE,
)


_MAX_WHOLE_WORDS = 8


# ============================================================
# NORMALIZATION
# ============================================================

_SEP_RE = re.compile(r"[\s.,\-/]+")
_NORMALIZE_STRIP_RE = re.compile(r"[\s./\\\-]+")


# Explicit version suffix:
#   v1
#   v2
#   v1.0
#   v1.1
#   v2.
_VERSION_SUFFIX_EXPLICIT_RE = re.compile(
    r"(?:^|\s+)(v\d+(?:[._/-]\d+){0,4})\.?\s*$",
    re.IGNORECASE,
)


# Bare numeric version:
#   2.0
#   3.1
#
# Whitespace before the number is mandatory so that:
#   030/2022
# is NOT interpreted as a version.
_BARE_VERSION_SUFFIX_RAW_RE = re.compile(
    r"\s+(\d+(?:[._/-]\d+)+)\.?\s*$",
    re.IGNORECASE,
)


# Keyword-prefixed version (operates on diacritics-stripped text):
#   version 4            -> v4
#   version 4.0          -> v4.0
#   ver 4                -> v4
#   phien ban 4          -> v4   ("phiên bản 4")
#   ban 4                -> v4   ("bản 4")
#   phien ban 4.0        -> v4.0
# The keyword is explicit, so a bare trailing number here is safe (unlike
# _BARE_VERSION_SUFFIX_RAW_RE, this will not misread a số hiệu like 030/2022
# because that has no version keyword). The captured group is the number only;
# normalize_version() adds the "v" prefix.
_VERSION_KEYWORD_SUFFIX_RE = re.compile(
    r"\s+(?:version|ver|phien\s*ban|ban)\s*[:\-]?\s*"
    r"(\d+(?:[._/-]\d+){0,4})\.?\s*$",
    re.IGNORECASE,
)


# Kept for backward compatibility with callers/tests that may import it.
_VERSION_SUFFIX_RAW_RE = re.compile(
    r"(?:\s+|/|[-_.])v\d+(?:[._/-]\d+)*\.?\s*$",
    re.IGNORECASE,
)


_VERSION_SUFFIX_COLLAPSED_RE = re.compile(
    r"/v\d+(?:/\d+){0,4}$",
    re.IGNORECASE,
)

_TRAILING_MARKUP_NOISE_RE = re.compile(
    r"(?:\\?[_*~`])+\s*$"
)


@dataclass(frozen=True)
class DocCodeDetection:
    """Result of :func:`detect`."""

    is_doc_code: bool
    kind: str  # "code" | "sohieu" | "none"
    candidate: str | None
    collapsed: str | None
    reason: str
    version: str | None = None


# ============================================================
# BASIC HELPERS
# ============================================================

def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics and normalize Đ/Ð variants to D/d."""
    text = (
        text
        .replace("đ", "d")
        .replace("Đ", "D")
        .replace("ð", "d")
        .replace("Ð", "D")
    )

    nfkd = unicodedata.normalize("NFKD", text)

    return "".join(
        ch
        for ch in nfkd
        if not unicodedata.combining(ch)
    )


def collapse_separators(text: str) -> str:
    """
    Collapse separators into '/'.

    Examples:
        BM07.SP31/DN/TD
            -> BM07/SP31/DN/TD

        1299/2021/QĐ-TPB.IT
            -> 1299/2021/QD/TPB/IT
    """
    return _SEP_RE.sub("/", text).strip("/")


def normalize_version(value: str | None) -> str:
    """Normalize version to a stable form such as v1, v1.1, v2.0."""
    if not value:
        return ""

    version = str(value).strip().lower()
    version = version.rstrip(".,;:)]}>")

    if not version.startswith("v"):
        version = "v" + version

    version = (
        version
        .replace("_", ".")
        .replace("/", ".")
        .replace("-", ".")
    )

    # Collapse accidental repeated dots.
    version = re.sub(r"\.{2,}", ".", version)

    return version


def version_match(
    query_version: str | None,
    doc_version: str | None,
) -> bool:
    """True if the query's version matches the document's ``docVersion``.

    Both sides are normalized via :func:`normalize_version` so that the
    query form (``"v4.0"``) and the payload form (``"4.0"``) compare equal.

    Matching rules:
      - Exact: ``v4.0`` matches ``v4.0``.
      - Major-only fallback: when the query carries only a major component
        (e.g. ``v4``), it matches any doc version with the same major
        (``v4.0``, ``v4.1``). A query with a minor component (``v4.0``)
        does NOT match a different minor (``v4.1``).

    Returns False if either side is missing/empty.
    """
    q = normalize_version(query_version)
    d = normalize_version(doc_version)

    if not q or not d:
        return False

    if q == d:
        return True

    # Major-only fallback: query has no dot (e.g. "v4") -> match on major.
    if "." not in q:
        d_major = d.split(".", 1)[0]
        return q == d_major

    return False


def _extract_version_suffix(
    text: str,
) -> tuple[str, str | None]:
    
    value = text.strip()

    match = _VERSION_SUFFIX_EXPLICIT_RE.search(value)
    if match:
        return (
            value[:match.start(1)].strip(),
            normalize_version(match.group(1)),
        )

    # Keyword-prefixed form (version/ver/phien ban/ban + number). Tried before
    # the bare-number form: the explicit keyword lets us safely accept a bare
    # trailing integer (e.g. "version 4") without touching số-hiệu handling.
    match = _VERSION_KEYWORD_SUFFIX_RE.search(value)
    if match:
        # Strip from the start of the whole keyword match, not just the number,
        # so "BM09.QT01/QLCL version 4" -> base "BM09.QT01/QLCL".
        return (
            value[:match.start()].strip(),
            normalize_version(match.group(1)),
        )

    match = _BARE_VERSION_SUFFIX_RAW_RE.search(value)
    if match:
        return (
            value[:match.start(1)].strip(),
            normalize_version(match.group(1)),
        )

    return value, None


def _strip_version_suffix_raw(text: str) -> str:
    """Backward-compatible helper; now delegates to version extraction."""
    base, _version = _extract_version_suffix(text)
    return base


def _strip_version_suffix_collapsed(text: str) -> str:

    return _VERSION_SUFFIX_COLLAPSED_RE.sub("", text).strip("/")


# ============================================================
# COMPACT IDENTIFIER
# ============================================================

def _compact_identifier_to_code(
    candidate: str,
) -> str | None:
    source = strip_diacritics(candidate).strip()
    text = source.lower()

    if not text:
        return None

    # Preserve an already-valid single segment exactly. Without this guard,
    # compact parsing may over-split strings such as QDD17 into QD/D17.
    if re.fullmatch(_SEGMENT, text):
        return text

    # This helper is only for compact identifiers without separators.
    if re.search(r"[\s.,/\\\-]", text):
        return None

    if not any(ch.isdigit() for ch in text):
        return None

    matches: list[str] = []

    def dfs(
        start: int,
        parts: list[str],
    ) -> None:
        if start == len(text):
            code = "/".join(parts)

            if (
                parts
                and _has_digit_segment(code)
                and bool(_CODE_SHAPE_RE.fullmatch(code))
            ):
                matches.append(code)
            return

        max_len = min(len(text) - start, 7)

        for length in range(2, max_len + 1):
            seg = text[start:start + length]

            if not re.fullmatch(_SEGMENT, seg):
                continue

            dfs(
                start + length,
                parts + [seg],
            )

    dfs(0, [])

    if not matches:
        return None

    def seg_score(seg: str) -> int:
        if re.search(r"\d", seg):
            if re.search(r"\d$", seg):
                return 20 + len(seg)
            return 10 + len(seg)

        if len(seg) in (2, 3):
            return 1

        return -3

    def score(code: str) -> int:
        total = 0
        offset = 0

        for seg in code.split("/"):
            total += seg_score(seg)

            source_seg = source[
                offset:offset + len(seg)
            ]

            source_prefix_letters = "".join(
                ch
                for ch in source_seg[:-1]
                if ch.isalpha()
            )

            if (
                re.search(r"\d[a-z]$", seg)
                and source_seg[-1:].islower()
                and source_prefix_letters.isupper()
            ):
                total += 20

            offset += len(seg)

        return total

    return max(matches, key=score)


def _has_digit_segment(
    collapsed: str,
) -> bool:
    return any(
        re.search(r"\d", seg)
        for seg in collapsed.split("/")
        if seg
    )


def _expand_mixed_identifier(
    raw: str,
) -> str:
    parts: list[str] = []

    for token in re.split(
        r"[\s.,/\\\-]+",
        raw,
    ):
        if not token:
            continue

        parts.append(
            _compact_identifier_to_code(token)
            or strip_diacritics(token.lower())
        )

    return "/".join(parts)


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def _extract_identifier_from_sentence(
    text: str,
) -> str | None:
    has_lookup_tail = bool(
        _LOOKUP_TAIL_RE.search(text)
    )

    clean = _TRAILING_RE.sub(
        "",
        text,
    ).strip()

    if not clean:
        return None

    tokens = re.findall(
        r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*",
        clean,
    )

    if not tokens:
        return None

    best_candidate: str | None = None
    best_score: tuple[int, int, int] | None = None

    for start in range(len(tokens)):
        if not any(ch.isdigit() for ch in tokens[start]):
            continue

        for end in range(
            start + 1,
            min(start + 6, len(tokens)) + 1,
        ):
            if end < len(tokens) and not has_lookup_tail:
                continue

            next_token = tokens[end - 1]

            if (
                end > start + 1
                and not any(ch.isdigit() for ch in next_token)
                and len(next_token) > 4
            ):
                break

            raw = " ".join(tokens[start:end])
            raw, _version = _extract_version_suffix(raw)

            candidate = (
                _compact_identifier_to_code(raw)
                or _expand_mixed_identifier(raw)
            )

            if not candidate:
                continue

            candidate = collapse_separators(candidate)
            candidate = _strip_version_suffix_collapsed(candidate)

            if (
                _CODE_SHAPE_RE.fullmatch(candidate)
                and _has_digit_segment(candidate)
            ):
                score = (
                    candidate.count("/"),
                    len(candidate),
                    len(raw.replace(" ", "")),
                )

                if (
                    best_score is None
                    or score > best_score
                ):
                    best_candidate = candidate
                    best_score = score

    return best_candidate


# ============================================================
# PUBLIC NORMALIZATION
# ============================================================

def normalize_identifier(
    value: str | None,
) -> str:
    if not value:
        return ""

    return _NORMALIZE_STRIP_RE.sub(
        "",
        strip_diacritics(
            str(value).lower()
        ),
    )


# ============================================================
# DETECTOR
# ============================================================

def detect(
    query: str,
    code_shape_pattern: str = "",
    sohieu_shape_pattern: str = "",
) -> DocCodeDetection:
    """Detect whether *query* looks like a document code or số hiệu.

    Parameters
    ----------
    query:
        Raw user query string.
    code_shape_pattern:
        Optional regex string to override the built-in ``_CODE_SHAPE_RE``.
        Empty string (default) keeps the built-in pattern.
    sohieu_shape_pattern:
        Optional regex string to override the built-in ``_SO_HIEU_SHAPE_RE``.
        Empty string (default) keeps the built-in pattern.
    """
    # Compile per-request override patterns when provided (different from module-level defaults).
    _code_re   = re.compile(code_shape_pattern,   re.IGNORECASE) if code_shape_pattern   != _settings.default_code_shape_pattern   else _CODE_SHAPE_RE
    _sohieu_re = re.compile(sohieu_shape_pattern, re.IGNORECASE) if sohieu_shape_pattern != _settings.default_sohieu_shape_pattern else _SO_HIEU_SHAPE_RE

    clean_query = strip_diacritics(query).strip()

    # Remove trailing markup artifacts first.
    clean_query = _TRAILING_MARKUP_NOISE_RE.sub(
        "",
        clean_query,
    ).strip()

    norm = clean_query.lower()

    # Extract a trailing version from the query itself, before sentence
    # extraction potentially removes it from the identifier candidate.
    version_base, version = _extract_version_suffix(norm)

    # Use the version-free query only when it is otherwise the same query.
    # Natural-language tails are handled separately below.
    if version:
        norm_for_detection = version_base
    else:
        norm_for_detection = norm

    # --------------------------------------------------------
    # Step 1: extract embedded identifier from sentence
    # --------------------------------------------------------
    extracted = _extract_identifier_from_sentence(norm_for_detection)

    if extracted:
        candidate = extracted
        via = "embedded_id"

    else:
        # ----------------------------------------------------
        # Step 2: label-based lookup
        # ----------------------------------------------------
        match = _LABEL_RE.search(norm_for_detection)

        if match:
            candidate = norm_for_detection[match.end():].strip()
            via = "label_match"

        else:
            # ------------------------------------------------
            # Step 3: short whole-query detection
            # ------------------------------------------------
            if len(norm_for_detection.split()) > _MAX_WHOLE_WORDS:
                return DocCodeDetection(
                    is_doc_code=False,
                    kind="none",
                    candidate=None,
                    collapsed=None,
                    reason="too_many_words",
                    version=None,
                )

            candidate = norm_for_detection
            via = "whole_short"

    # --------------------------------------------------------
    # Remove natural-language tail
    # --------------------------------------------------------
    candidate = _TRAILING_RE.sub(
        "",
        candidate,
    ).strip()

    # Remove surrounding punctuation/noise.
    candidate = candidate.strip(
        " \t\r\n`*_.,:;()[]{}<>"
    )

    # --------------------------------------------------------
    # IMPORTANT: extract version BEFORE creating base id.
    # We do NOT throw it away.
    # --------------------------------------------------------
    candidate, detected_version = _extract_version_suffix(
        candidate
    )

    if detected_version:
        version = detected_version

    # --------------------------------------------------------
    # Separator normalization
    # --------------------------------------------------------
    collapsed = collapse_separators(candidate)

    # --------------------------------------------------------
    # Compact identifier normalization
    # --------------------------------------------------------
    compact_collapsed = _compact_identifier_to_code(candidate)

    if compact_collapsed:
        collapsed = compact_collapsed

    # Defensive removal in case a caller passes a pre-collapsed
    # identifier containing an explicit /vX suffix.
    collapsed = _strip_version_suffix_collapsed(
        collapsed
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------
    code_ok = bool(
        _code_re.fullmatch(collapsed)
    )

    sohieu_ok = bool(
        _sohieu_re.fullmatch(collapsed)
    )

    digit_ok = _has_digit_segment(collapsed)

    # --------------------------------------------------------
    # SỐ HIỆU FIRST
    # --------------------------------------------------------
    if sohieu_ok:
        return DocCodeDetection(
            is_doc_code=True,
            kind="sohieu",
            candidate=candidate,
            collapsed=collapsed,
            reason=f"{via}; sohieu_shape",
            version=version,
        )

    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------
    if code_ok and digit_ok:
        return DocCodeDetection(
            is_doc_code=True,
            kind="code",
            candidate=candidate,
            collapsed=collapsed,
            reason=f"{via}; code_shape",
            version=version,
        )

    # --------------------------------------------------------
    # NOT IDENTIFIER
    # --------------------------------------------------------
    return DocCodeDetection(
        is_doc_code=False,
        kind="none",
        candidate=candidate,
        collapsed=collapsed,
        reason=f"{via}; no_shape",
        version=version,
    )


# ============================================================
# CONVENIENCE
# ============================================================

def is_doc_code(
    query: str,
) -> bool:
    """True if query looks like CODE / SỐ HIỆU."""
    return detect(query).is_doc_code


def query_identifier(
    query: str,
) -> str:
    result = detect(query)

    if (
        not result.is_doc_code
        or not result.collapsed
    ):
        return ""

    return normalize_identifier(
        result.collapsed
    )


def query_version(
    query: str,
) -> str:
    result = detect(query)
    return result.version or ""


# ============================================================
# IDENTIFIER PATH NORMALIZATION / MATCHING
# ============================================================
# Pure identifier logic moved out of api/router.py (Refactor A).
# These functions have no HTTP / Qdrant dependency — they operate on
# plain identifier strings and paths only.


def normalize_identifier_path(value: str | None) -> str:
    """Normalize an identifier while preserving segment boundaries.

    Whitespace and punctuation commonly used as separators are collapsed
    to "/", keeping segment boundaries intact (unlike ``normalize_identifier``
    which strips all separators).
    """
    if not value:
        return ""

    text = strip_diacritics(str(value).lower()).strip()

    # Treat whitespace and punctuation commonly used as separators as
    # path separators. This intentionally preserves segment boundaries.
    text = text.replace("\\", "/")
    text = _PATH_SEP_RE.sub("/", text)
    text = _PATH_MULTI_SLASH_RE.sub("/", text)
    return text.strip("/")


def remove_pl_prefix(parts: list[str]) -> list[str]:
    """Remove an optional leading PLxx family prefix."""
    if parts and _PL_PREFIX_RE.fullmatch(parts[0]):
        return parts[1:]
    return parts


def identifier_family_match(
    query_path: str,
    stored_path: str,
) -> bool:
    """True if ``stored_path`` is in the same identifier family as ``query_path``."""
    qparts = normalize_identifier_path(query_path).split("/")
    sparts = normalize_identifier_path(stored_path).split("/")

    qparts = [p for p in qparts if p]
    sparts = [p for p in sparts if p]

    if not qparts or not sparts:
        return False

    qcore = remove_pl_prefix(qparts)
    score = remove_pl_prefix(sparts)

    if not qcore or not score:
        return False

    # 1. Same family after optional PLxx removal.
    if qcore == score:
        return True

    # 2. Same-length family segment match.
    # Example: BM -> BM01b while the remaining segments match exactly.
    if len(qcore) == len(score):
        family_used = False

        for qseg, sseg in zip(qcore, score):
            if qseg == sseg:
                continue

            if (
                _SEG_LETTERS_RE.fullmatch(qseg)
                and re.fullmatch(
                    rf"{re.escape(qseg)}\d+[a-z]?",
                    sseg,
                    re.IGNORECASE,
                )
            ):
                family_used = True
                continue

            family_used = False
            break

        if family_used:
            return True

    # 3. Stored code extends the query with one or two trailing segments.
    # This covers e.g. SP142/CN -> SP142/CN/TD and
    # SP12/CN -> SP12/CN/THE.
    if (
        len(score) > len(qcore)
        and len(score) - len(qcore) <= 2
        and score[: len(qcore)] == qcore
    ):
        return True

    # 4. One inserted code segment containing digits.
    # BM27/HD01/CN/TD -> BM27/HD01/SP13/CN/TD
    if len(score) == len(qcore) + 1:
        for index, extra in enumerate(score):
            if not _SEG_DIGIT_RE.fullmatch(extra):
                continue

            without_extra = (
                score[:index]
                + score[index + 1 :]
            )

            if without_extra == qcore:
                return True

    # 5. Final segment may be extended in the stored code.
    # PL02/HD05/QT -> HD05/QTNL
    if (
        len(score) == len(qcore)
        and len(qcore) >= 2
        and score[:-1] == qcore[:-1]
    ):
        qlast = qcore[-1]
        slast = score[-1]

        if (
            len(slast) > len(qlast)
            and slast.startswith(qlast)
        ):
            return True

    return False


def identifier_typo_variants_path(
    query_path: str,
) -> set[str]:
    """Generate conservative docMaVanBan typo variants.

    Only one adjacent duplicated alphabetic character is removed from a
    segment that contains at least one digit. Example:
        QDD17/CNTT -> QD17/CNTT
    """
    parts = normalize_identifier_path(query_path).split("/")
    parts = [p for p in parts if p]

    if not parts:
        return set()

    variants: set[str] = set()

    for index, part in enumerate(parts):
        if not _HAS_DIGIT_RE.search(part):
            continue

        for pos in range(len(part) - 1):
            if not part[pos].isalpha():
                continue
            if part[pos] != part[pos + 1]:
                continue

            candidate_part = (
                part[:pos]
                + part[pos + 1 :]
            )

            if not _SEG_OPT_DIGIT_RE.fullmatch(candidate_part):
                continue

            candidate_parts = list(parts)
            candidate_parts[index] = candidate_part
            variants.add("/".join(candidate_parts))

    return variants


def identifier_confusable_typo_variants_path(
    query_path: str,
) -> set[str]:
    """Generate O->Q variants on a digit-bearing segment.

    OCR/keyboard leftover: OT31/TD -> QT31/TD.
    Direction is O->Q only so queries that already start with Q
    (e.g. QT29/NV) are not expanded toward O.
    """
    parts = normalize_identifier_path(query_path).split("/")
    parts = [p for p in parts if p]

    if not parts:
        return set()

    variants: set[str] = set()

    for index, part in enumerate(parts):
        if not part.startswith("o"):
            continue
        if not _HAS_DIGIT_RE.search(part):
            continue

        candidate_part = "q" + part[1:]
        if not _SEG_OPT_DIGIT_RE.fullmatch(candidate_part):
            continue

        candidate_parts = list(parts)
        candidate_parts[index] = candidate_part
        variants.add("/".join(candidate_parts))

    return variants


def sohieu_prefix_match(
    query_value: str | None,
    stored_value: str | None,
) -> bool:
    """Conservative prefix match for partial document numbers.

    Example:
        047/2022/SP
            -> 047/2022/SP-TPB.CB

    This rule is restricted to a document-number shape:
        <1..5 digits>/<4-digit year>/<suffix>
    and never applies fuzzy matching to arbitrary text.
    """
    qparts = normalize_identifier_path(query_value).split("/")
    sparts = normalize_identifier_path(stored_value).split("/")

    qparts = [p for p in qparts if p]
    sparts = [p for p in sparts if p]

    if len(qparts) < 3 or len(sparts) < len(qparts):
        return False

    if not _SOHIEU_NUM_RE.fullmatch(qparts[0]):
        return False

    if not _SOHIEU_YEAR_RE.fullmatch(qparts[1]):
        return False

    suffix = qparts[2:]
    if not all(
        _SOHIEU_SUFFIX_SEG_RE.fullmatch(part)
        for part in suffix
    ):
        return False

    return sparts[: len(qparts)] == qparts


# ============================================================
# IDENTIFIER MATCH DECISION (pure)
# ============================================================


def identifier_match_kind(
    query_compact: str,
    query_path_norm: str,
    ma_raw: str | None,
    sohieu_raw: str | None,
    typo_compact: set[str],
) -> tuple[str | None, str, str]:
    """Decide how a stored identifier matches the query (pure, no I/O).

    Returns a tuple ``(matched_field, raw_value_field, match_kind)`` where:
      - ``matched_field`` is one of
        "docMaVanBan" | "docSoHieu" | "docSoHieu_prefix" |
        "docMaVanBan_family" | "docMaVanBan_typo" | None
      - ``raw_value_field`` tells the caller which raw payload value to attach
        ("ma" for docMaVanBan-derived, "sohieu" for docSoHieu-derived)
      - ``match_kind`` is the debug label.

    Priority mirrors the original ``_find_identifier_matches``:
      1. exact docMaVanBan
      2. exact docSoHieu
      3. conservative docSoHieu prefix
      4. docMaVanBan family
      5. conservative docMaVanBan typo
    """
    ma = normalize_identifier(ma_raw)
    sohieu = normalize_identifier(sohieu_raw)
    ma_path = normalize_identifier_path(ma_raw)

    # 1. Exact docMaVanBan / docSoHieu
    if query_compact and query_compact == ma:
        return "docMaVanBan", "ma", "IDENTIFIER_EXACT_MATCH"
    if query_compact and query_compact == sohieu:
        return "docSoHieu", "sohieu", "IDENTIFIER_EXACT_MATCH"

    # 2. Conservative prefix match on docSoHieu
    if sohieu_prefix_match(query_path_norm, sohieu_raw):
        return "docSoHieu_prefix", "sohieu", "IDENTIFIER_SOHIEU_PREFIX_MATCH"

    # 3. Family match on docMaVanBan only
    if ma_path and identifier_family_match(query_path_norm, ma_path):
        return "docMaVanBan_family", "ma", "IDENTIFIER_FAMILY_MATCH"

    # 4. Conservative typo match on docMaVanBan only
    if ma and ma in typo_compact:
        return "docMaVanBan_typo", "ma", "IDENTIFIER_TYPO_MATCH"

    return None, "", "IDENTIFIER_NO_MATCH"


# ============================================================
# IDENTIFIER RETRIEVAL QUERY BUILDERS
# ============================================================
# Build conservative extra sparse queries from a detected identifier.
# These are rewrites of the detected code/sohieu, never semantic expansions.


def identifier_family_retrieval_query(
    detection: DocCodeDetection | None,
) -> str | None:
    """Drop the family prefix for one relaxed sparse search.

    For codes like "BM27.HD01/CN/TD", drops "BM27" to get "HD01/CN/TD".
    For codes like "BM.SP31/CN/TD", drops "BM" to get "SP31/CN/TD".

    Family prefixes are typically 1-4 chars (letters and/or digits).
    """
    if detection is None:
        return None

    path = normalize_identifier_path(detection.collapsed)
    parts = path.split("/") if path else []
    if len(parts) < 2:
        return None

    # Drop first segment if it looks like a family prefix:
    # 1-4 characters of letters and/or digits (e.g., "bm", "bm27", "pl001").
    first_part = parts[0]
    if 1 <= len(first_part) <= 4 and _FAMILY_PREFIX_RE.fullmatch(first_part):
        return "/".join(parts[1:]) if len(parts) > 1 else None

    return None


def identifier_middle_segment_variants(path: str) -> list[str]:
    """Generate retrieval variants by removing one inserted alphanumeric segment.

    Mirrors ``identifier_family_match`` Rule 4 at *retrieval* time. If the stored
    docMaVanBan has one extra segment inserted (e.g., "sp13" in
    "bm27/hd01/sp13/cn/td"), the query "bm27/hd01/cn/td" will never match via
    BM25 because the token overlap is too low. By also searching for the path
    with each candidate middle segment removed we improve recall without
    loosening the match guard (Rule 4 still does the exact gating at match time).

    A segment is a candidate for removal only if it matches ``_SEG_DIGIT``
    (e.g. sp13, bm27, hd01) AND it is not the first or last segment.
    """
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:  # need at least 3 parts to have a removable middle
        return []

    variants: list[str] = []

    for i in range(1, len(parts) - 1):  # skip first and last
        if _SEG_DIGIT_END_RE.fullmatch(parts[i]):
            variant = "/".join(parts[:i] + parts[i + 1:])
            if variant not in variants:
                variants.append(variant)

    return variants


def sohieu_extra_retrieval_queries(normalized_path: str) -> list[str]:
    """Generate fallback extras for sohieu (num/year/...) queries.

    For "04/2025/qd/tpb/cthdqt", generates:
      - Short form: "04/2025"
      - Suffix only: "qd/tpb/cthdqt"
    """
    parts = normalized_path.split("/")
    extras: list[str] = []

    # Check if this looks like sohieu: num/year/...
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        # Short form: num/year
        extras.append(f"{parts[0]}/{parts[1]}")

        # Suffix only: qd/tpb/cthdqt (without num/year)
        if len(parts) > 2:
            extras.append("/".join(parts[2:]))

    return extras


def identifier_extra_retrieval_queries(
    detection: DocCodeDetection | None,
    original_query: str,
) -> list[str]:
    """Extra sparse queries for leftover identifier cases.

    Only identifier routing uses this. Each extra query is a conservative
    rewrite of the detected code/sohieu, not a semantic expansion.
    """
    if detection is None or not detection.is_doc_code or not detection.collapsed:
        return []

    path = normalize_identifier_path(detection.collapsed)
    extras: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        normalized = normalize_identifier_path(value)
        if not normalized or normalized in seen:
            return
        if (
            normalized == normalize_identifier_path(original_query)
            and value == original_query
        ):
            return
        seen.add(normalized)
        extras.append(normalized)

    original_stripped = original_query.strip()
    original_ascii = strip_diacritics(original_stripped).lower()
    if path and original_ascii != original_stripped.lower():
        add(path)
    elif path and normalize_identifier_path(original_stripped) != path:
        add(path)
    add(identifier_family_retrieval_query(detection))

    # sohieu-specific extras (short form + suffix only)
    if detection.kind == "sohieu":
        for variant in sohieu_extra_retrieval_queries(path):
            add(variant)

    for variant in sorted(identifier_typo_variants_path(path)):
        add(variant)

    for variant in sorted(identifier_confusable_typo_variants_path(path)):
        add(variant)

    if detection.kind == "code":
        parts = [p for p in path.split("/") if p]
        if (
            len(parts) >= 4
            and _SEG_LETTERS_RE.fullmatch(parts[-1])
        ):
            add("/".join(parts[:-1]))
        # Add variants with one middle alphanumeric segment removed.
        # Mirrors identifier_family_match Rule 4 at retrieval time.
        for variant in identifier_middle_segment_variants(path):
            add(variant)

    # Drop last segment for variant/prefix matching (cases #35, #39).
    # For "qd20/td", query "qd20" to find all QD20/* variants.
    # For "qt01/dn/nv", query "qt01/dn" to find all QT01/DN/* variants.
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        base_prefix = "/".join(parts[:-1])
        add(base_prefix)

        # For 3+ segments, also add first-segment-only for broader match.
        if len(parts) >= 3:
            add(parts[0])

    # Suffix-only for 2+ segment identifiers (e.g. "qt01/dn/ch" -> "ch").
    if len(parts) >= 2:
        add(parts[-1])

    return extras
