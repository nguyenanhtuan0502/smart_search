"""Client-facing DTOs for search API."""

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..settings import ServiceSettings

# Load once at import so field defaults come from env (SERVE_SEARCH_*), while
# remaining overridable per-request. String-typed fields keep string defaults.
_settings = ServiceSettings()


# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────


class FilterCondition(BaseModel):


    model_config = ConfigDict(extra="allow")

    key: str = Field(min_length=1, description="Payload field name to filter on")
    value: str | list[str] = Field(
        min_length=1,
        description="Value or list of values to match (OR within a condition)",
    )
    match_type: Literal["value", "text"] = Field(
        default="value",
        description=(
            "Filter match type: 'value' for exact match (default), "
            "'text' for text/contains match. "
            "Used by must_not_conditions."
        ),
    )


class UdfField(BaseModel):
    """A user-defined field in the MAS payload format."""

    model_config = ConfigDict(extra="allow")

    key: str = Field(min_length=1, description="User-defined field name")
    value: Any = Field(description="User-defined field value")


# ─────────────────────────────────────────────
# Search Options
# ─────────────────────────────────────────────


class SearchOptions(BaseModel):
    """Search tuning parameters."""

    model_config = ConfigDict(extra="allow")

    vector_weight: str = Field(
        default_factory=lambda: str(_settings.default_vector_weight),
        description="Dense vector weight in RRF fusion (0.0 to 1.0)",
    )
    keyword_weight: str = Field(
        default_factory=lambda: str(_settings.default_keyword_weight),
        description="Sparse (BM25) weight in RRF fusion (0.0 to 1.0)",
    )
    rrf_k: str = Field(
        default_factory=lambda: str(_settings.default_rrf_k),
        description="RRF constant k (1 to 200)",
    )
    candidate_multiplier: str = Field(
        default_factory=lambda: str(_settings.default_candidate_multiplier),
        description="Multiply limit by this to get candidate pool size (2 to 20)",
    )
    fusion_mode: Literal["client", "server_rrf", "batch_client_rrf"] = Field(
        default_factory=lambda: _settings.default_fusion_mode,  # type: ignore[arg-type]
        description="Fusion strategy for hybrid search",
    )
    query_instruction: str = Field(
        default_factory=lambda: _settings.default_query_instruction,
        description=(
            "Qwen3-Embedding instruction applied to the dense QUERY side only "
            "(documents are embedded raw at ingest). Formatted as "
            "'Instruct: {instruction}\\nQuery: {query}'. Empty string disables it "
            "(query embedded raw). Ignored for identifier/sparse-only queries."
        ),
    )
    auto_route: str = Field(
        default_factory=lambda: str(_settings.default_auto_route).lower(),
        description="Dynamic routing for document-code queries ('true'/'false')",
    )
    code_shape_pattern: str = Field(
        default_factory=lambda: _settings.default_code_shape_pattern,
        description=(
            "Regex pattern for doc-code shape detection (re.IGNORECASE). "
            "Overrides env SERVE_SEARCH_DEFAULT_CODE_SHAPE_PATTERN and settings default. "
            "Default: r'^[a-z]{1,4}\\d{0,3}[a-z]?(?:/[a-z]{1,4}\\d{0,3}[a-z]?){1,5}$'"
        ),
    )
    sohieu_shape_pattern: str = Field(
        default_factory=lambda: _settings.default_sohieu_shape_pattern,
        description=(
            "Regex pattern for sohieu shape detection (re.IGNORECASE). "
            "Overrides env SERVE_SEARCH_DEFAULT_SOHIEU_SHAPE_PATTERN and settings default. "
            "Default: r'^\\d{1,5}/\\d{4}(?:/[a-z0-9]{1,12}){0,6}$'"
        ),
    )
    identifier_bonus: str = Field(
        default_factory=lambda: str(_settings.default_identifier_bonus),
        description=(
            "Score added to a document (after rerank) when its docMaVanBan/docSoHieu "
            "exactly matches the detected query identifier ."
        ),
    )
    version_match: str = Field(
        default_factory=lambda: str(_settings.default_version_match).lower(),
        description=(
            "Master switch for the version-matching flow ('true'/'false'). When "
            "'false', the query version is still extracted but no version bonus is "
            "applied (version has no effect on ranking)."
        ),
    )
    version_bonus: str = Field(
        default_factory=lambda: str(_settings.default_version_bonus),
        description=(
            "Extra score added (on top of identifier_bonus, after rerank) when the "
            "query carries a version (e.g. 'v4.0') AND the document's docVersion "
            "matches it. Prioritizes the exact-version document within the same "
            "docMaVanBan family. Only applied to identifier queries that have a version. "
            "Ignored when version_match is 'false'."
        ),
    )
    title_search: str = Field(
        default_factory=lambda: str(_settings.default_title_search).lower(),
        description=(
            "Also search title chunks (chunk 0) and return documents whose title "
            "matches the query in a separate 'title_result' field ('true'/'false'). "
            "'result' is unaffected. Empty for document-code queries."
        ),
    )
    title_top_k: str = Field(
        default_factory=lambda: str(_settings.default_title_top_k),
        description=(
            "Title chunks retrieved per branch (dense/sparse) for title_result "
            "(1 to 100)"
        ),
    )
    title_rrf_k: str = Field(
        default_factory=lambda: str(_settings.default_title_rrf_k),
        description="RRF constant k for ranking title chunks (1 to 200)",
    )
    title_min_coverage: str = Field(
        default_factory=lambda: str(_settings.default_title_min_coverage),
        description=(
            "Minimum IDF-weighted share of query tokens a title must contain to "
            "appear in title_result (0.0 to 1.0)"
        ),
    )
    citation: str = Field(
        default_factory=lambda: str(_settings.default_citation).lower(),
        description=(
            "Attach highlighted snippets to each document result ('true'/'false'). "
            "Built from the chunks the search already returned; ranking is unaffected."
        ),
    )
    score_threshold: str | None = Field(
        default_factory=lambda: (
            None if _settings.default_score_threshold is None
            else str(_settings.default_score_threshold)
        ),
        description=(
            "Drop documents whose final rerank score is below this threshold "
            "(0.0 to 1.0). Applies only when rerank is used. None disables filtering."
        ),
    )
    filter_conditions: list["FilterCondition"] | None = Field(
        default=None,
        description=(
            "Additional filter conditions to merge with top-level 'filter' field. "
            "Both are combined with AND logic. "
            "Example: [{key: 'group_ids', value: ['hr']}]"
        ),
    )
    must_not_conditions: list["FilterCondition"] | None = Field(
        default=None,
        description=(
            "Exclusion filter conditions (must_not). Documents matching ANY of these "
            "conditions will be excluded from results. "
            "For match_type='text' (substring), values must be tokenized according to "
            "the collection's text index tokenizer (e.g. 'vv' not 'v/v' if tokenizer='word'). "
            "Example: [{key: 'docTenVB', value: 'vv', match_type: 'text'}]"
        ),
    )

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold_string(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"score_threshold must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {value}")
        return v

    @field_validator("vector_weight", "keyword_weight")
    @classmethod
    def validate_weight_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError:
            raise ValueError(f"Weight must be a valid number string, got '{v}'")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {value}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"rrf_k must be a valid integer string, got '{v}'")
        if not (1 <= value <= 200):
            raise ValueError(f"rrf_k must be between 1 and 200, got {value}")
        return v

    @field_validator("candidate_multiplier")
    @classmethod
    def validate_candidate_multiplier_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"candidate_multiplier must be a valid integer string, got '{v}'")
        if not (2 <= value <= 20):
            raise ValueError(f"candidate_multiplier must be between 2 and 20, got {value}")
        return v

    @field_validator("auto_route")
    @classmethod
    def validate_auto_route_string(cls, v: str) -> str:
        if v.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                f"auto_route must be a boolean string (true/false), got '{v}'"
            )
        return v

    @field_validator("code_shape_pattern", "sohieu_shape_pattern")
    @classmethod
    def validate_regex_pattern(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {v!r}: {exc}") from exc
        return v

    def auto_route_enabled(self) -> bool:
        """Parse auto_route string to bool for internal use."""
        return self.auto_route.strip().lower() in {"true", "1", "yes"}

    @field_validator("version_match")
    @classmethod
    def validate_version_match_string(cls, v: str) -> str:
        if v.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                f"version_match must be a boolean string (true/false), got '{v}'"
            )
        return v

    def version_match_enabled(self) -> bool:
        """Parse version_match string to bool for internal use."""
        return self.version_match.strip().lower() in {"true", "1", "yes"}

    @field_validator("citation")
    @classmethod
    def validate_citation_string(cls, v: str) -> str:
        if v.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                f"citation must be a boolean string (true/false), got '{v}'"
            )
        return v

    def citation_enabled(self) -> bool:
        return self.citation.strip().lower() in {"true", "1", "yes"}

    @field_validator("identifier_bonus")
    @classmethod
    def validate_identifier_bonus_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"identifier_bonus must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 5.0):
            raise ValueError(f"identifier_bonus must be between 0.0 and 5.0, got {value}")
        return v

    @field_validator("version_bonus")
    @classmethod
    def validate_version_bonus_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"version_bonus must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 5.0):
            raise ValueError(f"version_bonus must be between 0.0 and 5.0, got {value}")
        return v

    @field_validator("title_search")
    @classmethod
    def validate_title_search_string(cls, v: str) -> str:
        if v.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                f"title_search must be a boolean string (true/false), got '{v}'"
            )
        return v

    def title_search_enabled(self) -> bool:
        """Parse title_search string to bool for internal use."""
        return self.title_search.strip().lower() in {"true", "1", "yes"}

    @field_validator("title_min_coverage")
    @classmethod
    def validate_title_min_coverage_string(cls, v: str) -> str:
        try:
            value = float(v)
        except ValueError as err:
            raise ValueError(
                f"title_min_coverage must be a valid number string, got '{v}'"
            ) from err
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"title_min_coverage must be between 0.0 and 1.0, got {value}")
        return v

    @field_validator("title_top_k", "title_rrf_k")
    @classmethod
    def validate_title_int_string(cls, v: str, info: ValidationInfo) -> str:
        high = 100 if info.field_name == "title_top_k" else 200
        try:
            value = int(v)
        except ValueError as err:
            raise ValueError(
                f"{info.field_name} must be a valid integer string, got '{v}'"
            ) from err
        if not (1 <= value <= high):
            raise ValueError(f"{info.field_name} must be between 1 and {high}, got {value}")
        return v

    @model_validator(mode="after")
    def validate_weights_not_both_zero(self) -> "SearchOptions":
        vector_w = float(self.vector_weight)
        keyword_w = float(self.keyword_weight)
        if vector_w == 0 and keyword_w == 0:
            raise ValueError(
                "At least one of vector_weight or keyword_weight must be > 0."
            )
        return self


# ─────────────────────────────────────────────
# Rerank Options
# ─────────────────────────────────────────────


class RerankOptions(BaseModel):
    """Document-level reranking configuration."""

    model_config = ConfigDict(extra="allow")

    strategy: Literal["maxp", "top_m_mean", "hit_count", "rrf", "hybrid"] = Field(
        default_factory=lambda: _settings.default_rerank_strategy,  # type: ignore[arg-type]
        description="Rerank strategy: maxp | top_m_mean | hit_count | rrf | hybrid",
    )
    weights: str | None = Field(
        default=None,
        description=(
            "Hybrid strategy weights as string: '(w1, w2, w3, w4)' for (maxp, top_m_mean, hit_count, rrf). "
            "Must sum to 1.0. Only used when strategy='hybrid'."
        ),
    )
    mean_m: str = Field(
        default_factory=lambda: str(_settings.default_rerank_mean_m),
        description="Top M chunks for top_m_mean strategy"
    )
    rrf_k: str = Field(
        default_factory=lambda: str(_settings.default_rerank_rrf_k),
        description="RRF k constant for rrf/hybrid strategy"
    )
    document_id_field: str = Field(
        default="document_id",
        description="Payload field name that identifies the parent document",
    )
    top_chunks: str = Field(
        default_factory=lambda: str(_settings.default_rerank_top_chunks),
        description="Number of top chunks used internally for scoring",
    )

    @staticmethod
    def _parse_weights(weights: str) -> tuple[float, ...]:
        """Parse weights string to tuple.
        
        Accepts formats like:
        - "(0.25, 0.25, 0.25, 0.25)"
        - "0.25, 0.25, 0.25, 0.25"
        """
        weights_str = weights.strip()
        if weights_str.startswith("(") and weights_str.endswith(")"):
            weights_str = weights_str[1:-1]
        return tuple(float(w.strip()) for w in weights_str.split(","))

    @field_validator("mean_m")
    @classmethod
    def validate_mean_m_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"mean_m must be a valid integer string, got '{v}'")
        if value < 1:
            raise ValueError(f"mean_m must be >= 1, got {value}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"rrf_k must be a valid integer string, got '{v}'")
        if value <= 0:
            raise ValueError(f"rrf_k must be > 0, got {value}")
        return v

    @field_validator("top_chunks")
    @classmethod
    def validate_top_chunks_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"top_chunks must be a valid integer string, got '{v}'")
        if not (1 <= value <= 20):
            raise ValueError(f"top_chunks must be between 1 and 20, got {value}")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "RerankOptions":
        if self.strategy != "hybrid" and self.weights is not None:
            raise ValueError("weights is only supported when strategy='hybrid'.")
        if self.weights is not None:
            try:
                weight_values = self._parse_weights(self.weights)
            except ValueError:
                raise ValueError(f"weights must be comma-separated floats, got '{self.weights}'")
            
            if len(weight_values) != 4:
                raise ValueError(f"weights must have exactly 4 values, got {len(weight_values)}")
            
            if any(w < 0 for w in weight_values):
                raise ValueError("All weights must be non-negative.")
            if abs(sum(weight_values) - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {sum(weight_values):.6f}."
                )
        return self

    def get_weights_tuple(self) -> tuple[float, ...] | None:
        """Get weights as tuple for service consumption."""
        if not self.weights:
            return None
        return self._parse_weights(self.weights)


_CITATION_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "idf_tau": (0.0, 20.0, False),
    "top_keywords": (1, 20, True),
    "min_cohesion": (0.0, 1.0, False),
    "window_chars": (100, 5000, True),
    "companion_ratio": (0.0, 1.0, False),
    "max_chunks": (1, 20, True),
    "semantic_gate": (0.0, 1.0, False),
    "max_phrase_tokens": (1, 10, True),
    "nested_ratio": (0.5, 1.0, False),
    "secondary_tau": (0.0, 20.0, False),
}


class CitationOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    idf_tau: str = Field(
        default_factory=lambda: str(_settings.citation_idf_tau),
        description="Minimum IDF for a term or phrase to be highlighted (0 to 20)",
    )
    top_keywords: str = Field(
        default_factory=lambda: str(_settings.citation_top_keywords),
        description="Maximum number of highlighted keywords (1 to 20)",
    )
    min_cohesion: str = Field(
        default_factory=lambda: str(_settings.citation_min_cohesion),
        description=(
            "Keep a phrase only if it occurs in at least this share of the chunks "
            "containing its rarest word (0.0 to 1.0)"
        ),
    )
    window_chars: str = Field(
        default_factory=lambda: str(_settings.citation_window_chars),
        description="Snippet length in characters around the keywords (100 to 5000)",
    )
    companion_ratio: str = Field(
        default_factory=lambda: str(_settings.citation_companion_ratio),
        description="Extra chunk needs this fraction of the anchor's cosine score (0.0 to 1.0)",
    )
    max_chunks: str = Field(
        default_factory=lambda: str(_settings.citation_max_chunks),
        description="Maximum cited chunks per document (1 to 20)",
    )
    semantic_gate: str = Field(
        default_factory=lambda: str(_settings.citation_semantic_gate),
        description="Cosine score for a keyword-free anchor to count as 'semantic' (0.0 to 1.0)",
    )
    max_phrase_tokens: str = Field(
        default_factory=lambda: str(_settings.citation_max_phrase_tokens),
        description="Longest query phrase considered, in syllables (1 to 10)",
    )
    nested_ratio: str = Field(
        default_factory=lambda: str(_settings.citation_nested_ratio),
        description=(
            "A phrase yields to a one-syllable-longer phrase found in this share "
            "of its chunks (0.5 to 1.0)"
        ),
    )
    secondary_tau: str = Field(
        default_factory=lambda: str(_settings.citation_secondary_tau),
        description=(
            "Minimum IDF for other multi-syllable query words to be highlighted inside "
            "the chosen snippet; a value >= idf_tau turns this off (0 to 20)"
        ),
    )

    @field_validator(*_CITATION_BOUNDS)
    @classmethod
    def validate_bounds(cls, v: str, info: ValidationInfo) -> str:
        name = info.field_name or ""
        low, high, integer = _CITATION_BOUNDS[name]
        try:
            value = int(v) if integer else float(v)
        except ValueError as err:
            kind = "an integer" if integer else "a number"
            raise ValueError(f"{name} must be {kind} string, got '{v}'") from err
        if not (low <= value <= high):
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")
        return v


# ─────────────────────────────────────────────
# Search Request
# ─────────────────────────────────────────────

class HybridSearchRequest(BaseModel):

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    query: str = Field(min_length=1, description="Query text (question / search keywords)")
    collection: str = Field(
        default_factory=lambda: _settings.default_collection,
        min_length=1,
        description="Qdrant collection name (default from SERVE_SEARCH_DEFAULT_COLLECTION)",
    )
    limit: str = Field(default="10", description="Max number of documents to return (1-100)")

    embedding_url: str | None = Field(
        default=None,
        min_length=1,
        description="Override the dense embedding gateway base URL (default: from env)",
    )
    embedding_model: str | None = Field(
        default=None,
        min_length=1,
        description="Override the dense embedding model (default: from env)",
    )
    embedding_api_key: str | None = Field(
        default=None,
        min_length=1,
        description="Override the API key sent to the embedding gateway (default: from env)",
    )

    category_parent_id: str | None = Field(
        default=None,
        alias="categoryParentId",
        description=(
            "Pre-filter: only return documents whose payload 'categoryParentId' "
            "matches this value exactly. Mapped to a Qdrant must-filter on the "
            "'categoryParentId' field. Documents missing the field are naturally "
            "excluded by Qdrant. Omit/null to disable."
        ),
    )

    search_options: SearchOptions | None = Field(
        default=None,
        description="Search tuning parameters (weights, thresholds, etc.)",
        alias="searchOptions",
    )
    filter: list[FilterCondition] | None = Field(
        default=None,
        description=(
            'Payload filter conditions (AND logic between conditions). '
            'Example: [{"key": "group_ids", "value": ["hr"]}]'
        ),
    )
    rerank: RerankOptions | None = Field(
        default=None,
        description="Document-level reranking options",
    )
    citation_options: CitationOptions | None = Field(
        default=None,
        description="Citation tuning; used only when search_options.citation is true",
    )
    udf: list[UdfField] | None = Field(
        default=None,
        description="User-defined fields in MAS format: [{key, value}, ...]",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be blank.")
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit_string(cls, v: str) -> str:
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"limit must be a valid integer string, got '{v}'")
        if not (1 <= value <= 100):
            raise ValueError(f"limit must be between 1 and 100, got {value}")
        return v

    @model_validator(mode="after")
    def validate_candidate_pool(self) -> "HybridSearchRequest":
        # The candidate pool actually retrieved is
        #   min(limit * candidate_multiplier, max_candidate_limit).
        # Reject (instead of silently capping) when the requested pool exceeds
        # the hard cap, so the caller is not misled about how many candidates
        # are used. Both fields validate individually first, so int() is safe.
        opts = self.search_options or SearchOptions()
        pool = int(self.limit) * int(opts.candidate_multiplier)
        if pool > _settings.max_candidate_limit:
            raise ValueError(
                f"limit * candidate_multiplier = {pool} exceeds the candidate pool "
                f"cap of {_settings.max_candidate_limit}; reduce limit or candidate_multiplier."
            )
        return self


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────


class CitationHighlight(BaseModel):
    start: int
    end: int


class CitationItem(BaseModel):
    chunk_id: str
    chunk_index: int | None = None
    text: str = Field(description="Plaintext snippet, truncated to the window size")
    highlights: list[CitationHighlight] = Field(default_factory=list)
    match_count: int = Field(
        default=0,
        description="Keyword matches in the whole chunk, including any cut from the window",
    )
    truncated: bool = False
    is_anchor: bool = Field(
        default=False, description="The chunk the document surfaced on"
    )


class DocumentResultItem(BaseModel):
    """A document-level result."""

    document_id: str
    rank_document: int = Field(description="Document rank (1-based)")
    citation_tier: str | None = Field(
        default=None,
        description=(
            "How the snippets were chosen: 'literal' (keyword matches), 'semantic' "
            "(strong dense match, no keywords), 'whole' (chunk shown as-is), or "
            "'title_only' (no citable chunk; the citation is the full document title). "
            "None when citation is off."
        ),
    )
    citations: list[CitationItem] = Field(default_factory=list)


class TitleResultItem(BaseModel):
    """A document whose title (chunk 0) matches the query."""

    document_id: str
    rank_document: int = Field(description="Rank within title_result (1-based)")
    title_score: float = Field(
        description="Title rank strength (RRF of dense + sparse, best = 1) × coverage"
    )
    title: str | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Document title (docTenVB); only when citation is on",
    )
    title_highlights: list[CitationHighlight] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Spans of title matching the query; only when citation is on",
    )


class HybridSearchResponse(BaseModel):

    status: int = 200
    result: list[DocumentResultItem] = Field(default_factory=list)
    title_result: list[TitleResultItem] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description=(
            "Title branch documents, present only when search_options.title_search "
            "is 'true'"
        ),
    )
