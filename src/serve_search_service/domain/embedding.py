"""Embedding domain models and rules (provider-neutral).

Ported from serve-data-processing-service so the search service can embed a
user query (dense + sparse) in-process instead of calling out over HTTP.
"""

from dataclasses import dataclass

Vector = tuple[float, ...]


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]


class EmbedderUnavailableError(Exception):
    """The model embedding is unreachable or returned an unusable response."""


@dataclass(frozen=True)
class EmbeddedText:
    index: int
    text: str
    input_type: str
    vector: Vector
    sparse: SparseVector


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    dimensions: int
    normalized: bool
    sparse_model: str
    items: tuple[EmbeddedText, ...]


def format_query(instruction: str, query: str) -> str:
    """Qwen3-Embedding query template, applied on the query side only.

    Matches the official prompt in Qwen3-Embedding's
    config_sentence_transformers.json exactly: the query is appended directly
    after 'Query:' with NO trailing space. Documents are embedded raw (the
    model card defines the document prompt as an empty string).
    """
    return f"Instruct: {instruction}\nQuery:{query}"
