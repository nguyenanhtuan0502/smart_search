"""Public HTTP response models for search."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProblemDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    code: str
    trace_id: str | None = None
    retryable: bool = False


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    passage_id: str
    text: str
    score: float
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[SearchResult]
    trace_id: str | None = None
