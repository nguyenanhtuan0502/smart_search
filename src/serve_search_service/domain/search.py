"""Provider-neutral search domain models."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchQuery:
    tenant_id: str
    text: str
    limit: int
    filters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    passage_id: str
    text: str
    score: float
    source_uri: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
