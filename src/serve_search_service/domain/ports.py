"""Provider-neutral search boundaries."""

from collections.abc import Sequence
from typing import Protocol

from .search import SearchHit, SearchQuery


class TenantResolver(Protocol):
    def resolve(self, headers: dict[str, str]) -> str: ...


class Retriever(Protocol):
    async def search(self, query: SearchQuery) -> Sequence[SearchHit]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: SearchQuery, candidates: Sequence[SearchHit]
    ) -> Sequence[SearchHit]: ...
