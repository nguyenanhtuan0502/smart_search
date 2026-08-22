"""Port interfaces for the search service."""

from collections.abc import Sequence
from typing import Protocol

from .embedding import SparseVector, Vector


class TenantResolver(Protocol):
    def resolve(self, headers: dict[str, str]) -> str: ...


class EmbedderPort(Protocol):
    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> Sequence[Vector]: ...

    async def embed_queries(
        self,
        queries: Sequence[str],
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        instruction: str | None = None,
    ) -> Sequence[Vector]: ...


class SparseEmbedderPort(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> Sequence[SparseVector]: ...

    async def embed_queries(
        self, queries: Sequence[str], *, instruction: str | None = None
    ) -> Sequence[SparseVector]: ...
