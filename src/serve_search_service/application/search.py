"""Search use case; concrete retrieval is supplied by an adapter."""

from collections.abc import Sequence

from ..domain.ports import Retriever
from ..domain.search import SearchHit, SearchQuery


class SearchService:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        return await self._retriever.search(query)
