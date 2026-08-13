"""Qdrant adapter — HTTP calls to Qdrant for search operations."""

from typing import Any

import httpx

from ..logging import TraceIdAdapter, get_logger
from ..settings import ServiceSettings

_logger: TraceIdAdapter = get_logger(__name__)


class QdrantClient:
    """Async HTTP client for Qdrant search operations.

    Maintains a single httpx.AsyncClient instance for connection reuse.
    Must be closed via aclose() on application shutdown.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self._base_url = settings.qdrant_url
        headers = {"Content-Type": "application/json"}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=15,
            trust_env=False,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call on app shutdown)."""
        await self._client.aclose()

    async def dense_search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 50,
        score_threshold: float | None = None,
        payload_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> dict[str, Any]:
        """Dense vector similarity search."""
        body: dict[str, Any] = {
            "vector": {"name": "dense", "vector": vector},
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": False,
        }
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        if payload_filter:
            body["filter"] = payload_filter

        url = f"/collections/{collection}/points/search"

        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
        except httpx.TimeoutException:
            return {"error": True, "status": 504, "detail": "Qdrant timeout (dense)"}
        except httpx.HTTPError as e:
            return {"error": True, "status": 503, "detail": f"Qdrant error (dense): {e}"}

        return resp.json()

    async def sparse_search(
        self,
        collection: str,
        sparse_vector: dict[str, Any],
        limit: int = 50,
        payload_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> dict[str, Any]:
        """Sparse (lexical/BM25) vector search."""
        body: dict[str, Any] = {
            "vector": {"name": "sparse", "vector": sparse_vector},
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": False,
        }
        if payload_filter:
            body["filter"] = payload_filter

        url = f"/collections/{collection}/points/search"

        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
        except httpx.TimeoutException:
            return {"error": True, "status": 504, "detail": "Qdrant timeout (sparse)"}
        except httpx.HTTPError as e:
            return {"error": True, "status": 503, "detail": f"Qdrant error (sparse): {e}"}

        return resp.json()
