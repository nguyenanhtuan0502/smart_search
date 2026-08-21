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
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key

        _logger.info(
            "Initializing QdrantClient: base_url=%s, has_api_key=%s",
            self._base_url, bool(settings.qdrant_api_key),
        )

        # Structured timeout configuration instead of magic number
        timeout = httpx.Timeout(
            connect=5.0,    # Connection timeout
            read=30.0,      # Read timeout
            write=30.0,     # Write timeout
            pool=5.0,       # Pool timeout
        )

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        _logger.info("QdrantClient initialized successfully")

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call on app shutdown)."""
        await self._client.aclose()

    async def dense_search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 50,
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
        if payload_filter is not None:
            body["filter"] = payload_filter

        url = f"/collections/{collection}/points/search"
        _logger.info("Dense search: url=%s%s, vector_len=%d", self._base_url, url, len(vector))

        try:
            resp = await self._client.post(url, json=body)
            _logger.info("Dense search response: status=%d", resp.status_code)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            _logger.error("Dense search timeout for collection=%s: %s", collection, e)
            return {"error": True, "status": 504, "detail": f"Qdrant timeout (dense): {e}"}
        except httpx.HTTPStatusError as e:
            # HTTP error with status code (4xx, 5xx)
            _logger.error(
                "Dense search HTTP error for collection=%s: status=%d, detail=%s",
                collection, e.response.status_code, e.response.text[:200],
            )
            return {
                "error": True,
                "status": e.response.status_code,
                "detail": f"Qdrant HTTP error (dense): {e.response.status_code}",
            }
        except httpx.RequestError as e:
            # Network/connection errors (connection failed, DNS resolution, etc.)
            _logger.error("Dense search network error for collection=%s: %s", collection, e)
            return {
                "error": True,
                "status": 503,
                "detail": f"Qdrant unavailable (dense): {str(e)[:100]}",
            }

        result: dict[str, Any] = resp.json()
        return result

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
        if payload_filter is not None:
            body["filter"] = payload_filter

        url = f"/collections/{collection}/points/search"
        indices_len = len(sparse_vector.get("indices", []))
        _logger.info("Sparse search: url=%s%s, indices_len=%d", self._base_url, url, indices_len)

        try:
            resp = await self._client.post(url, json=body)
            _logger.info("Sparse search response: status=%d", resp.status_code)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            _logger.error("Sparse search timeout for collection=%s: %s", collection, e)
            return {"error": True, "status": 504, "detail": f"Qdrant timeout (sparse): {e}"}
        except httpx.HTTPStatusError as e:
            # HTTP error with status code (4xx, 5xx)
            _logger.error(
                "Sparse search HTTP error for collection=%s: status=%d, detail=%s",
                collection, e.response.status_code, e.response.text[:200],
            )
            return {
                "error": True,
                "status": e.response.status_code,
                "detail": f"Qdrant HTTP error (sparse): {e.response.status_code}",
            }
        except httpx.RequestError as e:
            # Network/connection errors (connection failed, DNS resolution, etc.)
            _logger.error("Sparse search network error for collection=%s: %s", collection, e)
            return {
                "error": True,
                "status": 503,
                "detail": f"Qdrant unavailable (sparse): {str(e)[:100]}",
            }

        result: dict[str, Any] = resp.json()
        return result
