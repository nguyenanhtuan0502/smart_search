"""Embedding adapter — HTTP client to call embedding API (serve-data-processing-service).

Converts query text into dense + sparse vectors for hybrid search.
"""

from typing import Any

import httpx

from ..logging import TraceIdAdapter, get_logger
from ..settings import ServiceSettings

_logger: TraceIdAdapter = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding API call fails."""


class EmbeddingClient:
    """Async HTTP client for the embedding service.

    Calls POST /v1/embeddings to get dense + sparse vectors from text.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self._base_url = settings.embedding_url.rstrip("/")
        self._model = settings.embedding_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json"},
            timeout=60,
            trust_env=False,
        )
        _logger.info(
            "EmbeddingClient initialized: base_url=%s, model=%s",
            self._base_url, self._model,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call on app shutdown)."""
        await self._client.aclose()
        _logger.info("EmbeddingClient closed")

    async def embed_query(self, text: str) -> dict[str, Any]:
        """Embed a query text and return dense + sparse vectors.

        Args:
            text: The query string to embed.

        Returns:
            {
                "embedding": [float...],
                "sparse_embedding": {"indices": [...], "values": [...]}
            }

        Raises:
            EmbeddingError: If the embedding API call fails.
        """
        body = {
            "input": text,
            "model": self._model,
            "input_type": "query",
        }

        _logger.info(
            "Embedding request: url=%s/v1/embeddings, model=%s, text_len=%d",
            self._base_url, self._model, len(text),
        )

        try:
            resp = await self._client.post("/v1/embeddings", json=body)
        except httpx.TimeoutException as e:
            _logger.error("Embedding API timeout: %s", e)
            raise EmbeddingError(f"Embedding API timeout after 60s: {e}") from e
        except httpx.ConnectError as e:
            _logger.error("Embedding API connection failed: %s", e)
            raise EmbeddingError(
                f"Cannot connect to Embedding API at {self._base_url}: {e}"
            ) from e
        except httpx.HTTPError as e:
            _logger.error("Embedding API HTTP error: %s", e)
            raise EmbeddingError(f"Embedding API HTTP error: {e}") from e

        # Check HTTP status
        if resp.status_code != 200:
            detail = resp.text[:500] if resp.text else "No response body"
            _logger.error(
                "Embedding API returned status=%d, body=%s",
                resp.status_code, detail,
            )
            raise EmbeddingError(
                f"Embedding API returned HTTP {resp.status_code}: {detail}"
            )

        # Parse response
        try:
            data = resp.json()
        except Exception as e:
            _logger.error("Embedding API response is not valid JSON: %s", e)
            raise EmbeddingError(f"Embedding API returned invalid JSON: {e}") from e

        # Extract vectors
        try:
            items = data["data"]
            if not items:
                raise ValueError("data array is empty")
            first = items[0]
            embedding = first["embedding"]
            sparse_embedding = first["sparse_embedding"]

            if not isinstance(embedding, list) or len(embedding) == 0:
                raise ValueError(
                    f"embedding is invalid: type={type(embedding)}, "
                    f"len={len(embedding) if isinstance(embedding, list) else 'N/A'}"
                )

            if not isinstance(sparse_embedding, dict):
                raise ValueError(f"sparse_embedding is not a dict: {type(sparse_embedding)}")

        except (KeyError, IndexError, ValueError, TypeError) as e:
            _logger.error(
                "Unexpected embedding response format: %s. Response keys: %s",
                e, list(data.keys()) if isinstance(data, dict) else type(data),
            )
            raise EmbeddingError(
                f"Unexpected embedding response format: {e}"
            ) from e

        _logger.info(
            "Embedding response OK: dense_dim=%d, sparse_tokens=%d",
            len(embedding),
            len(sparse_embedding.get("indices", [])),
        )

        return {
            "embedding": embedding,
            "sparse_embedding": sparse_embedding,
        }
