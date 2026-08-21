"""Embedding adapter — HTTP client to call embedding API (serve-data-processing-service).

Converts query text into dense + sparse vectors for hybrid search.
Uses POST /v1/query-embeddings (synchronous, query-side embedding).
"""

from typing import Any

import httpx

from ..logging import TraceIdAdapter, get_logger
from ..settings import ServiceSettings

_logger: TraceIdAdapter = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding API call fails.
    
    Attributes:
        detail: Human-readable error message
        status: HTTP status code (502 if connection error, actual status if HTTP error)
        retryable: Whether the error is retryable
    """
    
    def __init__(
        self,
        detail: str,
        status: int = 502,
        retryable: bool = True,
    ) -> None:
        self.detail = detail
        self.status = status
        self.retryable = retryable
        super().__init__(detail)


class EmbeddingClient:
    """Async HTTP client for the embedding service.

    Calls POST /v1/query-embeddings to get dense + sparse vectors from query text.
    This is the query-side embedding (search), NOT the ingest-side (/v1/embeddings).
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

        body: dict[str, Any] = {
            "query": text,
            "model": self._model,
        }

        _logger.info(
            "Embedding request: url=%s/v1/query-embeddings, model=%s, text_len=%d",
            self._base_url, self._model, len(text),
        )

        try:
            resp = await self._client.post("/v1/query-embeddings", json=body)
        except httpx.TimeoutException as e:
            _logger.error("Embedding API timeout: %s", e)
            raise EmbeddingError(
                detail=f"Embedding API timeout after 60s",
                status=504,
                retryable=True,
            ) from e
        except httpx.ConnectError as e:
            _logger.error("Embedding API connection failed: %s", e)
            raise EmbeddingError(
                detail=f"Cannot connect to Embedding API at {self._base_url}",
                status=502,
                retryable=True,
            ) from e
        except httpx.HTTPError as e:
            _logger.error("Embedding API HTTP error: %s", e)
            raise EmbeddingError(
                detail=f"Embedding API HTTP error: {str(e)[:100]}",
                status=502,
                retryable=True,
            ) from e

        # Check HTTP status
        if resp.status_code != 200:
            detail = resp.text[:200] if resp.text else "No response body"
            _logger.error(
                "Embedding API returned status=%d, body=%s",
                resp.status_code, detail,
            )
            # Determine retryability based on status code
            retryable = resp.status_code >= 500  # 5xx = retryable, 4xx = not retryable
            raise EmbeddingError(
                detail=f"Embedding API returned HTTP {resp.status_code}: {detail}",
                status=resp.status_code,
                retryable=retryable,
            )

        # Parse response
        try:
            data = resp.json()
        except ValueError as e:
            _logger.error("Embedding API response is not valid JSON: %s", e)
            raise EmbeddingError(
                detail=f"Embedding API returned invalid JSON",
                status=502,
                retryable=True,
            ) from e

        # Extract vectors from QueryEmbeddingResponse format:
        # { "model": str, "dimensions": int, "dense": [...], "sparse": {...} | null, "trace_id": str }
        try:
            dense = data["dense"]
            if (
                not isinstance(dense, list)
                or not dense
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in dense)
            ):
                raise ValueError(
                    f"dense vector must be a non-empty list of numbers, "
                    f"got type={type(dense)}, len={len(dense) if isinstance(dense, list) else 'N/A'}"
                )

            sparse_embedding: dict[str, Any] | None = None
            sparse_raw = data.get("sparse")
            if sparse_raw is not None:
                if not isinstance(sparse_raw, dict):
                    raise ValueError(
                        f"sparse must be a dict or null, got {type(sparse_raw)}"
                    )
                indices = sparse_raw.get("indices")
                values = sparse_raw.get("values")
                if not isinstance(indices, list) or not isinstance(values, list):
                    raise ValueError(
                        "sparse indices and values must be lists"
                    )
                if len(indices) != len(values):
                    raise ValueError(
                        f"sparse indices/values length mismatch: "
                        f"{len(indices)} != {len(values)}"
                    )
                if indices:
                    sparse_embedding = {"indices": indices, "values": values}

        except (KeyError, ValueError, TypeError) as e:
            _logger.error(
                "Unexpected embedding response format: %s. Response keys: %s",
                e, list(data.keys()) if isinstance(data, dict) else type(data),
            )
            raise EmbeddingError(
                detail=f"Unexpected embedding response format",
                status=502,
                retryable=False,
            ) from e

        sparse_count = len(sparse_embedding["indices"]) if sparse_embedding else 0
        _logger.info(
            "Embedding response OK: model=%s, dense_dim=%d, sparse_tokens=%d",
            data.get("model", "unknown"),
            len(dense),
            sparse_count,
        )

        return {
            "embedding": dense,
            "sparse_embedding": sparse_embedding,
        }
