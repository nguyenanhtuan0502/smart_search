"""In-process embedders for query-side hybrid search.

`DenseEmbedder` calls an OpenAI-compatible ``/embeddings`` endpoint (vLLM
gateway); `Bm25SparseEmbedder` computes corpus-free BM25 sparse vectors
locally. Both are ported from serve-data-processing-service, with the dense
embedder adapted so the base URL and model can be overridden per request
(falling back to the settings defaults).
"""

import asyncio
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx

from ..domain.embedding import EmbedderUnavailableError, SparseVector, Vector, format_query
from ..logging import TraceIdAdapter, get_logger

_PREVIEW_CHARS = 120
_PREVIEW_ITEMS = 5
_PREVIEW_DIMS = 4


def _preview_text(text: str) -> str:
    flattened = text.replace("\n", "\\n")
    if len(flattened) > _PREVIEW_CHARS:
        return f"{flattened[:_PREVIEW_CHARS]}…(+{len(flattened) - _PREVIEW_CHARS} chars)"
    return flattened


def _preview_inputs(texts: list[str]) -> str:
    shown = [_preview_text(text) for text in texts[:_PREVIEW_ITEMS]]
    if len(texts) > _PREVIEW_ITEMS:
        shown.append(f"…(+{len(texts) - _PREVIEW_ITEMS} more)")
    return "[" + " | ".join(shown) + "]"


def _preview_vector(vector: Vector) -> str:
    head = ", ".join(f"{component:.4f}" for component in vector[:_PREVIEW_DIMS])
    if len(vector) > _PREVIEW_DIMS:
        return f"[{head}, …(+{len(vector) - _PREVIEW_DIMS})]"
    return f"[{head}]"


def _mask_authorization(header: str | None) -> str:
    """Redact a bearer token, keeping only the scheme and a short fingerprint."""
    if not header:
        return "<none>"
    scheme, _, token = header.partition(" ")
    if not token:
        return "<set>"
    if len(token) <= 8:
        return f"{scheme} {token[:2]}…"
    return f"{scheme} {token[:6]}…{token[-4:]}"


class DenseEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
        max_retries: int = 0,
        retry_backoff_seconds: float = 1.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._default_base_url = base_url.rstrip("/")
        self._default_model = model
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers, trust_env=False
        )
        self._logger: TraceIdAdapter = get_logger(__name__)

    def _resolve_url(self, base_url: str | None) -> str:
        return (base_url or self._default_base_url).rstrip("/") + "/v1/embeddings"

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> Sequence[Vector]:
        return await self._embed(
            list(texts),
            url=self._resolve_url(base_url),
            model=model or self._default_model,
            api_key=api_key,
            input_type="document",
            instruction=None,
        )

    async def embed_queries(
        self,
        queries: Sequence[str],
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        instruction: str | None = None,
    ) -> Sequence[Vector]:
        texts = (
            [format_query(instruction, query) for query in queries]
            if instruction
            else list(queries)
        )
        return await self._embed(
            texts,
            url=self._resolve_url(base_url),
            model=model or self._default_model,
            api_key=api_key,
            input_type="query",
            instruction=instruction,
        )

    async def _embed(
        self,
        texts: list[str],
        *,
        url: str,
        model: str,
        api_key: str | None,
        input_type: str,
        instruction: str | None,
    ) -> list[Vector]:
        # Per-request key overrides the client's default Authorization header.
        override_header = f"Bearer {api_key}" if api_key else None
        auth = _mask_authorization(
            override_header or self._client.headers.get("Authorization")
        )
        instruction_note = f", instruction={instruction!r}" if instruction else ""
        self._logger.info(
            "Embedding model %s request → %s (Authorization: %s): "
            "%d input(s), input_type=%s%s %s",
            model,
            url,
            auth,
            len(texts),
            input_type,
            instruction_note,
            _preview_inputs(texts),
        )
        payload = await self._post_embeddings(url, model, texts, api_key=api_key)

        try:
            rows = sorted(payload["data"], key=lambda row: row["index"])
            vectors = [tuple(row["embedding"]) for row in rows]
        except (KeyError, TypeError) as exc:
            self._logger.warning(
                "Embedding model %s returned an unexpected payload: %s", model, exc
            )
            raise EmbedderUnavailableError(
                "Malformed embedding response from the gateway."
            ) from exc

        dimensions = len(vectors[0]) if vectors else 0
        self._logger.info(
            "Embedding model %s response: embedded %d/%d input(s) as %s, dims=%d, "
            "inputs=%s, sample=%s",
            model,
            len(vectors),
            len(texts),
            input_type,
            dimensions,
            _preview_inputs(texts),
            _preview_vector(vectors[0]) if vectors else "[]",
        )

        if len(vectors) != len(texts):
            raise EmbedderUnavailableError(
                f"Model embedding returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        return vectors

    async def _post_embeddings(
        self, url: str, model: str, texts: list[str], *, api_key: str | None = None
    ) -> Any:
        body = {"model": model, "input": texts, "encoding_format": "float"}
        loggable_body = {**body, "input": _preview_inputs(texts)}
        self._logger.info("Embedding model %s request body: %s", model, loggable_body)
        # httpx merges per-request headers over the client defaults.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        attempt = 0
        while True:
            try:
                response = await self._client.post(url, json=body, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                if attempt < self._max_retries:
                    sleep_seconds = self._retry_backoff_seconds * (2**attempt)
                    self._logger.warning(
                        "Embedding model %s transient error, retry %d/%d after %.1fs: %s",
                        model,
                        attempt + 1,
                        self._max_retries,
                        sleep_seconds,
                        exc,
                    )
                    await asyncio.sleep(sleep_seconds)
                    attempt += 1
                    continue
                self._logger.warning("Embedding model %s exceeded retries: %s", model, exc)
                raise EmbedderUnavailableError(str(exc)) from exc
            except (httpx.HTTPError, ValueError) as exc:
                self._logger.warning("Embedding model %s call failed: %s", model, exc)
                raise EmbedderUnavailableError(str(exc)) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_token(token: str) -> int:
    """Stable, process-independent 32-bit token id (Qdrant accepts u32 indices)."""
    return zlib.crc32(token.encode("utf-8"))


class Bm25SparseEmbedder:
    def __init__(
        self,
        *,
        model: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
        avg_len: float = 256.0,
    ) -> None:
        self._model = model
        self._k1 = k1
        self._b = b
        self._avg_len = avg_len
        self._logger: TraceIdAdapter = get_logger(__name__)

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        # Offload the CPU-bound work to a thread so it never blocks the event loop
        # and can run in parallel with the dense embedder's network call.
        vectors = await asyncio.to_thread(self._embed_documents, list(texts))
        self._logger.info(
            "BM25 %s embedded %d document(s) as sparse vectors", self._model, len(vectors)
        )
        return vectors

    async def embed_queries(
        self, queries: Sequence[str], *, instruction: str | None = None
    ) -> Sequence[SparseVector]:
        # `instruction` is a dense-model concern; it does not apply to BM25.
        vectors = await asyncio.to_thread(self._embed_queries, list(queries))
        self._logger.info(
            "BM25 %s embedded %d query/queries as sparse vectors", self._model, len(vectors)
        )
        return vectors

    def _embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed_document(text) for text in texts]

    def _embed_queries(self, queries: list[str]) -> list[SparseVector]:
        return [self._embed_query(query) for query in queries]

    def _embed_document(self, text: str) -> SparseVector:
        tokens = _tokenize(text)
        if not tokens:
            return SparseVector(indices=(), values=())
        counts = Counter(tokens)
        doc_len = len(tokens)
        denom_len = self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
        # Sum weights on the same index so hash collisions add rather than clobber.
        weights: dict[int, float] = {}
        for token, tf in counts.items():
            weight = tf * (self._k1 + 1) / (tf + denom_len)
            index = _hash_token(token)
            weights[index] = weights.get(index, 0.0) + weight
        return self._to_sparse(weights)

    def _embed_query(self, text: str) -> SparseVector:
        tokens = _tokenize(text)
        # One weight of 1.0 per unique token; IDF is applied by the vector store.
        weights = {_hash_token(token): 1.0 for token in set(tokens)}
        return self._to_sparse(weights)

    @staticmethod
    def _to_sparse(weights: dict[int, float]) -> SparseVector:
        if not weights:
            return SparseVector(indices=(), values=())
        indices = tuple(sorted(weights))
        values = tuple(weights[index] for index in indices)
        return SparseVector(indices=indices, values=values)
