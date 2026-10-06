"""Cross-encoder reranker client (Qwen3-Reranker served by vLLM ``/v1/rerank``).

Scores (query, chunk) pairs so the search flow can reorder fused chunk
candidates before they are grouped into documents. Large candidate lists are
split into batches that are sent concurrently.
"""

import asyncio
import time
from collections.abc import Sequence

import httpx

from ..logging import TraceIdAdapter, get_logger

_RETRYABLE_STATUS = {502, 503, 504}

# Official Qwen3-Reranker prompt (model card / vLLM example). The model judges
# "yes"/"no" relevance, so query and document must be wrapped exactly like this
# unless the serving side already applies the template.
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankerUnavailableError(Exception):
    """The reranker is unreachable or returned an unusable response."""


class Reranker:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        batch_size: int = 32,
        qwen3_template: bool = True,
    ) -> None:
        self._default_base_url = base_url.rstrip("/")
        self._default_model = model
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._batch_size = batch_size
        self._qwen3_template = qwen3_template
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers, trust_env=False
        )
        self._logger: TraceIdAdapter = get_logger(__name__)

    def _format(
        self, query: str, documents: Sequence[str], instruction: str
    ) -> tuple[str, list[str]]:
        if not self._qwen3_template:
            query_text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
            return query_text, list(documents)
        query_text = f"{_QWEN3_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        return query_text, [f"<Document>: {doc}{_QWEN3_SUFFIX}" for doc in documents]

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> list[float]:
        """Return one relevance score per document, in the input order."""
        url = (base_url or self._default_base_url).rstrip("/") + "/v1/rerank"
        model = model or self._default_model
        query_text, texts = self._format(query, documents, instruction)
        batches = [
            (start, texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        self._logger.info(
            "Reranker %s → %s: %d document(s) in %d batch(es) of <=%d, qwen3_template=%s",
            model,
            url,
            len(texts),
            len(batches),
            self._batch_size,
            self._qwen3_template,
        )
        results = await asyncio.gather(
            *(
                self._post(
                    url, model, query_text, batch, api_key=api_key, label=f"{i}/{len(batches)}"
                )
                for i, (_, batch) in enumerate(batches, 1)
            )
        )
        scores: list[float] = []
        for batch_scores in results:
            scores.extend(batch_scores)
        return scores

    async def _post(
        self,
        url: str,
        model: str,
        query: str,
        documents: list[str],
        *,
        api_key: str | None,
        label: str,
    ) -> list[float]:
        body = {"model": model, "query": query, "documents": documents}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            error: Exception | None = None
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except httpx.RequestError as exc:
                error = exc
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    break
                error = RerankerUnavailableError(f"HTTP {response.status_code}")
            if attempt == self._max_retries:
                self._logger.warning(
                    "Reranker batch %s failed after %d attempt(s): POST %s - %r",
                    label,
                    attempt + 1,
                    url,
                    error,
                )
                raise RerankerUnavailableError(str(error)) from error
            self._logger.warning(
                "Reranker batch %s retry %d/%d after %.1fs: POST %s - %r",
                label,
                attempt + 1,
                self._max_retries,
                self._retry_backoff_seconds,
                url,
                error,
            )
            await asyncio.sleep(self._retry_backoff_seconds)

        elapsed = time.perf_counter() - started
        if response.status_code >= 400:
            self._logger.warning(
                "Reranker batch %s HTTP %d: POST %s in %.2fs, detail=%s",
                label,
                response.status_code,
                url,
                elapsed,
                response.text[:500],
            )
            raise RerankerUnavailableError(f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            rows = response.json()["results"]
            by_index = {int(row["index"]): float(row["relevance_score"]) for row in rows}
            scores = [by_index[i] for i in range(len(documents))]
        except (ValueError, KeyError, TypeError) as exc:
            raise RerankerUnavailableError(f"Malformed rerank response: {exc}") from exc
        self._logger.info(
            "Reranker batch %s: HTTP %d for POST %s in %.2fs, %d document(s)",
            label,
            response.status_code,
            url,
            elapsed,
            len(documents),
        )
        return scores

    async def aclose(self) -> None:
        await self._client.aclose()
