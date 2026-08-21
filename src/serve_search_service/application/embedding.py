"""Query-side embedding service (dense + sparse) for hybrid search.

Embeds a user query in-process: the dense model gets the Qwen3 query
instruction (when supplied) and the sparse model uses IDF-only weighting, so a
caller can run a hybrid search against Qdrant. The dense ``model`` and backend
``base_url`` default to settings but may be overridden per call.
"""

import asyncio

from ..domain.embedding import EmbeddedText, EmbeddingResult
from ..domain.ports import EmbedderPort, SparseEmbedderPort
from ..logging import TraceIdAdapter, get_logger


class EmbeddingService:
    def __init__(
        self,
        *,
        dense: EmbedderPort,
        sparse: SparseEmbedderPort,
        default_model: str,
        default_base_url: str,
        sparse_model: str,
        fallback_dimensions: int,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._default_model = default_model
        self._default_base_url = default_base_url
        self._sparse_model = sparse_model
        self._fallback_dimensions = fallback_dimensions
        self._logger: TraceIdAdapter = get_logger(__name__)

    async def create_embeddings(
        self,
        *,
        texts: list[str],
        input_type: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        instruction: str | None = None,
    ) -> EmbeddingResult:
        resolved_model = model or self._default_model
        resolved_url = base_url or self._default_base_url

        if input_type == "query":
            vectors, sparse_vectors = await asyncio.gather(
                self._dense.embed_queries(
                    texts,
                    base_url=resolved_url,
                    model=resolved_model,
                    api_key=api_key,
                    instruction=instruction,
                ),
                self._sparse.embed_queries(texts, instruction=instruction),
            )
        else:
            vectors, sparse_vectors = await asyncio.gather(
                self._dense.embed_documents(
                    texts, base_url=resolved_url, model=resolved_model, api_key=api_key
                ),
                self._sparse.embed_documents(texts),
            )

        items = tuple(
            EmbeddedText(
                index=index,
                text=text,
                input_type=input_type,
                vector=vector,
                sparse=sparse,
            )
            for index, (text, vector, sparse) in enumerate(
                zip(texts, vectors, sparse_vectors, strict=True)
            )
        )
        dimensions = len(items[0].vector) if items else self._fallback_dimensions
        self._logger.info(
            "Embedded %d text(s) as %s with %s (+sparse %s)",
            len(items),
            input_type,
            resolved_model,
            self._sparse_model,
        )
        return EmbeddingResult(
            model=resolved_model,
            dimensions=dimensions,
            normalized=True,
            sparse_model=self._sparse_model,
            items=items,
        )
