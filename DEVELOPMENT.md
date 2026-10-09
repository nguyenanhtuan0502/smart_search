# Development Guide

## Local testing

```bash
uv sync --group dev
make check
make run
```

Open `http://127.0.0.1:8080/docs`. Uvicorn runs the API; `uv` only manages dependencies and `.venv`.

## Docker

```bash
docker build -t serve-search-service:local .
docker run --rm -p 8080:8080 serve-search-service:local
```

The image starts `uvicorn serve_search_service.main:app` on port `8080`.

## Add search capabilities

Add a provider-neutral interface under `domain/`, a use case under `application/`, and provider SDK code under `adapters/`. Register a public route in `api/router.py` only when the capability needs an HTTP API. Wire the selected adapter in `main.py`.

No retriever, reranker, vector database, or LLM adapter is included in this base code.
