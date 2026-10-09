# Serve Search Service

A single FastAPI application that provides a provider-neutral search API for RAG workloads.

## Structure

```text
src/serve_search_service/
├── main.py          # App setup and adapter wiring
├── api/             # Public routes, request models, and errors
├── application/     # Search use cases
├── domain/          # Search models and provider-neutral interfaces
├── adapters/        # Selected retriever, reranker, or storage integrations
├── logging.py       # JSON logging and trace context
├── settings.py      # Runtime settings
└── telemetry.py     # OpenTelemetry setup
```

## API

- `POST /v1/search`
- `GET /v1`
- Health probes under `/health/*`

Search returns `501 search_not_configured` until a retriever adapter is wired in `main.py`.

## Run locally

```bash
uv sync --group dev
make run
```

API docs: `http://127.0.0.1:8080/docs`

See [DEVELOPMENT.md](DEVELOPMENT.md) for local testing, Docker, and adding a search capability.
