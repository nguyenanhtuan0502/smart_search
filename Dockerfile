# Replace image tags with security-approved immutable digests before production use.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.22-python3.13-bookworm
ARG PYTHON_IMAGE=python:3.13-slim-bookworm

FROM ${UV_IMAGE} AS builder
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE}
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/app/.venv/bin:$PATH
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app && mkdir /app/logs && chown app:app /app/logs
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn", "serve_search_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
