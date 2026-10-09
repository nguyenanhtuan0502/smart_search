.DEFAULT_GOAL := check

run:
	.venv/bin/python -m uvicorn serve_search_service.main:app --reload --port 8080

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck
