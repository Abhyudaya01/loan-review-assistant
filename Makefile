.PHONY: lint type test

lint:
	uv run --locked ruff check .

type:
	uv run --locked mypy src

test:
	uv run --locked pytest -q
