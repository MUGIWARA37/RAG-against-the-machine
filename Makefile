.PHONY: install run debug clean lint lint-strict

install:
	@if [ -d "$$HOME/goinfre" ]; then \
		mkdir -p "$$HOME/goinfre/rag-venv" "$$HOME/goinfre/.cache/uv" "$$HOME/goinfre/.cache/huggingface" && \
		ln -sfn "$$HOME/goinfre/rag-venv" .venv && \
		UV_CACHE_DIR="$$HOME/goinfre/.cache/uv" uv sync --python 3.14; \
	else \
		uv sync --python 3.14; \
	fi

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/processed/query_cache.json

lint:
	uv run flake8 src/
	uv run mypy src/ --ignore-missing-imports

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict --ignore-missing-imports
