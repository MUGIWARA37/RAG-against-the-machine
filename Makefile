ifneq ($(wildcard $(HOME)/goinfre/.),)
  export UV_CACHE_DIR := $(HOME)/goinfre/.cache/uv
  export HF_HOME := $(HOME)/goinfre/.cache/huggingface
endif

.PHONY: install run debug clean lint lint-strict

install:
	@if [ -d "$$HOME/goinfre" ]; then \
		mkdir -p "$$HOME/goinfre/rag-venv" "$$HOME/goinfre/.cache/uv" "$$HOME/goinfre/.cache/huggingface" && \
		ln -sfn "$$HOME/goinfre/rag-venv" .venv; \
	fi
	uv sync --python 3.14

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict --ignore-missing-imports
