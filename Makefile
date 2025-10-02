.PHONY: install build test clean help dev lint format check

# Default target
help:
	@echo "Available targets:"
	@echo "  install    Install the package in editable mode"
	@echo "  build      Build distribution packages (wheel and sdist)"
	@echo "  dev        Sync development dependencies"
	@echo "  test       Run unit tests with pytest"
	@echo "  test-examples Run example scripts"
	@echo "  lint       Run linting tools"
	@echo "  format     Format code with ruff"
	@echo "  check      Run all checks (lint + test)"
	@echo "  clean      Clean build artifacts"
	@echo "  help       Show this help message"

# Sync all dependency groups
dev:
	uv sync --all-groups

# Sync development dependencies and install package in editable mode
install: dev
	uv run pip install -e .

# Build distribution packages
build:
	uv build

# Run all tests
test:
	uv run --group test pytest

# Run example scripts
test-examples:
	uv run python examples/test_binding.py
	uv run python examples/benchmark_threading.py

# Run linting tools
lint:
	uv run --group lint ruff check .
	uv run --group lint mypy knndbscan/ || true

# Format code
format:
	uv run --group lint ruff format .

# Run all checks
check: lint test

# Clean build artifacts
clean:
	rm -f .coverage
	rm -f knndbscan/*.so
	rm -f knndbscan/_core*.so
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/