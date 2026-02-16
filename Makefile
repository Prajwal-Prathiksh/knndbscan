.PHONY: install build test clean help dev lint format check

# Default target
help:
	@echo "Available targets:"
	@echo "  install    Install the package in editable mode"
	@echo "  install-gpu Install the package with GPU support in editable mode"
	@echo "  build      Build distribution packages (wheel and sdist)"
	@echo "  test       Run unit tests with pytest"
	@echo "  test-examples Run example scripts"
	@echo "  lint       Run linting tools"
	@echo "  format     Format code with ruff"
	@echo "  check      Run all checks (lint + test)"
	@echo "  clean      Clean build artifacts"
	@echo "  help       Show this help message"

# Sync development dependencies and install package in editable mode
install: dev
	uv sync --all-groups
	uv run pip install -e .

# Sync with GPU support and build C++ extension
install-gpu:
	uv sync --all-groups --extra gpu
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
	uv run --group lint clang-format --dry-run --Werror src/*.cpp include/*.h test/*.cpp

# Format code
format:
	uv run --group lint ruff format .
	uv run --group lint clang-format -i src/*.cpp include/*.h test/*.cpp

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