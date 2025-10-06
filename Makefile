.PHONY: help build check clean dev format install install-debug lint publish publish-test test test-examples

# Default target
help:
	@echo "Available targets:"
	@echo "  build      Build distribution packages (wheel and sdist)"
	@echo "  check      Run all checks (lint + test)"
	@echo "  clean      Clean build artifacts"
	@echo "  dev        Sync development dependencies"
	@echo "  format     Format code with ruff"
	@echo "  install    Install the package in editable mode"
	@echo "  install-debug Install package with optimizations and debug symbols (-O3 -g)"
	@echo "  lint       Run linting tools"
	@echo "  publish    Publish to PyPI"
	@echo "  publish-test Publish to Test PyPI"
	@echo "  test       Run unit tests with pytest"
	@echo "  test-examples Run example scripts"
	@echo "  help       Show this help message"

# Build distribution packages
build:
	uv build

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
	rm -rf __pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/

# Sync all dependency groups
dev:
	uv sync --all-groups

# Format code
format:
	uv run --group lint ruff format .

# Sync development dependencies and install package in editable mode
install: dev
	DEBUG=0 uv run pip install -e .

# Install package with optimizations and debug symbols
install-debug: dev
	DEBUG=1 uv run pip install -e . --force-reinstall --no-deps

# Run linting tools
lint:
	uv run --group lint ruff check .
	uv run --group lint mypy knndbscan/ || true

# Publish to Test PyPI
publish-test:
	@echo "Building package..."
	uv build
	@echo "Repairing wheels for manylinux compatibility..."
	@if command -v auditwheel >/dev/null 2>&1; then \
		echo "Using auditwheel to repair wheels..."; \
		for wheel in dist/*.whl; do \
			if [[ "$$wheel" == *"linux_x86_64.whl" ]]; then \
				echo "Repairing $$wheel"; \
				auditwheel repair "$$wheel" --plat manylinux_2_39_x86_64 -w dist/; \
				rm "$$wheel"; \
			fi; \
		done; \
	else \
		echo "auditwheel not found. Installing..."; \
		uv run pip install auditwheel; \
		for wheel in dist/*.whl; do \
			if [[ "$$wheel" == *"linux_x86_64.whl" ]]; then \
				echo "Repairing $$wheel"; \
				uv run auditwheel repair "$$wheel" --plat manylinux_2_39_x86_64 -w dist/; \
				rm "$$wheel"; \
			fi; \
		done; \
	fi
	@echo "Publishing to Test PyPI..."
	@if [ ! -f "$$HOME/.pypirc" ] && [ -z "$$TWINE_PASSWORD" ]; then \
		echo "ERROR: Authentication required. Either:"; \
		echo "  1. Create ~/.pypirc file with your API token, OR"; \
		echo "  2. Set environment variable: export TWINE_PASSWORD='pypi-...'"; \
		echo ""; \
		echo "Get API token from: https://test.pypi.org/manage/account/#api-tokens"; \
		exit 1; \
	fi
	uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI
publish:
	@echo "Building package..."
	uv build
	@echo "Repairing wheels for manylinux compatibility..."
	@if command -v auditwheel >/dev/null 2>&1; then \
		echo "Using auditwheel to repair wheels..."; \
		for wheel in dist/*.whl; do \
			if [[ "$$wheel" == *"linux_x86_64.whl" ]]; then \
				echo "Repairing $$wheel"; \
				auditwheel repair "$$wheel" --plat manylinux_2_39_x86_64 -w dist/; \
				rm "$$wheel"; \
			fi; \
		done; \
	else \
		echo "auditwheel not found. Installing..."; \
		uv run pip install auditwheel; \
		for wheel in dist/*.whl; do \
			if [[ "$$wheel" == *"linux_x86_64.whl" ]]; then \
				echo "Repairing $$wheel"; \
				uv run auditwheel repair "$$wheel" --plat manylinux_2_39_x86_64 -w dist/; \
				rm "$$wheel"; \
			fi; \
		done; \
	fi
	@echo "Publishing to PyPI..."
	@if [ ! -f "$$HOME/.pypirc" ] && [ -z "$$TWINE_PASSWORD" ]; then \
		echo "ERROR: Authentication required. Either:"; \
		echo "  1. Create ~/.pypirc file with your API token, OR"; \
		echo "  2. Set environment variable: export TWINE_PASSWORD='pypi-...'"; \
		echo ""; \
		echo "Get API token from: https://pypi.org/manage/account/#api-tokens"; \
		exit 1; \
	fi
	uv publish
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/

# Run all tests
test:
	uv run --group test pytest

# Run example scripts
test-examples:
	uv run python examples/test_binding.py
	uv run python examples/benchmark_threading.py