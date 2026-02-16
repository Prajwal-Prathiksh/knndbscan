.PHONY: test

# Default target
help:
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo "  build         Build distribution packages (wheel and sdist)"
	@echo "  check         Run all checks (lint + test)"
	@echo "  clean         Clean build artifacts"
	@echo "  format        Format code with ruff and clang-format"
	@echo "  install       Install the package in editable mode"
	@echo "  install-debug Install package with optimizations and debug symbols (-O3 -g)"
	@echo "  install-gpu   Install the package with GPU support in editable mode"
	@echo "  lint          Run linting tools"
	@echo "  publish       Publish to PyPI"
	@echo "  publish-test  Publish to Test PyPI"
	@echo "  repair-wheels Repair wheels for manylinux compatibility"
	@echo "  test          Run unit tests with pytest"
	@echo "  test-examples Run example scripts"
	@echo "  test-gpu      Run GPU-specific example scripts"

# Build distribution packages
build:
	DEBUG=0 uv build

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

# Format code
format:
	uv run --group lint ruff format .
	uv run --group lint clang-format -i src/*.cpp include/*.h test/*.cpp

# Sync development dependencies and install package in editable mode
install: 
	uv sync --all-groups
	DEBUG=0 uv run pip install -e .

# Install package with optimizations and debug symbols
install-debug:
	uv sync --all-groups
	DEBUG=1 uv run pip install -e . --force-reinstall --no-deps

# Sync with GPU support and build C++ extension
install-gpu:
	uv sync --all-groups --extra gpu
	DEBUG=0 uv run pip install -e .

# Run linting tools
lint:
	uv run --group lint ruff check .
	uv run --group lint mypy knndbscan/ || true
	uv run --group lint clang-format --dry-run --Werror src/*.cpp include/*.h test/*.cpp

# Publish to PyPI
publish: build repair-wheels
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

# Publish to Test PyPI
publish-test: build repair-wheels
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


# Repair wheels for manylinux compatibility
repair-wheels:
	@echo "Repairing wheels for manylinux compatibility..."
	for wheel in dist/*.whl; do \
		if [[ "$$wheel" == *"linux_x86_64.whl" ]]; then \
			echo "Repairing $$wheel"; \
			uv run auditwheel repair "$$wheel" --plat manylinux_2_39_x86_64 -w dist/; \
			rm "$$wheel"; \
		fi; \
	done

# Run all tests
test:
	uv run --group test pytest

# Run example scripts
test-examples:
	uv run python examples/test_binding.py
	uv run python examples/benchmark_threading.py
	uv run python examples/usage.py

# Run GPU-specific example scripts
test-gpu:
	uv run python examples/usage_gpu.py