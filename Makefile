SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help install test coverage lint format build test-wheel test-dist cli test-cli clean

PYTHONS ?= 3.10 3.11 3.12 3.13 3.14 3.15 3.14t
RESOLUTIONS ?= highest lowest-direct
CLI_BUILD := build/cli

help:
	@echo "install     uv sync: editable install plus the dev group"
	@echo "test        pytest on the editable install"
	@echo "coverage    pytest with a coverage report (htmlcov/)"
	@echo "lint        ruff, mypy and clang-format checks"
	@echo "format      apply ruff and clang-format"
	@echo "build       wheel and sdist into dist/"
	@echo "test-wheel  build, then test-dist"
	@echo "test-dist   install dist/*.whl for each of PYTHONS x RESOLUTIONS and run the tests"
	@echo "cli         build the MPI command-line program into $(CLI_BUILD)/"
	@echo "test-cli    build the CLI and compare it with the library on 1-3 MPI ranks"
	@echo "clean       remove build artifacts"

install:
	uv sync

test:
	uv run pytest

coverage:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run clang-format --dry-run --Werror cpp/*.cpp cpp/*.h cpp/include/*.h cpp/cli/*.cpp cpp/core/*

format:
	uv run ruff check --fix .
	uv run ruff format .
	uv run clang-format -i cpp/*.cpp cpp/*.h cpp/include/*.h cpp/cli/*.cpp cpp/core/*

build:
	rm -rf dist
	uv build

# The wheel is py3-none-<platform>; this proves it on each interpreter, with the newest
# and the oldest dependency versions the metadata allows. The dependencies are installed
# from pyproject.toml as direct requirements, because lowest-direct lowers only those.
test-wheel: build
	$(MAKE) test-dist

test-dist:
	@set -e; for python in $(PYTHONS); do \
	  for resolution in $(RESOLUTIONS); do \
	    echo "=== Python $$python, $$resolution dependencies"; \
	    env=build/test-wheel/$$python-$$resolution; rm -rf $$env; \
	    uv venv -q --python $$python $$env; \
	    uv pip install -q --python $$env pytest; \
	    uv pip install -q --python $$env --only-binary :all: --resolution $$resolution \
	      -r pyproject.toml; \
	    uv pip install -q --python $$env --no-deps dist/*.whl; \
	    uv pip list -q --python $$env 2>/dev/null | grep -E '^(numpy|scikit-learn) '; \
	    $$env/bin/python -m pytest -q -p no:cacheprovider tests; \
	  done; \
	done

cli:
	uv run cmake -S . -B $(CLI_BUILD) -G Ninja -DKNNDBSCAN_BUILD_LIBRARY=OFF -DKNNDBSCAN_BUILD_CLI=ON
	uv run cmake --build $(CLI_BUILD)

# The tests start up to 7 ranks; Open MPI refuses more ranks than physical cores unless
# told to oversubscribe (OMPI_ for Open MPI 4, PRTE_ for 5; MPICH ignores both).
test-cli: cli
	KNNDBSCAN_CLI=$(CURDIR)/$(CLI_BUILD)/knndbscan \
	OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe \
	uv run pytest tests/test_cli.py

# Configures the library and the MPI CLI (both MPI and non-MPI views of cpp/core) without
# building them, so editors resolve <omp.h>, <mpi.h> and the project headers.
compile-commands:
	uv run cmake -S . -B build/ide -G Ninja -DKNNDBSCAN_BUILD_CLI=ON
	ln -sf ide/compile_commands.json build/compile_commands.json

clean:
	rm -rf build dist htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
