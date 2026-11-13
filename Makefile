SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help install test coverage lint format build test-wheel test-dist cli test-cli test-cpp test-sanitize compile-commands clean

PYTHONS ?= 3.10 3.11 3.12 3.13 3.14 3.15 3.14t
RESOLUTIONS ?= highest lowest-direct
CLI_BUILD := build/cli
CPP_TEST_BUILD := build/cpp-tests
SANITIZE_BUILD := build/sanitize
SANITIZE_FLAGS := -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
# The tests start up to 7 ranks; Open MPI refuses more ranks than physical cores unless
# told to oversubscribe (OMPI_ for Open MPI 4, PRTE_ for 5; MPICH ignores both).
OVERSUBSCRIBE := OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
# Open MPI's UCX and libfabric components crash under AddressSanitizer; the ob1 messaging
# layer over its own shared-memory transport does not. Exclusions keep this valid for both
# Open MPI 4 (transport "vader") and 5 ("sm").
MPI_WITHOUT_UCX := OMPI_MCA_pml=ob1 OMPI_MCA_btl=^uct,ofi,smcuda OMPI_MCA_osc=^ucx OMPI_MCA_mtl=^ofi

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
	@echo "test-cli    build the MPI CLI and check it against the definitions on 1-7 ranks"
	@echo "test-cpp    build and run the C/C++ tests in cpp/tests (ctest)"
	@echo "test-sanitize  test-cpp, test and test-cli against AddressSanitizer+UBSan builds"
	@echo "compile-commands  build/compile_commands.json for editors (C/C++ extension, clangd)"
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

test-cli: cli
	KNNDBSCAN_CLI=$(CURDIR)/$(CLI_BUILD)/knndbscan $(OVERSUBSCRIBE) uv run pytest tests/test_cli.py

test-cpp:
	uv run cmake -S . -B $(CPP_TEST_BUILD) -G Ninja -DKNNDBSCAN_BUILD_TESTS=ON
	uv run cmake --build $(CPP_TEST_BUILD)
	uv run ctest --test-dir $(CPP_TEST_BUILD) --output-on-failure

# One build of the library, the CLI and the C tests with AddressSanitizer and UBSan (any
# undefined behaviour aborts). Python loads the sanitized library through
# KNNDBSCAN_LIBRARY, with the ASan runtime preloaded because python itself is not
# instrumented. Leak checks are off: the core never frees its per-call scratch arrays.
test-sanitize:
	uv run cmake -S . -B $(SANITIZE_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	  -DKNNDBSCAN_BUILD_CLI=ON -DKNNDBSCAN_BUILD_TESTS=ON \
	  "-DCMAKE_C_FLAGS=$(SANITIZE_FLAGS)" "-DCMAKE_CXX_FLAGS=$(SANITIZE_FLAGS)" \
	  "-DCMAKE_EXE_LINKER_FLAGS=$(SANITIZE_FLAGS)" "-DCMAKE_SHARED_LINKER_FLAGS=$(SANITIZE_FLAGS)"
	uv run cmake --build $(SANITIZE_BUILD)
	ASAN_OPTIONS=detect_leaks=0 uv run ctest --test-dir $(SANITIZE_BUILD) --output-on-failure
	KNNDBSCAN_LIBRARY=$(CURDIR)/$(SANITIZE_BUILD)/libknndbscan.so ASAN_OPTIONS=detect_leaks=0 \
	  LD_PRELOAD="$$(gcc -print-file-name=libasan.so)" \
	  .venv/bin/python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_cli.py
	KNNDBSCAN_CLI=$(CURDIR)/$(SANITIZE_BUILD)/knndbscan ASAN_OPTIONS=detect_leaks=0 \
	  $(MPI_WITHOUT_UCX) $(OVERSUBSCRIBE) uv run pytest -q -p no:cacheprovider tests/test_cli.py

# Configures the library and the MPI CLI (both MPI and non-MPI views of cpp/core) without
# building them, so editors resolve <omp.h>, <mpi.h> and the project headers.
compile-commands:
	uv run cmake -S . -B build/ide -G Ninja -DKNNDBSCAN_BUILD_CLI=ON
	ln -sf ide/compile_commands.json build/compile_commands.json

clean:
	rm -rf build dist htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
