# Developer tasks for knndbscan; `make` lists them. Tools come from the uv environment
# (`make install`); only the MPI targets need a system MPI (mpicxx, mpiexec).
SHELL := /bin/bash
.DEFAULT_GOAL := help
.SUFFIXES:

# ----------------------------------------------------------------------------- settings
# Override on the command line, e.g. `make test-dist PYTHONS=3.15 RESOLUTIONS=highest`.
PYTHONS ?= 3.10 3.11 3.12 3.13 3.14 3.15 3.14t
RESOLUTIONS ?= highest lowest-direct

BUILD := build
CPP_FILES := $(wildcard cpp/*.cpp cpp/*.h cpp/include/*.h cpp/cli/*.cpp cpp/core/* cpp/tests/*)

SANITIZE_FLAGS := -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
# The CLI tests start up to 7 ranks; Open MPI refuses more ranks than physical cores unless
# told to oversubscribe (OMPI_ for Open MPI 4, PRTE_ for 5; MPICH ignores both).
OVERSUBSCRIBE := OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
# Open MPI's UCX and libfabric components crash under AddressSanitizer; the ob1 layer over
# its own shared-memory transport does not. Exclusions hold for Open MPI 4 and 5 alike.
MPI_FOR_ASAN := OMPI_MCA_pml=ob1 OMPI_MCA_btl=^uct,ofi,smcuda OMPI_MCA_osc=^ucx OMPI_MCA_mtl=^ofi

# $(call cmake_build,<directory>,<CMake options>): configure and build one CMake tree.
define cmake_build
uv run cmake -S . -B $(1) -G Ninja $(2)
uv run cmake --build $(1)
endef

##@ Setup

.PHONY: install
install: ## Editable install with the dev tools; rebuilds the C library when cpp/ changes
	uv sync

.PHONY: compile-commands
compile-commands: ## build/compile_commands.json for editors (C/C++ extension, clangd)
	uv run cmake -S . -B $(BUILD)/ide -G Ninja -DKNNDBSCAN_BUILD_CLI=ON -DKNNDBSCAN_BUILD_TESTS=ON
	ln -sf ide/compile_commands.json $(BUILD)/compile_commands.json

##@ Checks

.PHONY: check
check: lint test test-cpp ## Lint and all tests that need no MPI
	@echo "check passed (run check-all where MPI is installed)"

.PHONY: check-all
check-all: check test-cli test-sanitize ## check, plus the MPI and sanitizer suites

.PHONY: lint
lint: ## ruff, mypy and clang-format, without changing files
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run clang-format --dry-run --Werror $(CPP_FILES)

.PHONY: format
format: ## Apply ruff fixes and formatting, and clang-format
	uv run ruff check --fix .
	uv run ruff format .
	uv run clang-format -i $(CPP_FILES)

##@ Tests

.PHONY: test
test: ## Python tests on the editable install
	uv run pytest

.PHONY: coverage
coverage: ## Python tests with a coverage report (terminal and htmlcov/)
	uv run pytest --cov --cov-report=term-missing --cov-report=html

.PHONY: test-cpp
test-cpp: ## C/C++ tests in cpp/tests: C API contract, MPI stand-in, cycle grouping
	$(call cmake_build,$(BUILD)/cpp-tests,-DKNNDBSCAN_BUILD_TESTS=ON)
	uv run ctest --test-dir $(BUILD)/cpp-tests --output-on-failure

.PHONY: test-cli
test-cli: cli ## The MPI program against the definitions on 1-7 ranks (needs MPI)
	KNNDBSCAN_CLI=$(CURDIR)/$(BUILD)/cli/knndbscan $(OVERSUBSCRIBE) uv run pytest tests/test_cli.py

# One build of the library, the CLI and the C tests with AddressSanitizer and UBSan (any
# undefined behaviour aborts). Python loads the sanitized library through
# KNNDBSCAN_LIBRARY, with the ASan runtime preloaded because python itself is not
# instrumented; plain python, not `uv run`, so uv is not preloaded too. Leak checks are
# off: the core never frees its per-call scratch arrays.
.PHONY: test-sanitize
test-sanitize: ## test-cpp, test and test-cli on AddressSanitizer + UBSan builds (needs MPI)
	$(call cmake_build,$(BUILD)/sanitize,-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	  -DKNNDBSCAN_BUILD_CLI=ON -DKNNDBSCAN_BUILD_TESTS=ON \
	  "-DCMAKE_C_FLAGS=$(SANITIZE_FLAGS)" "-DCMAKE_CXX_FLAGS=$(SANITIZE_FLAGS)" \
	  "-DCMAKE_EXE_LINKER_FLAGS=$(SANITIZE_FLAGS)" "-DCMAKE_SHARED_LINKER_FLAGS=$(SANITIZE_FLAGS)")
	ASAN_OPTIONS=detect_leaks=0 uv run ctest --test-dir $(BUILD)/sanitize --output-on-failure
	KNNDBSCAN_LIBRARY=$(CURDIR)/$(BUILD)/sanitize/libknndbscan.so ASAN_OPTIONS=detect_leaks=0 \
	  LD_PRELOAD="$$(gcc -print-file-name=libasan.so)" \
	  .venv/bin/python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_cli.py
	KNNDBSCAN_CLI=$(CURDIR)/$(BUILD)/sanitize/knndbscan ASAN_OPTIONS=detect_leaks=0 \
	  $(MPI_FOR_ASAN) $(OVERSUBSCRIBE) uv run pytest -q -p no:cacheprovider tests/test_cli.py

##@ Packaging

.PHONY: build
build: ## Wheel and sdist into dist/
	rm -rf dist
	uv build

.PHONY: test-wheel
test-wheel: build ## Build, then test-dist
	@$(MAKE) --no-print-directory test-dist

# The wheel is py3-none-<platform>; this proves it on each interpreter, with the newest and
# the oldest dependency versions the metadata allows. The dependencies are installed from
# pyproject.toml as direct requirements, because lowest-direct lowers only those.
.PHONY: test-dist
test-dist: ## Test dist/*.whl on each of PYTHONS x RESOLUTIONS, in fresh environments
	@set -e; for python in $(PYTHONS); do \
	  for resolution in $(RESOLUTIONS); do \
	    echo "=== Python $$python, $$resolution dependencies"; \
	    env=$(BUILD)/test-wheel/$$python-$$resolution; rm -rf $$env; \
	    uv venv -q --python $$python $$env; \
	    uv pip install -q --python $$env pytest; \
	    uv pip install -q --python $$env --only-binary :all: --resolution $$resolution \
	      -r pyproject.toml; \
	    uv pip install -q --python $$env --no-deps dist/*.whl; \
	    uv pip list -q --python $$env 2>/dev/null | grep -E '^(numpy|scikit-learn) '; \
	    $$env/bin/python -m pytest -q -p no:cacheprovider tests; \
	  done; \
	done

##@ Programs

.PHONY: cli
cli: ## The MPI command-line program, build/cli/knndbscan (needs MPI)
	$(call cmake_build,$(BUILD)/cli,-DKNNDBSCAN_BUILD_LIBRARY=OFF -DKNNDBSCAN_BUILD_CLI=ON)

.PHONY: examples
examples: ## Run the CPU examples (plots go to figures/)
	MPLBACKEND=Agg uv run python examples/compare_with_dbscan.py
	uv run python examples/benchmark_threads.py 500000

.PHONY: bench
bench: ## Thread scaling of the clustering step on a 2M-point graph
	uv run python examples/benchmark_threads.py

##@ Housekeeping

.PHONY: clean
clean: ## Remove build outputs, reports and caches (keeps .venv)
	rm -rf $(BUILD) dist htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache figures
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +

.PHONY: help
help: ## Show this list
	@awk 'BEGIN { FS = ":.*## " } \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
	  /^[a-z][a-z-]*:.*## / { printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
