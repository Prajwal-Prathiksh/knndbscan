.PHONY: install test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  install    Install the package in editable mode"
	@echo "  test       Run the test_binding.py script"
	@echo "  clean      Clean build artifacts"
	@echo "  help       Show this help message"

# Install the package in editable mode
install:
	uv sync
	uv run pip install -e .

# Run tests
test: install
	uv run python test_binding.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf *.egg-info/
	rm -f knndbscan/_core*.so
	rm -f knndbscan/*.so