.PHONY: build-ext test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  build-ext  Build the C++ extension in place"
	@echo "  test       Run the test_binding.py script"
	@echo "  clean      Clean build artifacts"
	@echo "  help       Show this help message"

# Build the extension
build-ext:
	uv run python setup.py build_ext --inplace

# Run tests
test: build-ext
	uv run python test_binding.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -f knndbscan/_core*.so
	rm -f knndbscan/*.so