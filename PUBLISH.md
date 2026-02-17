# Publishing to PyPI

## Setup

1. Get API tokens:
   - Test PyPI: https://test.pypi.org/manage/account/#api-tokens
   - Production PyPI: https://pypi.org/manage/account/#api-tokens

2. Create `.env` file:
```bash
# Test PyPI token
UV_PUBLISH_TOKEN=pypi-...your-test-token...

# Production PyPI token (add when ready for production)
UV_PUBLISH_TOKEN_PYPI=pypi-...your-production-token...
```

## Publishing

```bash
# Test on Test PyPI first
make publish-test

# Verify installation
pip install --index-url https://test.pypi.org/simple/ knndbscan

# Publish to production PyPI
make publish
```
