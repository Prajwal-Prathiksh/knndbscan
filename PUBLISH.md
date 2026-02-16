# Publishing to PyPI

## Prerequisites

- PyPI account: https://pypi.org/account/register/
- Test PyPI account: https://test.pypi.org/account/register/
- API tokens from both services

## Authentication

**Option 1: ~/.pypirc file (recommended)**
```ini
[testpypi]
  username = __token__
  password = pypi-...your-test-token...

[pypi]
  username = __token__
  password = pypi-...your-prod-token...
```

**Option 2: Environment variable**
```bash
export TWINE_PASSWORD='pypi-...your-token...'
```

## Quick Start

```bash
# Test on Test PyPI first
make publish-test

# Verify installation
pip install --index-url https://test.pypi.org/simple/ knndbscan
python -c "import knndbscan; print('Success!')"

# Publish to production
make publish
```
