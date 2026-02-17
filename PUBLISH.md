# Publishing to PyPI

## Setup

### 1. Create API tokens

Generate tokens for:

* **Test PyPI:** [https://test.pypi.org/manage/account/#api-tokens](https://test.pypi.org/manage/account/#api-tokens)
* **Production PyPI:** [https://pypi.org/manage/account/#api-tokens](https://pypi.org/manage/account/#api-tokens)

### 2. Configure environment variables

Create a `.env` file:

```bash
# Test PyPI token
UV_PUBLISH_TOKEN=pypi-...your-test-token...

# Production PyPI token (add when ready)
UV_PUBLISH_TOKEN_PYPI=pypi-...your-production-token...
```

---

## Publishing (recommended: `uv`)

### Publish workflow

```bash
# Publish to Test PyPI
make publish-test

# Verify install from Test PyPI
uv add knndbscan

# Install with GPU extras
uv add --extra gpu knndbscan

# Publish to production PyPI
make publish
```

---

## Installation

### Using uv (recommended)

#### Configure Test PyPI index

Add the following to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
explicit = true

[tool.uv.sources]
knndbscan = { index = "testpypi" }
```

**What this does?**
* Defines a custom package index (`testpypi`) for installing experimental builds.
* `explicit = true` ensures this index is used **only when explicitly requested**, preventing uv from searching it for unrelated packages.
* Maps `knndbscan` to the Test PyPI index.

**Why this matters?**
* Prevents mixing packages from different indexes (e.g., accidentally installing `numpy` or other dependencies from Test PyPI instead of the main PyPI).
* Avoids incompatible or experimental builds leaking into your environment.
* Ensures reproducible installs and safe testing of pre-release packages.

```bash
# Install without GPU support
uv add knndbscan

# Install with GPU support
uv add --extra gpu knndbscan
```

### Using pip (alternative)

```bash
# Test PyPI install
pip install --index-url https://test.pypi.org/simple/ knndbscan

# Test PyPI with GPU extras
pip install --index-url https://test.pypi.org/simple/ knndbscan[gpu]

# Production install
pip install knndbscan

# Production with GPU extras
pip install knndbscan[gpu]
```