# Releasing

Releases are built and published by `.github/workflows/ci.yml` with
[trusted publishing](https://docs.pypi.org/trusted-publishers/): no API tokens are stored
anywhere.

| Trigger | Publishes to | GitHub environment |
|---|---|---|
| pushing a tag `v*` | TestPyPI | `testpypi` |
| publishing a GitHub release | PyPI | `pypi` |

Both run only after every other job passes: lint, the tests, the MPI CLI test, wheels for
Linux x86_64 / aarch64 and macOS arm64, and the Linux wheel on every supported Python with
the newest and the oldest allowed dependencies.

## One-time setup

1. In the GitHub repository settings, create the environments `testpypi` and `pypi`
   (optionally with required reviewers for `pypi`).
2. On [TestPyPI](https://test.pypi.org/manage/account/publishing/) and
   [PyPI](https://pypi.org/manage/account/publishing/), add a trusted publisher for the
   project `knndbscan`: the repository owner and name, workflow `ci.yml`, and the matching
   environment name.

## Making a release

1. Set `version` in `pyproject.toml` and commit.
2. Run `make lint test test-cli test-wheel` locally.
3. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`. The workflow publishes to
   TestPyPI.
4. Check the TestPyPI release in a fresh environment:

   ```bash
   uv venv /tmp/check && uv pip install --python /tmp/check \
       --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       knndbscan==0.2.0
   ```

   `--extra-index-url` serves NumPy and scikit-learn from PyPI; TestPyPI's copies of
   them are not maintained.
5. Publish a GitHub release for the tag. The workflow publishes the same artifacts to PyPI.
