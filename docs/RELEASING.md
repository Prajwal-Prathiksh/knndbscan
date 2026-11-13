# Releasing

Releases are built, tested and published by `.github/workflows/ci.yml`, with
[trusted publishing](https://docs.pypi.org/trusted-publishers/): no API tokens are stored
anywhere. A release goes to TestPyPI first and is promoted to PyPI by hand.

| Step | Trigger | Result |
|---|---|---|
| 1 | push a tag `vX.Y.Z` | full CI; upload to TestPyPI; GitHub **pre-release** with notes and files |
| 2 | Actions > CI > Run workflow on the tag, with *publish_to_pypi* | full CI; upload to PyPI; the pre-release becomes the latest **release** |

Each upload waits for every other job: lint, the tests, the MPI and sanitizer suites,
wheels for Linux x86_64 / aarch64, macOS 26 arm64 / x86_64 and Windows x86_64 / arm64, and every wheel on the
oldest and newest supported Pythons with the oldest and newest allowed dependencies.

The GitHub release (`.github/workflows/release.yml`) has the version's section of
[CHANGELOG.md](../CHANGELOG.md) as its notes, followed by the list of merged pull requests,
and the wheels and sdist attached. It is created with the workflow's own token, so it
never starts another run.

## One-time setup

1. In the repository settings, under Environments, create `testpypi` and `pypi`. For
   each, limit deployments to tags matching `v*`; for `pypi`, also require a reviewer.
2. Add a trusted publisher to the `knndbscan` project on
   [TestPyPI](https://test.pypi.org/manage/projects/) and on
   [PyPI](https://pypi.org/manage/account/publishing/) (on PyPI, as a pending publisher
   until the first upload creates the project): owner and repository of this repository,
   workflow `ci.yml`, environment `testpypi` or `pypi`.

## Making a release

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty
   `## [Unreleased]` above it, and update the links at the bottom. Set `version` in
   `pyproject.toml` to `X.Y.Z`. Commit, and merge to `main` once CI passes.
2. Tag the merge commit and push only that tag:

   ```bash
   git tag -a vX.Y.Z -m "knndbscan X.Y.Z" && git push origin vX.Y.Z
   ```

3. Check the TestPyPI release in a fresh environment (`--extra-index-url` serves the
   dependencies from PyPI; TestPyPI's copies are not maintained):

   ```bash
   uv venv /tmp/check && uv pip install --python /tmp/check \
       --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       knndbscan==X.Y.Z
   ```

4. To publish to PyPI: Actions > CI > Run workflow, choose the tag `vX.Y.Z` under "Use
   workflow from", tick *publish_to_pypi*, and run it.

Indexes never accept a version twice. If something is wrong after an upload, fix it and
release the next version; a re-run of a failed upload skips the files already uploaded. To
rewrite a GitHub release's notes after editing CHANGELOG.md, run Actions > GitHub release
> Run workflow with the tag.
