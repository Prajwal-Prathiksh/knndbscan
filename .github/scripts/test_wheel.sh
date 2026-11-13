#!/usr/bin/env bash
# Install dist/*.whl into a fresh environment for one Python and one dependency resolution,
# then run the test suite against that install, not the source tree. Used by `make
# test-dist` and by CI on Linux, macOS and Windows (Git Bash).
#
# Usage: .github/scripts/test_wheel.sh PYTHON RESOLUTION    e.g. 3.15 lowest-direct
#
# The wheel is py3-none-<platform>, so this proves it on each interpreter, with the newest
# (highest) or the oldest (lowest-direct) dependency versions the metadata allows.
set -euo pipefail
python=$1
resolution=$2
env=build/test-wheel/$python-$resolution

rm -rf "$env"
uv venv -q --python "$python" "$env"
uv pip install -q --python "$env" pytest
# lowest-direct lowers only direct requirements, so the dependencies are installed directly.
uv pip install -q --python "$env" --only-binary :all: --resolution "$resolution" -r pyproject.toml
uv pip install -q --python "$env" --no-deps dist/*.whl
uv pip list -q --python "$env" | grep -E '^(numpy|scikit-learn) '

interpreter=$env/bin/python
[ -x "$interpreter" ] || interpreter=$env/Scripts/python.exe  # Windows layout
"$interpreter" -m pytest -q -p no:cacheprovider tests
