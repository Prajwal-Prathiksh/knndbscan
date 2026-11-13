"""Print the GitHub release notes of a tag: its CHANGELOG.md section, then the merged pull requests.

Usage: python3 .github/scripts/release_notes.py v0.2.0

The pull-request list comes from GitHub's release-notes API (through `gh`, for the
repository in $GITHUB_REPOSITORY). Exits with an error when CHANGELOG.md has no section
for the version, so that no release goes out undocumented.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def changelog_section(changelog: str, version: str) -> str:
    """The text under `## [version]`, up to the next section or the link definitions."""
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)"
    match = re.search(pattern, changelog, flags=re.MULTILINE | re.DOTALL)
    if match is None or not match.group(1).strip():
        sys.exit(f"CHANGELOG.md has no entries under '## [{version}]'")
    return match.group(1).strip()


def merged_pull_requests(repository: str, tag: str) -> str:
    response = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases/generate-notes",
            "-f",
            f"tag_name={tag}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(response.stdout)["body"].strip()


def main() -> None:
    tag = sys.argv[1]
    changelog = Path(__file__).resolve().parents[2].joinpath("CHANGELOG.md").read_text()
    notes = changelog_section(changelog, tag.removeprefix("v"))
    pull_requests = merged_pull_requests(os.environ["GITHUB_REPOSITORY"], tag)
    sys.stdout.write(f"{notes}\n\n{pull_requests}\n")


if __name__ == "__main__":
    main()
