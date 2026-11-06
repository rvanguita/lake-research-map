"""Pre-commit guard: refuse a documentation-only commit on `main`/`master`.

`main` is protected on GitHub; documentation changes go through a
`docs/<topic>` branch and a pull request like any other change. Wired in
`.pre-commit-config.yaml` as the `block-docs-on-main` hook.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from fnmatch import fnmatch

PROTECTED_BRANCHES = ("main", "master")
DOC_PATTERNS = ("docs/*", "*.md", "README*", "CLAUDE.md")


def is_doc_file(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in DOC_PATTERNS)


def blocks_commit(branch: str, staged_files: Sequence[str]) -> bool:
    """True when a commit on a protected branch stages documentation only."""
    files = [path for path in staged_files if path]
    return branch in PROTECTED_BRANCHES and bool(files) and all(map(is_doc_file, files))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    staged = _git("diff", "--cached", "--name-only").splitlines()
    if not blocks_commit(branch, staged):
        return 0
    print(
        f"Commit refused: it stages documentation only, directly on '{branch}'.\n"
        "Create a dedicated branch first, for example:\n"
        "  git checkout -b docs/<topic>\n"
        "and commit there.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
