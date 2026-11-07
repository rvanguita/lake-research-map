"""The core / `apps/` import boundary (CLAUDE.md "Layout: core and apps/").

The pipeline core (`config.py`, `pipeline.py`, `quality.py`, `db/`, `ingest/`,
`transform/`) must not depend on an application, and the applications import
the core but not each other -- the one sanctioned edge is the dashboard using
`apps.api`. Any sanctioned exception is written down below as an exact
allowlist -- empty since the Phase 3 refactor moved the last shared helpers into
`transform/` -- so the test fails both when a new edge appears and when a listed
one is removed without the list being tightened.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = ("config.py", "quality.py", "pipeline.py", "runs.py", "db", "ingest", "transform")

# module path (relative to src/) -> imported-module prefixes it may still use.
# Empty since Phase 3: the `analytics` helpers moved into the core and the app
# commands are wired by `cli/`, not `pipeline.py`. A new entry is new debt and
# needs a reason in SDD §12.
KNOWN_CORE_TO_APPS: dict[str, tuple[str, ...]] = {}


def _imports(path: Path) -> list[tuple[int, str]]:
    """(line, absolute module) for every import in `path`, lazy ones included.

    `from apps.qdrant import cli` names `apps.qdrant.cli`, so the allowlist can
    be as exact as the import itself.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
            found.extend((node.lineno, f"{node.module}.{alias.name}") for alias in node.names)
    return found


def _under(prefix: str, module: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _sources(*roots: str) -> list[tuple[str, Path]]:
    """(allowlist key, path) for every module under `roots`, keyed relative to src/."""
    paths: list[Path] = []
    for root in roots:
        target = SRC / root
        paths.extend([target] if target.is_file() else sorted(target.rglob("*.py")))
    assert paths, f"nothing to scan under {roots}"
    return [(path.relative_to(SRC).as_posix(), path) for path in paths]


def _violations(
    sources: list[tuple[str, Path]], forbidden: tuple[str, ...], allowed=None
) -> list[str]:
    allowed = allowed or {}
    hits: set[str] = set()
    for key, path in sources:
        found = _imports(path)
        by_line: dict[int, set[str]] = {}
        for line, module in found:
            by_line.setdefault(line, set()).add(module)
        for line, module in found:
            if not any(_under(prefix, module) for prefix in forbidden):
                continue
            allowed_here = allowed.get(key, ())
            if any(_under(prefix, module) for prefix in allowed_here):
                continue
            # `from apps.qdrant import cli` yields both `apps.qdrant` and
            # `apps.qdrant.cli`; the parent is covered only when an allowed child
            # is imported on that same line -- a bare `import apps.dashboard`
            # beside an allowed `apps.dashboard.analytics` would hand the core the
            # whole package by attribute, so it still fails.
            if any(
                _under(module, prefix) and any(_under(prefix, other) for other in by_line[line])
                for prefix in allowed_here
            ):
                continue
            hits.add(f"{key}:{line}: imports {module}")
    return sorted(hits)


def test_the_core_imports_no_application_beyond_the_known_debt():
    assert _violations(_sources(*CORE), ("apps",), KNOWN_CORE_TO_APPS) == []


def test_every_allowlisted_edge_still_exists():
    """A stale entry would let the edge come back unnoticed: tighten the list instead."""
    for key, prefixes in KNOWN_CORE_TO_APPS.items():
        modules = [module for _, module in _imports(SRC / key)]
        for prefix in prefixes:
            assert any(_under(prefix, module) for module in modules), (
                f"{key} no longer imports {prefix}; remove it from KNOWN_CORE_TO_APPS"
            )


def test_the_cli_is_the_composition_root_and_nothing_imports_it():
    """`src/cli/` wires the application commands; the core and the apps never import it.

    The app commands (`vectors`, `search`, `screen`, `mcp`) used to be wired by
    `pipeline.py`, which made the core import two applications. They moved to
    `cli/`, the one module tree outside `apps/` allowed to import an application,
    and an edge back into it would rebuild the inversion one hop removed.
    """
    cli_modules = {module for _, path in _sources("cli") for _, module in _imports(path)}
    assert any(_under("apps", module) for module in cli_modules)
    assert _violations(_sources(*CORE, "apps"), ("cli",)) == []


@pytest.mark.parametrize(
    ("app", "forbidden"),
    [
        # The API is what the dashboard and the CLI call; it must run headless.
        ("apps/api", ("streamlit", "apps.dashboard", "apps.qdrant")),
        ("apps/qdrant", ("streamlit", "apps.api", "apps.dashboard")),
    ],
)
def test_applications_do_not_import_each_other(app, forbidden):
    assert _violations(_sources(app), forbidden) == []


def test_the_scan_sees_a_violation_and_honours_an_exact_allowance(tmp_path):
    # Guards the scan itself, on a module written here rather than on src/, so
    # it keeps testing the scanner after the real debt is paid off.
    module = tmp_path / "evidence.py"
    module.write_text(
        "from apps.dashboard.analytics import canonical_author\n"
        "def lazy():\n"
        "    from apps.qdrant import cli\n",
        encoding="utf-8",
    )
    sources = [("transform/evidence.py", module)]

    assert _violations(sources, ("apps",)) == [
        "transform/evidence.py:1: imports apps.dashboard.analytics",
        "transform/evidence.py:1: imports apps.dashboard.analytics.canonical_author",
        "transform/evidence.py:3: imports apps.qdrant",
        "transform/evidence.py:3: imports apps.qdrant.cli",
    ]
    allowed = {"transform/evidence.py": ("apps.dashboard.analytics",)}
    assert _violations(sources, ("apps",), allowed) == [
        "transform/evidence.py:3: imports apps.qdrant",
        "transform/evidence.py:3: imports apps.qdrant.cli",
    ]
