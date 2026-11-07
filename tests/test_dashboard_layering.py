"""NFR-08: the dashboard's maths stays Streamlit-free, and caching stays in one layer.

Every module of the `analytics/` package, `forecasting.py` and `qualis.py` are pure
pandas/numpy so they can be unit-tested without a running app; one stray
`import streamlit` there would make every one of those tests depend on
Streamlit's runtime. `loaders.py` is the only `@st.cache_data` /
`@st.cache_resource` layer, so cache lifetimes and invalidation live in one
file -- a decorator on a page or a pure module would add a second, silent one.

Both are checked on the AST: `qualis.py`'s docstring names `@st.cache_data` in
prose, and `components.py` calls `st.cache_data.clear()` for the refresh
button, neither of which is a cache.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "apps" / "dashboard"
# Globbed, so a new analytics submodule is covered the day it is added.
PURE_MODULES = (
    *sorted(
        path.relative_to(DASHBOARD).as_posix() for path in (DASHBOARD / "analytics").glob("*.py")
    ),
    "forecasting.py",
    "qualis.py",
)
CACHE_DECORATORS = {"cache_data", "cache_resource"}


def _imported_modules(tree: ast.AST) -> list[str]:
    """Every module an Import/ImportFrom names, at any depth (lazy imports count)."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _cache_uses(tree: ast.AST) -> list[int]:
    """Lines that create a Streamlit cache: as a decorator or by calling it.

    `st.cache_data.clear()` is an Attribute *on* the cache object, never the
    cache call itself, so it is not reported.
    """

    def is_cache(node: ast.AST) -> bool:
        target = node.func if isinstance(node, ast.Call) else node
        return isinstance(target, ast.Attribute) and target.attr in CACHE_DECORATORS

    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            lines.extend(d.lineno for d in node.decorator_list if is_cache(d))
        elif isinstance(node, ast.Call) and is_cache(node):
            lines.append(node.lineno)
    return sorted(set(lines))


# The UI layers import Streamlit themselves, so a pure module reaching one of them
# would pull Streamlit in transitively -- the dependency NFR-08 keeps out of the
# unit tests.
UI_MODULES = tuple(
    f"apps.dashboard.{name}" for name in ("loaders", "components", "charts", "pages", "app")
)


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_dashboard_modules_do_not_import_streamlit(module):
    tree = ast.parse((DASHBOARD / module).read_text(encoding="utf-8"))

    offenders = [
        name
        for name in _imported_modules(tree)
        if name.split(".")[0] == "streamlit"
        or any(name == ui or name.startswith(ui + ".") for ui in UI_MODULES)
    ]

    assert offenders == [], f"{module} imports {offenders}"


def test_the_pure_module_list_covers_the_analytics_package():
    # Guards the glob: an empty match would leave the package unchecked.
    assert "analytics/counts.py" in PURE_MODULES


def test_streamlit_caches_are_declared_only_in_loaders():
    offenders = {}
    for path in sorted(DASHBOARD.rglob("*.py")):
        if path.name == "loaders.py" and path.parent == DASHBOARD:
            continue
        lines = _cache_uses(ast.parse(path.read_text(encoding="utf-8")))
        if lines:
            offenders[str(path.relative_to(DASHBOARD))] = lines

    assert offenders == {}


def test_loaders_does_declare_the_caches():
    # Guards the scan itself: if the detector stopped matching, the test above
    # would pass on an empty result.
    tree = ast.parse((DASHBOARD / "loaders.py").read_text(encoding="utf-8"))

    assert _cache_uses(tree)


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("@st.cache_data(ttl=60)\ndef f():\n    pass", [1]),
        ("@st.cache_resource\ndef f():\n    pass", [1]),
        ("cached = st.cache_data(ttl=5)(load)", [1]),
        ("st.cache_data.clear()", []),
        ('"""Caching is `@st.cache_data`\'s job."""', []),
    ],
)
def test_the_cache_detector(snippet, expected):
    assert _cache_uses(ast.parse(snippet)) == expected
