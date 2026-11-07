"""FR-11: nothing under `apps/dashboard/` writes -- not a table, not a file.

The dashboard is read-only with no exception (ADR-09): the pipeline runs from
the CLI only. That rule was broken twice before it was written down -- an
in-process embedding action that wrote the live `lit_chunks` table, and a
parquet cache `qualis.py` wrote next to the CAPES workbook inside `data/`. Both
were plain method calls a reviewer could miss, so this scans the source for
them instead of trusting review.

The scan is static and AST-based rather than a grep, because the dashboard is
full of the *names* of write calls doing harmless things: `DataFrame.merge`
(a join, not `Session.merge`), `to_csv(index=False)` (a string for a download
button, not a file), `set.add`, SQL `SELECT`s through `conn.execute`, and
docstrings that describe the old cache. Each rule below says what it lets
through and why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "apps" / "dashboard"

# Modules whose import alone means a write or a process launch.
_FORBIDDEN_MODULES = {"shutil", "subprocess"}
# `from sqlalchemy import insert|update|delete` builds DML statements.
_SQLALCHEMY_DML = {"insert", "update", "delete"}
# DML/DDL keywords inside whatever is handed to `execute(...)`.
_SQL_WRITE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|TRUNCATE|REPLACE|DROP|ALTER|CREATE)\b", re.IGNORECASE
)
# pandas writers that always take a destination.
_ALWAYS_WRITES = {
    "to_parquet",
    "to_pickle",
    "to_feather",
    "to_excel",
    "to_hdf",
    "to_sql",
    "to_stata",
}
# pandas writers that return a string when called with no destination -- the
# download buttons rely on exactly that (`to_csv(index=False).encode(...)`).
_WRITES_WITH_PATH = {"to_csv", "to_json"}
_PATH_KEYWORDS = {"path_or_buf", "path"}
# `pathlib.Path` methods that change the filesystem. (`rename`/`replace` are
# left out: on a DataFrame and a str they are pure.)
_PATH_MUTATORS = {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch", "symlink_to"}
_OS_MUTATORS = {
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rename",
    "renames",
    "replace",
    "makedirs",
    "mkdir",
    "truncate",
    "system",
}
# ORM writes. `add`/`flush` only count on something named like a session,
# because `set.add` and `file.flush` are everywhere; the others have no pure
# homonym on the objects a page handles.
_SESSION_ONLY = {"add", "flush"}
_ANY_RECEIVER = {"add_all", "commit", "delete", "bulk_save_objects", "bulk_insert_mappings"}
_SESSION_NAME = re.compile(r"(session|sess|^db)$", re.IGNORECASE)
# `DataFrame.merge` / `pd.merge` is a join; `Session.merge(instance)` is a write.
# Any pandas join keyword marks the call as the former.
_PANDAS_MERGE_KEYWORDS = {
    "on",
    "how",
    "left_on",
    "right_on",
    "left_index",
    "right_index",
    "suffixes",
    "indicator",
    "validate",
}


def _terminal_name(node: ast.AST) -> str:
    """`gold_session` for `gold_session`, `session` for `self.session`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _open_mode_writes(call: ast.Call, mode_position: int) -> bool:
    mode: ast.AST | None = None
    if len(call.args) > mode_position:
        mode = call.args[mode_position]
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if mode is None:
        return False  # the default mode is "r"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(flag in mode.value for flag in "wax+")
    return True  # a computed mode cannot be proven read-only


def write_calls(source: str, filename: str = "<snippet>") -> list[str]:
    """Every write-shaped call in `source`, as `filename:line: reason`."""
    tree = ast.parse(source, filename=filename)
    hits: list[str] = []

    def hit(node: ast.AST, reason: str) -> None:
        hits.append(f"{filename}:{node.lineno}: {reason}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                    hit(node, f"imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_MODULES:
                hit(node, f"imports from {node.module}")
            if root == "sqlalchemy":
                for alias in node.names:
                    if alias.name in _SQLALCHEMY_DML:
                        hit(node, f"imports the DML builder sqlalchemy.{alias.name}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in _FORBIDDEN_MODULES:
                hit(node, f"uses {node.value.id}.{node.attr}")
            elif node.value.id == "os" and node.attr in _OS_MUTATORS:
                hit(node, f"uses os.{node.attr}")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            if _open_mode_writes(node, 1):
                hit(node, "open() in a write mode")
            continue
        if not isinstance(func, ast.Attribute):
            continue
        method = func.attr
        receiver = _terminal_name(func.value)
        if method in _ANY_RECEIVER:
            hit(node, f".{method}()")
        elif method in _SESSION_ONLY and _SESSION_NAME.search(receiver):
            hit(node, f"{receiver}.{method}()")
        elif method == "merge":
            keywords = {keyword.arg for keyword in node.keywords}
            if receiver != "pd" and not keywords & _PANDAS_MERGE_KEYWORDS:
                hit(node, ".merge() with no join keyword (an ORM merge?)")
        elif method in {"execute", "exec_driver_sql"}:
            statement = " ".join(ast.unparse(arg) for arg in node.args)
            if _SQL_WRITE.search(statement):
                hit(node, f".{method}() of a write statement")
        elif method in _ALWAYS_WRITES:
            hit(node, f".{method}()")
        elif method in _WRITES_WITH_PATH:
            keywords = {keyword.arg for keyword in node.keywords}
            if node.args or keywords & _PATH_KEYWORDS:
                hit(node, f".{method}() to a path")
        elif method in _PATH_MUTATORS:
            hit(node, f".{method}()")
        elif method == "open" and _open_mode_writes(node, 0):
            hit(node, ".open() in a write mode")
    return hits


def _dashboard_sources() -> list[Path]:
    sources = sorted(DASHBOARD.rglob("*.py"))
    assert sources, f"no dashboard sources under {DASHBOARD}"
    return sources


def test_no_write_call_anywhere_under_the_dashboard():
    hits = [
        found
        for path in _dashboard_sources()
        for found in write_calls(
            path.read_text(encoding="utf-8"), str(path.relative_to(DASHBOARD.parents[1]))
        )
    ]

    assert hits == [], "the dashboard must not write:\n" + "\n".join(hits)


# The detector has to be shown to fire, or a silent bug in it reads as a pass.
# The first case is the cache `qualis.py` carried until 2026-11-06, verbatim.
@pytest.mark.parametrize(
    "snippet",
    [
        "reference.to_parquet(cache, index=False)",
        'df.to_csv("out.csv", index=False)',
        "df.to_csv(path_or_buf=target)",
        "frame.to_sql('lit_articles', engine)",
        "session.add(row)",
        "gold_session.add(row)",
        "self.session.flush()",
        "session.add_all(rows)",
        "session.commit()",
        "session.delete(row)",
        "session.merge(PublicationState(id=1, active_version_id=v))",
        'conn.execute(text("DELETE FROM lit_semantics"))',
        'conn.execute(text(f"TRUNCATE TABLE `{table}`"))',
        "conn.execute(update(table).values(x=1))",
        "from sqlalchemy import insert",
        'open(path, "w")',
        'open(path, mode="a", encoding="utf-8")',
        "open(path, mode)",
        'Path(cache).write_text("x")',
        'Path(cache).open("wb")',
        "cache.unlink(missing_ok=True)",
        "os.remove(cache)",
        "import shutil",
        "shutil.rmtree(target)",
        "import subprocess",
        "from subprocess import run",
    ],
)
def test_the_detector_flags_each_kind_of_write(snippet):
    assert write_calls(snippet), f"not flagged: {snippet}"


@pytest.mark.parametrize(
    "snippet",
    [
        "data = table[present].to_csv(index=False).encode('utf-8')",
        "frame.to_json(orient='records')",
        "left.merge(right, on='doi', how='left')",
        "pd.merge(left, right, on=['year', 'theme_label'])",
        "seen.add(doi)",
        "st.cache_data.clear()",
        'conn.execute(text("SELECT COUNT(*) FROM `lit_articles`")).scalar()',
        "conn.execute(select(t).order_by(t.c.created_at.desc()))",
        "open(path)",
        'open(path, "rb")',
        "frame.rename(columns={'a': 'b'})",
        "title.replace('-', ' ')",
        '"""The parquet cache used to call to_parquet() here."""',
    ],
)
def test_the_detector_lets_reads_through(snippet):
    assert write_calls(snippet) == []
