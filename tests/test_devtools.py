"""Developer tools: the docs-on-main guard, the hero generator, the fingerprint exclusion."""

from __future__ import annotations

import pytest

from lake_research_map.devtools import hero
from lake_research_map.devtools.docs_branch_guard import blocks_commit


@pytest.mark.parametrize("branch", ["main", "master"])
def test_a_docs_only_commit_on_a_protected_branch_is_blocked(branch):
    assert blocks_commit(branch, ["docs/PRD.md", "README.md", "CLAUDE.md", "docs/SDD.md"])


@pytest.mark.parametrize(
    ("branch", "staged"),
    [
        ("main", ["docs/SDD.md", "src/lake_research_map/pipeline.py"]),
        ("main", []),
        ("main", [""]),
        ("docs/consolidation", ["docs/PRD.md"]),
        ("feat/x", ["README.md"]),
    ],
)
def test_mixed_empty_or_feature_branch_commits_pass(branch, staged):
    assert not blocks_commit(branch, staged)


def test_the_hero_svg_is_generated_without_retired_components():
    svg = hero.generate_svg()

    assert svg.lstrip().startswith("<svg")
    assert "Airflow" not in svg


def test_editing_a_developer_tool_does_not_mint_a_new_version(tmp_path):
    from lake_research_map.ingest.snapshots import code_fingerprint

    tool = tmp_path / "src/lake_research_map/devtools/hero.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("A = 1\n")
    _, first = code_fingerprint(tmp_path)
    tool.write_text("A = 2\n")
    _, second = code_fingerprint(tmp_path)

    assert first == second
