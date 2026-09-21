"""Persistent human review of cross-DOI near-duplicate candidates."""

from __future__ import annotations

import pytest

from lake_research_map import pipeline
from lake_research_map.dashboard import data as dashboard_data
from lake_research_map.db.gold_models import DuplicateOverride, DuplicatePair
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.duplicate_resolution import (
    DuplicateResolutionError,
    build_merge_plan,
    list_review_rows,
    normalized_pair,
    record_decision,
    undo_decision,
)


def _silver(doi: str) -> SilverArticle:
    return SilverArticle(
        doi=doi,
        sources=["ieee"],
        record_type="article",
        title=f"Article {doi}",
        authors=[],
        keywords=[],
    )


def _candidate(gold_session, doi_a: str, doi_b: str, similarity: float = 0.99) -> None:
    gold_session.add(DuplicatePair(doi_a=doi_a, doi_b=doi_b, similarity=similarity))
    gold_session.commit()


def _override(
    doi_a: str,
    doi_b: str,
    canonical: str | None,
    decision: str = "merge",
) -> DuplicateOverride:
    left, right = sorted((doi_a, doi_b))
    return DuplicateOverride(
        doi_a=left,
        doi_b=right,
        decision=decision,
        canonical_doi=canonical,
        reason="Reviewed",
    )


def test_normalized_pair_handles_urls_order_and_invalid_self_pair():
    assert normalized_pair("https://doi.org/10.2/B", " 10.1/A ") == ("10.1/a", "10.2/b")
    with pytest.raises(DuplicateResolutionError, match="distinct"):
        normalized_pair("10.1/a", "https://doi.org/10.1/A")


def test_decision_lifecycle_validation_and_dashboard_queue(
    gold_session, silver_session, monkeypatch
):
    silver_session.add_all(
        [_silver("10.1/a"), _silver("10.1/b"), _silver("10.1/c"), _silver("10.1/d")]
    )
    silver_session.commit()
    _candidate(gold_session, "10.1/a", "10.1/b")

    kept = record_decision(
        gold_session,
        silver_session,
        decision="keep",
        doi_a="10.1/b",
        doi_b="10.1/a",
        reason="Distinct journal extension",
    )
    assert list_review_rows(gold_session) == []
    history = list_review_rows(gold_session, include_resolved=True)
    assert [row["kind"] for row in history] == ["resolved"]

    merged = record_decision(
        gold_session,
        silver_session,
        decision="merge",
        doi_a="10.1/a",
        doi_b="10.1/b",
        canonical_doi="10.1/b",
        reason="Same work after full-text review",
    )
    assert merged.id == kept.id
    assert merged.canonical_doi == "10.1/b"
    assert undo_decision(gold_session, doi_a="10.1/b", doi_b="10.1/a") is True
    assert [row["kind"] for row in list_review_rows(gold_session)] == ["candidate"]

    record_decision(
        gold_session,
        silver_session,
        decision="merge",
        doi_a="10.1/a",
        doi_b="10.1/b",
        canonical_doi="10.1/a",
        reason="Approved base merge",
    )
    _candidate(gold_session, "10.1/b", "10.1/c")
    with pytest.raises(DuplicateResolutionError, match="chains"):
        record_decision(
            gold_session,
            silver_session,
            decision="merge",
            doi_a="10.1/b",
            doi_b="10.1/c",
            canonical_doi="10.1/b",
            reason="Would create a merge chain",
        )
    assert gold_session.query(DuplicateOverride).count() == 1

    with pytest.raises(DuplicateResolutionError, match="review queue"):
        record_decision(
            gold_session,
            silver_session,
            decision="keep",
            doi_a="10.1/c",
            doi_b="10.1/d",
            reason="Not detected",
        )

    _candidate(gold_session, "10.1/c", "10.1/missing")
    with pytest.raises(DuplicateResolutionError, match="not found in Silver"):
        record_decision(
            gold_session,
            silver_session,
            decision="keep",
            doi_a="10.1/c",
            doi_b="10.1/missing",
            reason="Missing source row",
        )

    record_decision(
        gold_session,
        silver_session,
        decision="keep",
        doi_a="10.1/a",
        doi_b="10.1/b",
        reason="Reviewed again",
    )
    _candidate(gold_session, "10.1/c", "10.1/d", similarity=0.98)
    monkeypatch.setattr(dashboard_data, "get_engine", lambda _layer: gold_session.get_bind())

    result = dashboard_data.load_duplicate_pairs()

    unresolved = set(result[["doi_a", "doi_b"]].itertuples(index=False, name=None))
    assert ("10.1/a", "10.1/b") not in unresolved
    assert ("10.1/c", "10.1/d") in unresolved


def test_merge_plan_is_stable_and_rejects_chains():
    overrides = [
        _override("10.1/a", "10.1/c", "10.1/a"),
        _override("10.1/a", "10.1/b", "10.1/a"),
        _override("10.1/a", "10.1/missing", "10.1/a"),
    ]
    plan = build_merge_plan(overrides, {"10.1/a", "10.1/b", "10.1/c"})
    assert plan.groups == {"10.1/a": ("10.1/b", "10.1/c")}
    assert plan.inactive_overrides == 1

    with pytest.raises(DuplicateResolutionError, match="chains"):
        build_merge_plan(
            [
                _override("10.1/a", "10.1/b", "10.1/a"),
                _override("10.1/b", "10.1/c", "10.1/b"),
            ],
            {"10.1/a", "10.1/b", "10.1/c"},
        )


def test_cli_preserves_stage_routing_and_formats_domain_errors(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(pipeline, "run", called.append)
    pipeline.main(["--stage", "silver"])
    assert called == ["silver"]

    def fail(_args):
        raise DuplicateResolutionError("invalid reviewed pair")

    monkeypatch.setattr(pipeline, "_run_duplicate_command", fail)
    with pytest.raises(SystemExit) as exc_info:
        pipeline.main(["duplicates", "list"])
    assert exc_info.value.code == 2
    assert "invalid reviewed pair" in capsys.readouterr().err
