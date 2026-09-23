"""WP-24: the forward crawl must resume, and coverage must count the crawl.

Two defects of the same shape. `works[:max_works]` was the same leading slice
on every invocation, so the crawl re-requested what it already had instead of
advancing. And `citation_graph_coverage` inferred forward coverage from edge
presence, which silently excludes every work nobody cites -- for which the
answer "zero citing works" is known, and usable.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from sqlalchemy import select

from lake_research_map.db.bronze_models import CitationEdge, ExternalWork
from lake_research_map.ingest.openalex import citation_graph_coverage

OBSERVED = datetime(2026, 9, 22, tzinfo=UTC).replace(tzinfo=None)


def _work(session, work_id, **kwargs):
    session.add(
        ExternalWork(
            provider="openalex",
            provider_work_id=work_id,
            doi=f"10.1/{work_id.lower()}",
            first_observed_at=OBSERVED,
            last_observed_at=OBSERVED,
            **kwargs,
        )
    )
    session.commit()


def _args(**overrides):
    base = {
        "max_works": 2,
        "delay": 0.0,
        "max_pages": 1,
        "refresh_all": False,
        "commit_every": 25,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch_crawl(monkeypatch, results, calls):
    """Answer the batched entry point one work at a time, recording the order."""

    def _batch(work_ids, **_kwargs):
        answered = {}
        for work_id in work_ids:
            calls.append(work_id)
            result = results.get(work_id, {"citing_work_ids": [], "truncated": False})
            answered[work_id] = {"throttled": False, "error": None, **result}
        return answered

    monkeypatch.setattr("lake_research_map.ingest.openalex.fetch_openalex_citing_batch", _batch)


def test_a_second_run_advances_instead_of_recrawling(bronze_session, monkeypatch):
    from lake_research_map import pipeline as pipeline_module

    for work_id in ("W1", "W2", "W3", "W4"):
        _work(bronze_session, work_id)
    calls: list[str] = []
    _patch_crawl(
        monkeypatch, {"W1": {"citing_work_ids": ["W9"], "truncated": False, "pages": 1}}, calls
    )

    first = pipeline_module._refresh_citation_edges(bronze_session, _args())
    assert first["works_crawled"] == 2
    assert first["remaining"] == 2

    second = pipeline_module._refresh_citation_edges(bronze_session, _args())

    assert second["already_crawled"] == 2
    assert second["works_crawled"] == 2
    assert second["remaining"] == 0
    assert calls == ["W1", "W2", "W3", "W4"]


def test_a_work_nobody_cites_is_not_retried_forever(bronze_session, monkeypatch):
    """Zero citing works is an answer, not a missing result."""
    from lake_research_map import pipeline as pipeline_module

    _work(bronze_session, "W1")
    calls: list[str] = []
    _patch_crawl(monkeypatch, {}, calls)

    pipeline_module._refresh_citation_edges(bronze_session, _args(max_works=5))
    pipeline_module._refresh_citation_edges(bronze_session, _args(max_works=5))

    assert calls == ["W1"]
    work = bronze_session.scalar(select(ExternalWork))
    assert work.citing_crawled_at is not None
    assert work.citing_truncated is False
    assert bronze_session.scalars(select(CitationEdge.id)).all() == []


def test_truncation_is_recorded_on_the_work(bronze_session, monkeypatch):
    from lake_research_map import pipeline as pipeline_module

    _work(bronze_session, "W1")
    _patch_crawl(
        monkeypatch,
        {"W1": {"citing_work_ids": ["W7"], "truncated": True, "pages": 5}},
        [],
    )

    stats = pipeline_module._refresh_citation_edges(bronze_session, _args(max_works=5))

    assert stats["truncated_works"] == 1
    assert bronze_session.scalar(select(ExternalWork)).citing_truncated is True


def test_coverage_counts_a_crawled_work_with_no_citing_edges(bronze_session):
    """The fix that matters: an uncited work is covered, not missing."""
    _work(bronze_session, "W1", citing_crawled_at=OBSERVED, citing_truncated=False)
    bronze_session.add(
        CitationEdge(
            provider="openalex",
            citing_work_id="W1",
            cited_work_id="Wx",
            observed_at=OBSERVED,
            discovered_via="referenced_works",
        )
    )
    bronze_session.commit()

    coverage = citation_graph_coverage(bronze_session, ["W1"])

    # No incoming edge exists, yet the forward answer for W1 is known.
    assert coverage["with_forward_edges"] == 0
    assert coverage["with_forward"] == 1
    assert coverage["backward_coverage"] == 1.0
    assert coverage["usable_for_disruption"] == 1


def test_a_truncated_work_is_excluded_from_the_usable_population(bronze_session):
    _work(bronze_session, "W1", citing_crawled_at=OBSERVED, citing_truncated=True)
    bronze_session.add_all(
        [
            CitationEdge(
                provider="openalex",
                citing_work_id="W1",
                cited_work_id="Wx",
                observed_at=OBSERVED,
                discovered_via="referenced_works",
            ),
            CitationEdge(
                provider="openalex",
                citing_work_id="Wy",
                cited_work_id="W1",
                observed_at=OBSERVED,
                discovered_via="cites_query",
            ),
        ]
    )
    bronze_session.commit()

    coverage = citation_graph_coverage(bronze_session, ["W1"])

    # Both directions are present, but the forward tail was cut off, so a
    # disruption index computed over it would be wrong.
    assert coverage["with_backward"] == 1
    assert coverage["with_forward"] == 1
    assert coverage["truncated"] == 1
    assert coverage["usable_for_disruption"] == 0
