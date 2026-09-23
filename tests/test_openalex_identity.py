"""WP-07/WP-24: the credential gate and the contact address it turns on.

The refresh used to demand `OPENALEX_API_KEY`, which no OpenAlex account
issues for the public corpus, so a refresh was unreachable by construction
however the environment was configured. What the API actually wants is a
contact address for the polite pool.
"""

from __future__ import annotations

import argparse

import pytest

from lake_research_map.ingest.openalex import DEFAULT_USER_AGENT, _user_agent
from lake_research_map.pipeline import _require_openalex_identity


def test_a_contact_address_alone_is_enough(monkeypatch):
    monkeypatch.setenv("OPENALEX_EMAIL", "someone@example.org")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    assert _require_openalex_identity() == "someone@example.org"


def test_a_missing_contact_address_is_refused_with_the_reason(monkeypatch):
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    monkeypatch.setenv("OPENALEX_API_KEY", "irrelevant")

    with pytest.raises(ValueError, match="OPENALEX_EMAIL"):
        _require_openalex_identity()


def test_a_blank_contact_address_does_not_pass_as_configured(monkeypatch):
    monkeypatch.setenv("OPENALEX_EMAIL", "   ")

    with pytest.raises(ValueError, match="OPENALEX_EMAIL"):
        _require_openalex_identity()


def test_the_user_agent_carries_a_real_address_or_none_at_all(monkeypatch):
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    assert _user_agent(None) == DEFAULT_USER_AGENT
    assert "mailto:" not in _user_agent(None)

    assert "mailto:someone@example.org" in _user_agent("someone@example.org")

    monkeypatch.setenv("OPENALEX_EMAIL", "from-env@example.org")
    assert "mailto:from-env@example.org" in _user_agent(None)


def test_the_citation_crawl_refuses_a_bronze_layer_with_nothing_to_crawl(bronze_session):
    from lake_research_map.pipeline import _refresh_citation_edges

    args = argparse.Namespace(max_works=10, delay=0.0, max_pages=1)
    with pytest.raises(ValueError, match="refresh-openalex"):
        _refresh_citation_edges(bronze_session, args)


def test_the_citation_crawl_persists_edges_and_counts_truncation(bronze_session, monkeypatch):
    """The crawl, the store and the coverage audit had no caller between them.

    This is the wiring test: `fetch_openalex_citing_works` and
    `persist_incoming_edges` were implemented and unit-tested, but nothing in
    the CLI could reach either, so the package's capability existed only in
    the test suite.
    """
    from datetime import UTC, datetime

    from lake_research_map import pipeline as pipeline_module
    from lake_research_map.db.bronze_models import CitationEdge, ExternalWork
    from lake_research_map.ingest.openalex import citation_graph_coverage

    for work_id in ("W1", "W2"):
        bronze_session.add(
            ExternalWork(
                provider="openalex",
                provider_work_id=work_id,
                doi=f"10.1000/{work_id.lower()}",
                first_observed_at=datetime.now(UTC).replace(tzinfo=None),
                last_observed_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
    bronze_session.commit()

    crawled: dict[str, dict] = {
        "W1": {"citing_work_ids": ["W9", "W8"], "truncated": False, "pages": 1},
        "W2": {"citing_work_ids": ["W7"], "truncated": True, "pages": 5},
    }
    monkeypatch.setattr(
        "lake_research_map.ingest.openalex.fetch_openalex_citing_batch",
        lambda work_ids, **kwargs: {
            work_id: {"throttled": False, "error": None, **crawled[work_id]} for work_id in work_ids
        },
    )

    args = argparse.Namespace(
        max_works=10, delay=0.0, max_pages=None, refresh_all=False, commit_every=25
    )
    stats = pipeline_module._refresh_citation_edges(bronze_session, args)

    assert stats["works_crawled"] == 2
    assert stats["edges_inserted"] == 3
    # The truncated work is counted, not absorbed: an exhausted page budget
    # and a genuinely uncited work are otherwise indistinguishable.
    assert stats["truncated_works"] == 1

    edges = bronze_session.query(CitationEdge).all()
    assert {edge.discovered_via for edge in edges} == {"cites_query"}
    assert {edge.cited_work_id for edge in edges} == {"W1", "W2"}

    coverage = citation_graph_coverage(bronze_session, ["W1", "W2"])
    assert coverage["forward_coverage"] == 1.0
    # No reference lists were ingested, so nothing is usable for disruption
    # yet -- the intersection, not the larger direction, drives the gate.
    assert coverage["backward_coverage"] == 0.0
    assert coverage["usable_for_disruption"] == 0
