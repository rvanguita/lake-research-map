"""WP-24: incoming citation edges and the coverage gate they feed.

No network. `fetch_openalex_citing_works` takes its HTTP getter by injection
precisely so the pagination and truncation logic can be tested without calling
OpenAlex, and the corpus has no `OPENALEX_EMAIL` configured anyway.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lake_research_map.db.bronze_models import CitationEdge
from lake_research_map.ingest.openalex import (
    CITING_PAGE_SIZE,
    citation_graph_coverage,
    fetch_openalex_citing_works,
    persist_incoming_edges,
)

OBSERVED_AT = datetime(2026, 9, 22, tzinfo=UTC).replace(tzinfo=None)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _pager(pages):
    """Return a getter that serves `pages` in order and records its calls."""
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        return _FakeResponse(pages[len(calls) - 1])

    get.calls = calls
    return get


def test_citing_crawl_follows_cursors_and_stops_cleanly():
    get = _pager(
        [
            {"results": [{"id": "W1"}, {"id": "W2"}], "meta": {"next_cursor": "c2"}},
            {"results": [{"id": "W3"}], "meta": {"next_cursor": None}},
        ]
    )

    result = fetch_openalex_citing_works("W_target", session_factory=get)

    assert result["citing_work_ids"] == ["W1", "W2", "W3"]
    assert result["truncated"] is False
    assert result["pages"] == 2
    assert get.calls[0]["filter"] == "cites:W_target"
    assert get.calls[0]["per-page"] == CITING_PAGE_SIZE
    assert get.calls[1]["cursor"] == "c2"


def test_citing_crawl_reports_truncation_instead_of_implying_completeness():
    """A crawl that ran out of budget must not look like an uncited work.

    Without this flag an incomplete forward-citation set is indistinguishable
    from a complete one, and the disruption index WP-24 exists to enable would
    be computed on a silently missing tail.
    """
    endless = [
        {"results": [{"id": f"W{page}"}], "meta": {"next_cursor": f"c{page}"}} for page in range(10)
    ]

    result = fetch_openalex_citing_works("W_target", session_factory=_pager(endless), max_pages=3)

    assert result["pages"] == 3
    assert result["truncated"] is True
    assert len(result["citing_work_ids"]) == 3


def test_incoming_and_outgoing_edges_for_the_same_pair_coexist(bronze_session):
    """A cites B and B cites A are different facts, not a conflict."""
    bronze_session.add(
        CitationEdge(
            provider="openalex",
            citing_work_id="W_a",
            cited_work_id="W_b",
            observed_at=OBSERVED_AT,
            discovered_via="referenced_works",
        )
    )
    inserted = persist_incoming_edges(
        bronze_session, "W_a", ["W_b", "W_c", "W_a", "W_b"], OBSERVED_AT
    )
    bronze_session.commit()

    # W_a is skipped (self), W_b deduplicated against itself.
    assert inserted == 2
    edges = bronze_session.query(CitationEdge).all()
    assert len(edges) == 3
    assert {e.discovered_via for e in edges} == {"referenced_works", "cites_query"}
    # The outgoing A->B and the incoming B->A are both present and distinct.
    pairs = {(e.citing_work_id, e.cited_work_id) for e in edges}
    assert ("W_a", "W_b") in pairs
    assert ("W_b", "W_a") in pairs


def test_coverage_uses_the_intersection_not_the_larger_direction(bronze_session):
    """The disruption index needs both directions for the same work."""
    # W1 has references only; W2 has citers only; W3 has both.
    rows = [
        ("W1", "X1", "referenced_works"),
        ("W3", "X2", "referenced_works"),
        ("Y1", "W2", "cites_query"),
        ("Y2", "W3", "cites_query"),
    ]
    for citing, cited, via in rows:
        bronze_session.add(
            CitationEdge(
                provider="openalex",
                citing_work_id=citing,
                cited_work_id=cited,
                observed_at=OBSERVED_AT,
                discovered_via=via,
            )
        )
    bronze_session.commit()

    coverage = citation_graph_coverage(bronze_session, ["W1", "W2", "W3", "W4"])

    assert coverage["population"] == 4
    assert coverage["with_backward"] == 2  # W1, W3
    assert coverage["with_forward"] == 2  # W2, W3
    assert coverage["backward_coverage"] == 0.5
    assert coverage["forward_coverage"] == 0.5
    # Only W3 has both, so the usable population is 1 of 4 -- never 2 of 4.
    assert coverage["usable_for_disruption"] == 1
    assert coverage["disruption_coverage"] == 0.25


def test_coverage_of_an_empty_population_is_zero_not_undefined(bronze_session):
    coverage = citation_graph_coverage(bronze_session, [])
    assert coverage["population"] == 0
    assert coverage["usable_for_disruption"] == 0
