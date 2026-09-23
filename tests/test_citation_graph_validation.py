"""WP-23/WP-24 closure: batched graph collection, reference years, and the gates.

Three defects the first full-corpus audit exposed are pinned here: an empty
`counts_by_year` read as a gap when it is a known zero, a zero-reference work
read as uncovered when its reference list is known and empty, and a throttled
work stamped as crawled so the resume skipped it forever.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from lake_research_map.db.bronze_models import (
    AccessObservation,
    CitationEdge,
    CitationYearCount,
    EnrichmentObservation,
    ExternalWork,
    ReferenceWork,
)
from lake_research_map.ingest import openalex

NOW = datetime(2026, 9, 23, tzinfo=UTC).replace(tzinfo=None)
URL = "https://openalex.org/"


class _Page:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, results, count=None, next_cursor=None):
        self._payload = {"results": results, "meta": {"count": count, "next_cursor": next_cursor}}

    def json(self):
        return self._payload


def _work(session, short, doi, year=2020, **kw):
    session.add(
        ExternalWork(
            provider="openalex",
            provider_work_id=URL + short,
            doi=doi,
            publication_year=year,
            first_observed_at=NOW,
            last_observed_at=NOW,
            **kw,
        )
    )


def _observation(session, short, doi, citations=0, references=0):
    session.add(
        EnrichmentObservation(
            provider="openalex",
            doi=doi,
            provider_work_id=URL + short,
            observed_at=NOW,
            status="success",
            citation_count=citations,
            reference_count=references,
        )
    )


def _edge(session, citing, cited, via):
    # The two indexes are always collected by separate passes, so their edges
    # carry different timestamps -- and must, because edge uniqueness is
    # (provider, citing, cited, observed_at) and does not include the index.
    session.add(
        CitationEdge(
            provider="openalex",
            citing_work_id=URL + citing,
            cited_work_id=URL + cited,
            observed_at=NOW if via == "referenced_works" else NOW.replace(hour=1),
            discovered_via=via,
        )
    )


# -- batched citing crawl -----------------------------------------------------


def test_one_query_attributes_each_citer_to_the_batch_members_it_cites():
    seen = []

    def _get(url, params=None, headers=None, timeout=None):
        seen.append(params["filter"])
        return _Page(
            [
                {"id": URL + "C1", "referenced_works": [URL + "W1", URL + "W2", URL + "X9"]},
                {"id": URL + "C2", "referenced_works": [URL + "W2"]},
            ],
            count=2,
        )

    result = openalex.fetch_openalex_citing_batch([URL + "W1", URL + "W2"], session_factory=_get)

    assert seen == ["cites:W1|W2"], "two works must cost one query, in short-id form"
    assert result[URL + "W1"]["citing_work_ids"] == [URL + "C1"]
    assert result[URL + "W2"]["citing_work_ids"] == [URL + "C1", URL + "C2"]
    assert not result[URL + "W1"]["truncated"]


def test_truncation_is_decided_by_the_providers_own_count():
    """A crawl is complete when it holds `count` results, not when it stopped."""

    def _get(url, params=None, headers=None, timeout=None):
        return _Page(
            [{"id": URL + "C1", "referenced_works": [URL + "W1"]}], count=5, next_cursor="n"
        )

    result = openalex.fetch_openalex_citing_batch(
        [URL + "W1", URL + "W2"], session_factory=_get, max_pages=1
    )

    assert result[URL + "W1"]["truncated"] and result[URL + "W2"]["truncated"]


def test_a_rejected_filter_fails_loudly_in_one_request():
    calls = []

    class _Bad(_Page):
        status_code = 400

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return _Bad([])

    result = openalex.fetch_openalex_citing_batch([URL + "W1"], session_factory=_get)

    assert calls == [1]
    assert result[URL + "W1"]["error"] == "OpenAlex returned HTTP 400"


# -- reference years ----------------------------------------------------------


def test_reference_years_resolve_only_what_is_outside_the_corpus(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    _work(bronze_session, "W2", "10.1/b", year=2015)
    _edge(bronze_session, "W1", "W2", "referenced_works")  # in corpus: already dated
    _edge(bronze_session, "W1", "R1", "referenced_works")
    _edge(bronze_session, "W1", "R2", "referenced_works")
    bronze_session.commit()
    requested = []

    def _get(url, params=None, headers=None, timeout=None):
        requested.append(params["filter"])
        return _Page([{"id": URL + "R1", "publication_year": 1999}], count=1)

    stats = openalex.resolve_reference_years(bronze_session, delay=0, session_factory=_get)

    assert requested == ["ids.openalex:R1|R2"]
    assert stats["resolved"] == 1 and stats["not_found"] == 1 and stats["remaining"] == 0
    rows = {r.provider_work_id: r for r in bronze_session.scalars(select(ReferenceWork)).all()}
    assert rows[URL + "R1"].publication_year == 1999
    assert rows[URL + "R2"].status == "not_found"


def test_reference_resolution_resumes_and_does_not_store_throttles(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    for ref in ("R1", "R2", "R3"):
        _edge(bronze_session, "W1", ref, "referenced_works")
    bronze_session.commit()

    class _TooMany(_Page):
        status_code = 429
        headers = {"Retry-After": "99999"}

    openalex.resolve_reference_years(
        bronze_session, delay=0, batch_size=1, session_factory=lambda *a, **k: _TooMany([])
    )
    assert bronze_session.scalars(select(ReferenceWork)).all() == []

    stats = openalex.resolve_reference_years(
        bronze_session,
        delay=0,
        session_factory=lambda *a, **k: _Page(
            [{"id": URL + r, "publication_year": 2001} for r in ("R1", "R2", "R3")]
        ),
    )
    assert stats["resolved"] == 3


def test_hash_order_is_stable_and_not_the_sorted_order():
    ids = [f"W{index:05d}" for index in range(200)]
    assert openalex.hash_order(ids) == openalex.hash_order(list(reversed(ids)))
    assert openalex.hash_order(ids) != sorted(ids)


# -- coverage semantics -------------------------------------------------------


def test_a_never_cited_work_has_a_known_trajectory(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    _work(bronze_session, "W2", "10.1/b")
    _observation(bronze_session, "W1", "10.1/a", citations=4)
    _observation(bronze_session, "W2", "10.1/b", citations=0)
    bronze_session.add(
        CitationYearCount(
            provider="openalex",
            provider_work_id=URL + "W1",
            year=2021,
            citation_count=4,
            observed_at=NOW,
        )
    )
    bronze_session.commit()

    cov = openalex.citation_year_coverage(bronze_session)

    assert cov["with_series"] == 1
    assert cov["never_cited"] == 1
    assert cov["known_coverage"] == 1.0
    assert cov["unexplained"] == 0


def test_works_published_before_the_series_are_left_censored(bronze_session):
    _work(bronze_session, "W1", "10.1/a", year=2005)
    _work(bronze_session, "W2", "10.1/b", year=2018)
    for short, doi in (("W1", "10.1/a"), ("W2", "10.1/b")):
        _observation(bronze_session, short, doi, citations=3)
        bronze_session.add(
            CitationYearCount(
                provider="openalex",
                provider_work_id=URL + short,
                year=2012,
                citation_count=1,
                observed_at=NOW,
            )
        )
    bronze_session.commit()

    cov = openalex.citation_year_coverage(bronze_session)

    assert cov["series_start"] == 2012
    assert cov["left_censored"] == 1
    assert cov["complete_history"] == 1


def test_reference_year_coverage_counts_corpus_and_resolved_references(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    _work(bronze_session, "W2", "10.1/b", year=2010)
    _edge(bronze_session, "W1", "W2", "referenced_works")
    _edge(bronze_session, "W1", "R1", "referenced_works")
    _edge(bronze_session, "W1", "R2", "referenced_works")
    bronze_session.add(
        ReferenceWork(
            provider="openalex",
            provider_work_id=URL + "R1",
            publication_year=1990,
            status="success",
            observed_at=NOW,
        )
    )
    bronze_session.commit()

    cov = openalex.citation_year_coverage(bronze_session)

    assert cov["reference_pairs"] == 3
    assert cov["dated_reference_pairs"] == 2


def test_a_zero_reference_work_is_backward_covered(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    _work(bronze_session, "W2", "10.1/b")
    _observation(bronze_session, "W1", "10.1/a", references=0)
    _observation(bronze_session, "W2", "10.1/b", references=1)
    bronze_session.commit()

    coverage = openalex.citation_graph_coverage(bronze_session, [URL + "W1", URL + "W2"])

    # W1's reference list is known and empty; W2 reports one reference we hold
    # no edge for, so it stays uncovered.
    assert coverage["with_backward"] == 1
    assert coverage["with_backward_edges"] == 0


# -- integrity and access -----------------------------------------------------


def test_the_two_edge_indexes_are_cross_checked(bronze_session):
    _work(bronze_session, "W1", "10.1/a", citing_crawled_at=NOW, citing_truncated=False)
    _work(bronze_session, "W2", "10.1/b", citing_crawled_at=NOW, citing_truncated=False)
    _work(bronze_session, "W3", "10.1/c", citing_crawled_at=NOW, citing_truncated=False)
    _edge(bronze_session, "W1", "W2", "referenced_works")
    _edge(bronze_session, "W1", "W2", "cites_query")  # agrees
    _edge(bronze_session, "W1", "W3", "referenced_works")  # never re-found
    bronze_session.commit()

    integrity = openalex.citation_graph_integrity(bronze_session)

    assert integrity["self_loops"] == 0 and integrity["dangling_forward"] == 0
    assert integrity["backward_confirmed"] == 1 and integrity["backward_checkable"] == 2
    assert integrity["forward_confirmed"] == 1 and integrity["forward_checkable"] == 1
    assert integrity["agreement"] == 2 / 3


def test_a_truncated_crawl_is_not_held_against_the_index(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    _work(bronze_session, "W2", "10.1/b", citing_crawled_at=NOW, citing_truncated=True)
    _edge(bronze_session, "W1", "W2", "referenced_works")
    bronze_session.commit()

    assert openalex.citation_graph_integrity(bronze_session)["agreement"] is None


def test_an_open_licence_must_be_open_in_openalex(bronze_session):
    for short, doi, is_oa, lic in (
        ("W1", "10.1109/a", True, "cc-by"),
        ("W2", "10.1109/b", False, None),
        ("W3", "10.1109/c", True, None),
    ):
        _work(bronze_session, short, doi)
        bronze_session.add(
            AccessObservation(
                provider="openalex",
                provider_work_id=URL + short,
                observed_at=NOW,
                is_oa=is_oa,
                license=lic,
            )
        )
    bronze_session.commit()

    result = openalex.access_validation(
        bronze_session,
        # W3 is IEEE-copyright yet open (a green repository copy): not a contradiction.
        {"10.1109/a": "CCBY", "10.1109/b": "CCBYNCND", "10.1109/c": "IEEE"},
    )

    assert result["ieee_open_licensed"] == 2
    assert result["open_confirmed"] == 1
    assert result["contradictions"] == ["10.1109/b"]
    assert result["licence_pairs"] == 1 and result["licence_agreement"] == 1.0


# -- WP-23 closure: batch width and year validation --------------------------


def test_a_rejected_wide_filter_falls_back_to_the_proven_width(bronze_session):
    """100 values halve the requests; if the provider refuses, 50 must still finish."""
    _work(bronze_session, "W1", "10.1/a")
    refs = [f"R{index:03d}" for index in range(120)]
    for ref in refs:
        _edge(bronze_session, "W1", ref, "referenced_works")
    bronze_session.commit()
    widths = []

    class _Bad(_Page):
        status_code = 400

    def _get(url, params=None, headers=None, timeout=None):
        ids = params["filter"].split(":", 1)[1].split("|")
        widths.append(len(ids))
        if len(ids) > 50:
            return _Bad([])
        return _Page([{"id": URL + short, "publication_year": 2000} for short in ids])

    stats = openalex.resolve_reference_years(
        bronze_session, delay=0, batch_size=100, session_factory=_get
    )

    assert widths[0] == 100, "the wide filter is tried first"
    assert widths[1:] == [50, 50, 20]
    assert stats["resolved"] == 120 and stats["remaining"] == 0
    assert stats["stopped_early"] is None and stats["batch_size"] == 50


def test_the_wide_filter_is_kept_when_the_provider_accepts_it(bronze_session):
    _work(bronze_session, "W1", "10.1/a")
    for index in range(150):
        _edge(bronze_session, "W1", f"R{index:03d}", "referenced_works")
    bronze_session.commit()
    widths = []

    def _get(url, params=None, headers=None, timeout=None):
        ids = params["filter"].split(":", 1)[1].split("|")
        widths.append(len(ids))
        return _Page([{"id": URL + short, "publication_year": 2000} for short in ids])

    stats = openalex.resolve_reference_years(
        bronze_session, delay=0, batch_size=100, session_factory=_get
    )

    assert widths == [100, 50]
    assert stats["requests"] == 2


def test_provider_years_are_validated_against_corpus_metadata(bronze_session):
    _work(bronze_session, "W1", "10.1/a", year=2020)  # agrees
    _work(bronze_session, "W2", "10.1/b", year=2021)  # online-first gap: tolerated
    _work(bronze_session, "W3", "10.1/c", year=2010)  # wrong by a decade
    bronze_session.commit()

    cov = openalex.citation_year_coverage(
        bronze_session, {"10.1/a": 2020, "10.1/B": 2020, "10.1/c": 2020}
    )

    assert cov["year_agreement_pairs"] == 3
    assert cov["year_agreement"] == 2 / 3


def test_a_reference_newer_than_its_citer_is_counted_as_inconsistent(bronze_session):
    _work(bronze_session, "W1", "10.1/a", year=2015)
    _work(bronze_session, "W2", "10.1/b", year=2016)  # in press: allowed
    _work(bronze_session, "W3", "10.1/c", year=2019)  # impossible
    _work(bronze_session, "W4", "10.1/d", year=2001)
    for cited in ("W2", "W3", "W4"):
        _edge(bronze_session, "W1", cited, "referenced_works")
    bronze_session.commit()

    cov = openalex.citation_year_coverage(bronze_session)

    assert cov["temporal_checked"] == 3
    assert cov["temporal_inconsistent"] == 1


def test_validation_is_unavailable_rather_than_passing_when_nothing_is_comparable(bronze_session):
    _work(bronze_session, "W1", "10.1/a", year=2020)
    bronze_session.commit()

    cov = openalex.citation_year_coverage(bronze_session)

    assert cov["year_agreement"] is None
    assert cov["temporal_consistency"] is None
