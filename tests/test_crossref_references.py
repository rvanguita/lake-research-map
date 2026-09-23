"""WP-23 via Crossref: reference lists, year lookups, and one list per work."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from lake_research_map.db.bronze_models import (
    CitationEdge,
    CrossrefReference,
    CrossrefReferenceList,
    DoiYear,
    EnrichmentObservation,
    ExternalWork,
    ReferenceWork,
)
from lake_research_map.ingest import crossref

NOW = datetime(2026, 9, 23, tzinfo=UTC).replace(tzinfo=None)
URL = "https://openalex.org/"


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, items, status_code=200, headers=None):
        self._items = items
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return {"message": {"items": self._items}}


def _item(doi, references=None, year=None):
    item = {"DOI": doi.upper()}
    if references is not None:
        item["reference"] = references
    if year is not None:
        item["issued"] = {"date-parts": [[year, 1, 1]]}
    return item


# -- parsing ------------------------------------------------------------------


def test_reference_years_are_parsed_from_what_publishers_actually_write():
    assert crossref.parse_reference_year("2019") == 2019
    assert crossref.parse_reference_year("2019a") == 2019
    assert crossref.parse_reference_year("in press") is None
    assert crossref.parse_reference_year("") is None
    assert crossref.parse_reference_year(None) is None
    assert crossref.parse_reference_year("1234") is None


# -- fetching -----------------------------------------------------------------


def test_one_request_carries_the_whole_list_batch():
    seen = []

    def _get(url, params=None, headers=None, timeout=None):
        seen.append(params)
        return _Response(
            [_item("10.1/a", [{"DOI": "10.9/X", "year": "2015"}, {"unstructured": "A book"}])]
        )

    result = crossref.fetch_crossref_reference_lists(
        ["10.1/a", "10.1/missing"], session_factory=_get, email="me@example.org"
    )

    assert len(seen) == 1
    assert seen[0]["filter"] == "doi:10.1/a,doi:10.1/missing"
    assert seen[0]["select"] == "DOI,reference,issued"
    assert seen[0]["mailto"] == "me@example.org"
    assert result["lists"]["10.1/a"] == [
        {"doi": "10.9/x", "year": 2015},
        {"doi": None, "year": None},
    ]
    # A DOI Crossref does not return is recorded as not found, not as empty.
    assert result["lists"]["10.1/missing"] is None


def test_no_contact_address_is_sent_without_consent(monkeypatch):
    monkeypatch.delenv("CROSSREF_EMAIL", raising=False)
    monkeypatch.setenv("OPENALEX_EMAIL", "openalex-only@example.org")
    seen = []

    def _get(url, params=None, headers=None, timeout=None):
        seen.append((params, headers))
        return _Response([])

    crossref.fetch_crossref_years(["10.1/a"], session_factory=_get)

    params, headers = seen[0]
    # Consent given to OpenAlex does not extend to Crossref.
    assert "mailto" not in params
    assert "openalex-only" not in headers["User-Agent"]


def test_a_rejected_query_fails_in_one_request():
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return _Response([], status_code=400)

    result = crossref.fetch_crossref_reference_lists(["10.1/a"], session_factory=_get)

    assert calls == [1]
    assert result["status"] == "error"


# -- collection ---------------------------------------------------------------


def _responder(lists=None, years=None):
    lists = lists or {}
    years = years or {}

    def _get(url, params=None, headers=None, timeout=None):
        dois = [part.split(":", 1)[1] for part in params["filter"].split(",")]
        if "reference" in params["select"]:
            return _Response([_item(d, lists[d]) for d in dois if d in lists])
        return _Response([_item(d, year=years[d]) for d in dois if d in years])

    return _get


def test_collection_stores_lists_then_dates_the_references_that_need_it(bronze_session):
    get = _responder(
        lists={
            "10.1/a": [{"DOI": "10.9/x", "year": "2010"}, {"DOI": "10.9/y"}, {"DOI": "10.1/b"}],
            "10.1/b": [],
        },
        years={"10.9/y": 2012},
    )

    stats = crossref.collect_crossref_references(
        bronze_session, ["10.1/a", "10.1/b", "10.1/c"], delay=0, session_factory=get
    )

    assert stats["lists_fetched"] == 3
    assert stats["lists_deposited"] == 1
    assert stats["lists_not_found"] == 1
    assert stats["references_stored"] == 3
    # 10.9/x carried its year inline; 10.1/b is a corpus DOI whose year is
    # already known; only 10.9/y needs a lookup.
    assert stats["years_requested"] == 1
    assert stats["years_found"] == 1
    row = bronze_session.scalar(select(DoiYear))
    assert (row.doi, row.year) == ("10.9/y", 2012)


def test_collection_resumes_without_refetching(bronze_session):
    calls = []
    get = _responder(lists={"10.1/a": [{"DOI": "10.9/y"}]}, years={"10.9/y": 2012})

    def _counting(url, params=None, headers=None, timeout=None):
        calls.append(params["select"])
        return get(url, params=params)

    crossref.collect_crossref_references(
        bronze_session, ["10.1/a"], delay=0, session_factory=_counting
    )
    calls.clear()
    stats = crossref.collect_crossref_references(
        bronze_session, ["10.1/a"], delay=0, session_factory=_counting
    )

    assert calls == []
    assert stats["lists_already"] == 1


def test_throttled_batches_are_not_recorded_and_the_breaker_stops(bronze_session):
    def _throttled(url, params=None, headers=None, timeout=None):
        return _Response([], status_code=429, headers={"Retry-After": "99999"})

    dois = [f"10.1/{index:03d}" for index in range(200)]
    stats = crossref.collect_crossref_references(
        bronze_session, dois, delay=0, list_batch=20, session_factory=_throttled
    )

    assert stats["stopped_early"] == "throttled"
    assert stats["requests"] == crossref.CONSECUTIVE_FAILURE_LIMIT
    assert bronze_session.scalars(select(CrossrefReferenceList)).all() == []


# -- coverage -----------------------------------------------------------------


def _list(session, doi, refs):
    session.add(
        CrossrefReferenceList(doi=doi, observed_at=NOW, status="success", deposited=len(refs))
    )
    for position, (ref_doi, year) in enumerate(refs):
        session.add(
            CrossrefReference(
                citing_doi=doi, position=position, reference_doi=ref_doi, year=year, observed_at=NOW
            )
        )


def _openalex_work(session, short, doi, refs=(), reference_count=None, year=2020):
    session.add(
        ExternalWork(
            provider="openalex",
            provider_work_id=URL + short,
            doi=doi,
            publication_year=year,
            first_observed_at=NOW,
            last_observed_at=NOW,
        )
    )
    session.add(
        EnrichmentObservation(
            provider="openalex",
            doi=doi,
            provider_work_id=URL + short,
            observed_at=NOW,
            status="success",
            reference_count=len(refs) if reference_count is None else reference_count,
        )
    )
    for cited in refs:
        session.add(
            CitationEdge(
                provider="openalex",
                citing_work_id=URL + short,
                cited_work_id=URL + cited,
                observed_at=NOW,
                discovered_via="referenced_works",
            )
        )


def test_a_crossref_deposit_is_used_even_when_openalex_has_the_work(bronze_session):
    """One list per work: splicing two enumerations would double count."""
    _list(bronze_session, "10.1/a", [("10.9/x", 2010), (None, None)])
    _openalex_work(bronze_session, "W1", "10.1/a", refs=("R1", "R2", "R3"))
    bronze_session.commit()

    cov = crossref.reference_year_coverage(bronze_session, {"10.1/a": 2020})

    assert cov["works_by_source"] == {"crossref": 1, "openalex": 0}
    assert cov["references"] == 2 and cov["dated"] == 1


def test_a_work_without_a_deposit_falls_back_to_openalex(bronze_session):
    bronze_session.add(
        CrossrefReferenceList(doi="10.1/a", observed_at=NOW, status="success", deposited=0)
    )
    _openalex_work(bronze_session, "W1", "10.1/a", refs=("R1", "R2"))
    bronze_session.add(
        ReferenceWork(
            provider="openalex",
            provider_work_id=URL + "R1",
            publication_year=2001,
            status="success",
            observed_at=NOW,
        )
    )
    bronze_session.commit()

    cov = crossref.reference_year_coverage(bronze_session, {"10.1/a": 2020})

    assert cov["works_by_source"] == {"crossref": 0, "openalex": 1}
    assert cov["references"] == 2 and cov["dated"] == 1


def test_lookups_and_corpus_years_date_references_without_an_inline_year(bronze_session):
    _list(bronze_session, "10.1/a", [("10.9/y", None), ("10.1/b", None), ("10.9/z", None)])
    bronze_session.add(
        DoiYear(provider="crossref", doi="10.9/y", year=2012, status="success", observed_at=NOW)
    )
    bronze_session.commit()

    cov = crossref.reference_year_coverage(bronze_session, {"10.1/a": 2020, "10.1/b": 2015})

    assert cov["dated"] == 2  # looked up, and a corpus work


def test_empty_lists_leave_the_denominator_and_missing_ones_are_counted(bronze_session):
    _openalex_work(bronze_session, "W1", "10.1/a", refs=(), reference_count=0)
    bronze_session.commit()

    cov = crossref.reference_year_coverage(bronze_session, {"10.1/a": 2020, "10.1/b": 2021})

    assert cov["known_empty"] == 1
    assert cov["unenumerated"] == 1
    assert cov["references"] == 0


def test_temporal_consistency_and_count_agreement_are_measured(bronze_session):
    _list(bronze_session, "10.1/a", [("10.9/x", 2010), ("10.9/y", 2025)])
    _openalex_work(bronze_session, "W1", "10.1/a", refs=("R1", "R2"))
    bronze_session.commit()

    cov = crossref.reference_year_coverage(bronze_session, {"10.1/a": 2020})

    assert cov["temporal_checked"] == 2
    assert cov["temporal_inconsistent"] == 1
    assert cov["count_pairs"] == 1 and cov["count_agreement"] == 1.0
