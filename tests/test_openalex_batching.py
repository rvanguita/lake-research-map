"""One request per 50 DOIs instead of one per DOI.

The quota is 1,000 requests per window, so asking per DOI made the 3,115-DOI
corpus a four-window, roughly 22-hour job. Batching makes it 63 requests. It is
also the polite way to ask: the same data for a sixtieth of the load.
"""

from __future__ import annotations

from lake_research_map.db.bronze_models import EnrichmentObservation
from lake_research_map.ingest import openalex
from lake_research_map.ingest.openalex import OPENALEX_FILTER_BATCH, fetch_openalex_batch


class _Ok:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, works):
        self._works = works

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": self._works, "meta": {}}


def _work(doi, work_id):
    return {
        "id": work_id,
        "doi": f"https://doi.org/{doi}",
        "display_name": "A work",
        "publication_year": 2020,
        "cited_by_count": 7,
        "referenced_works": ["W100", "W101"],
        "counts_by_year": [{"year": 2021, "cited_by_count": 3}],
        "open_access": {"is_oa": True, "oa_status": "gold"},
    }


def test_one_request_carries_the_whole_chunk():
    seen = []

    def _get(url, params=None, headers=None, timeout=None):
        seen.append(params)
        return _Ok([_work("10.1/a", "Wa"), _work("10.1/b", "Wb")])

    results = fetch_openalex_batch(["10.1/a", "10.1/B"], session_factory=_get)

    assert len(seen) == 1, "two DOIs must cost one request"
    assert seen[0]["filter"] == "doi:https://doi.org/10.1/a|https://doi.org/10.1/b"
    # The response DOI is normalized back to the key the caller asked for.
    assert set(results) == {"10.1/a", "10.1/b"}
    assert results["10.1/a"]["status"] == "success"
    assert results["10.1/a"]["citation_count"] == 7
    assert results["10.1/a"]["reference_count"] == 2
    assert len(results["10.1/a"]["response_sha256"]) == 64


def test_a_doi_the_response_omits_is_not_found_rather_than_an_error():
    """OpenAlex omits unknown works from a filtered result.

    Calling that an error would retry it on every later run, forever.
    """
    results = fetch_openalex_batch(
        ["10.1/a", "10.1/missing"],
        session_factory=lambda *a, **k: _Ok([_work("10.1/a", "Wa")]),
    )

    assert results["10.1/a"]["status"] == "success"
    assert results["10.1/missing"]["status"] == "not_found"


def test_a_throttled_batch_fails_every_doi_in_it(monkeypatch):
    """One verdict for the chunk is what lets the breaker see a throttle."""

    class _TooMany:
        status_code = 429
        headers = {"Retry-After": "19587"}

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    results = fetch_openalex_batch(
        ["10.1/a", "10.1/b", "10.1/c"], session_factory=lambda *a, **k: _TooMany()
    )

    assert {r["status"] for r in results.values()} == {"rate_limited"}
    assert all(r["http_status"] == 429 for r in results.values())


def test_the_refresh_persists_everything_a_batch_returned(bronze_session, monkeypatch):
    dois = [f"10.1000/{index}" for index in range(4)]
    monkeypatch.setattr(
        openalex,
        "fetch_openalex_batch",
        lambda chunk, **kwargs: {
            doi: {
                "doi": doi,
                "provider_work_id": "W" + doi.rsplit("/", 1)[-1],
                "status": "success",
                "http_status": 200,
                "citation_count": 1,
                "reference_count": 2,
                "response_sha256": "d" * 64,
                "retry_count": 0,
                "payload": {
                    "id": "W" + doi.rsplit("/", 1)[-1],
                    "counts_by_year": [],
                    "referenced_works": [],
                },
                "error_message": None,
            }
            for doi in chunk
        },
    )

    stats = openalex.refresh_openalex_observations(
        bronze_session, dois, max_fetch=4, delay=0, batch_size=2
    )

    assert stats["success"] == 4
    assert len(bronze_session.scalars(openalex.select(EnrichmentObservation.id)).all()) == 4


def _failing_batches(calls, **extra):
    def _batch(chunk, **kwargs):
        calls.append(len(chunk))
        return {
            doi: {
                "doi": doi,
                "status": "rate_limited",
                "http_status": 429,
                "retry_count": 3,
                "error_message": "OpenAlex returned HTTP 429",
                **extra,
            }
            for doi in chunk
        }

    return _batch


def test_an_exhausted_quota_stops_on_its_first_request(bronze_session, monkeypatch):
    """A Retry-After beyond the cap means no retry in this run can succeed."""
    calls: list[int] = []
    monkeypatch.setattr(openalex, "fetch_openalex_batch", _failing_batches(calls, futile=True))

    stats = openalex.refresh_openalex_observations(
        bronze_session, [f"10.1000/{i}" for i in range(50)], max_fetch=50, delay=0, batch_size=10
    )

    assert calls == [10]
    assert stats["stopped_early"] == "rate_limited"


def test_one_failed_batch_is_one_failure_not_fifty(bronze_session, monkeypatch):
    """The breaker counts requests. Counting DOIs stopped a crawl on one 500."""
    calls: list[int] = []
    monkeypatch.setattr(openalex, "fetch_openalex_batch", _failing_batches(calls))

    stats = openalex.refresh_openalex_observations(
        bronze_session, [f"10.1000/{i}" for i in range(100)], max_fetch=100, delay=0, batch_size=10
    )

    assert calls == [10] * openalex.CONSECUTIVE_FAILURE_LIMIT
    assert stats["stopped_early"] == "rate_limited"


def test_the_default_batch_size_is_the_documented_filter_limit():
    assert OPENALEX_FILTER_BATCH == 50
