"""A crawl that has been told to stop must stop.

The 2026-09-22 corpus refresh collected 73 consecutive HTTP 429s while the loop
kept going, spending four requests and seven seconds of backoff per DOI against
an API that had already refused. Had it run to the end that was roughly 8,000
futile requests.
"""

from __future__ import annotations

import argparse

from lake_research_map.db.bronze_models import EnrichmentObservation, ExternalWork
from lake_research_map.ingest import openalex
from lake_research_map.ingest.openalex import (
    CONSECUTIVE_FAILURE_LIMIT,
    MAX_RETRY_AFTER_SECONDS,
    backoff_seconds,
)


class _Response:
    def __init__(self, status_code=429, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": [], "meta": {}}


def test_retry_after_is_preferred_over_the_guess():
    assert backoff_seconds(_Response(headers={"Retry-After": "12"}), 0) == 12.0


def test_retry_after_is_capped_so_a_header_cannot_park_the_process():
    huge = _Response(headers={"Retry-After": "86400"})
    assert backoff_seconds(huge, 0) == MAX_RETRY_AFTER_SECONDS


def test_a_missing_or_unparseable_header_falls_back_to_exponential():
    assert backoff_seconds(_Response(), 2) == 4.0
    assert (
        backoff_seconds(_Response(headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 1)
        == 2.0
    )


def _throttled_result(doi, **_kwargs):
    return {
        "doi": doi,
        "status": "rate_limited",
        "http_status": 429,
        "retry_count": 3,
        "error_message": "OpenAlex returned HTTP 429",
    }


def _ok_result(doi, **_kwargs):
    # Keyed on the whole DOI suffix: `doi[-1]` collided `10.1000/10` with
    # `10.1000/0` and tripped the access-observation uniqueness constraint.
    work_id = "W" + doi.rsplit("/", 1)[-1]
    return {
        "doi": doi,
        "provider_work_id": work_id,
        "status": "success",
        "http_status": 200,
        "citation_count": 1,
        "reference_count": 1,
        "response_sha256": "c" * 64,
        "retry_count": 0,
        "payload": {"id": work_id, "counts_by_year": [], "referenced_works": []},
        "error_message": None,
    }


# These tests exercise the per-DOI path explicitly (`batch_size=1`). Resume,
# breaker and payload behaviour are orthogonal to how many DOIs share a
# request; the batched default has its own file.
DOIS = [f"10.1000/{index}" for index in range(20)]


def test_the_batch_halts_after_five_consecutive_failures(bronze_session, monkeypatch):
    calls: list[str] = []

    def _always_throttled(doi, **kwargs):
        calls.append(doi)
        return _throttled_result(doi)

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _always_throttled)
    stats = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=20, delay=0, batch_size=1
    )

    assert len(calls) == CONSECUTIVE_FAILURE_LIMIT
    assert stats["stopped_early"] == "rate_limited"
    assert stats["success"] == 0
    # The failures themselves are still recorded -- they are evidence.
    assert len(bronze_session.scalars(openalex.select(EnrichmentObservation.id)).all()) == 5


def test_an_isolated_failure_does_not_trip_the_breaker(bronze_session, monkeypatch):
    calls: list[str] = []

    def _one_bad_apple(doi, **kwargs):
        calls.append(doi)
        return _throttled_result(doi) if len(calls) == 3 else _ok_result(doi)

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _one_bad_apple)
    stats = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=20, delay=0, batch_size=1
    )

    assert len(calls) == 20
    assert stats["stopped_early"] is None
    assert stats["success"] == 19


def test_progress_before_the_breaker_is_committed(bronze_session, monkeypatch):
    calls: list[str] = []

    def _good_then_dead(doi, **kwargs):
        calls.append(doi)
        return _ok_result(doi) if len(calls) <= 4 else _throttled_result(doi)

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _good_then_dead)
    stats = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=20, delay=0, batch_size=1, commit_every=2
    )

    assert stats["stopped_early"] == "rate_limited"
    assert stats["success"] == 4
    bronze_session.rollback()
    kept = bronze_session.scalars(
        openalex.select(EnrichmentObservation.doi).where(EnrichmentObservation.status == "success")
    ).all()
    assert len(kept) == 4


def test_the_remaining_count_reflects_what_still_needs_fetching(bronze_session, monkeypatch):
    monkeypatch.setattr(
        openalex, "fetch_openalex_observation", lambda doi, **k: _throttled_result(doi)
    )
    stats = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=20, delay=0, batch_size=1
    )

    # Nothing succeeded, so nothing was actually retired from the backlog.
    assert stats["remaining"] == len(DOIS)


def test_the_citing_crawl_survives_a_throttled_page(bronze_session):
    """A bare `raise_for_status()` used to kill a thousand-work crawl."""
    attempts: list[int] = []

    def _throttle_once(url, params=None, headers=None, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            return _Response(status_code=429, headers={"Retry-After": "0"})
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "headers": {},
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "results": [{"id": "W9", "referenced_works": ["W1"]}],
                    "meta": {"count": 1},
                },
            },
        )()

    result = openalex.fetch_openalex_citing_batch(
        ["W1"], session_factory=_throttle_once, max_pages=1
    )["W1"]

    assert result["citing_work_ids"] == ["W9"]
    assert result["throttled"] is False
    assert len(attempts) == 2


def test_a_persistently_throttled_work_is_reported_as_truncated(bronze_session):
    def _always_429(url, params=None, headers=None, timeout=None):
        return _Response(status_code=429, headers={"Retry-After": "0"})

    result = openalex.fetch_openalex_citing_batch(["W1"], session_factory=_always_429, max_pages=2)[
        "W1"
    ]

    # An empty forward set from a throttled crawl must never read as "uncited".
    assert result["citing_work_ids"] == []
    assert result["throttled"] is True
    assert result["truncated"] is True


def test_the_citing_batch_halts_after_five_throttled_works(bronze_session, monkeypatch):
    from datetime import UTC, datetime

    from lake_research_map import pipeline as pipeline_module

    now = datetime.now(UTC).replace(tzinfo=None)
    for index in range(20):
        bronze_session.add(
            ExternalWork(
                provider="openalex",
                provider_work_id=f"W{index:02d}",
                doi=f"10.1/{index}",
                first_observed_at=now,
                last_observed_at=now,
            )
        )
    bronze_session.commit()

    calls: list[list[str]] = []

    def _all_throttled(work_ids, **kwargs):
        calls.append(list(work_ids))
        return {
            work_id: {"citing_work_ids": [], "truncated": True, "throttled": True, "error": None}
            for work_id in work_ids
        }

    monkeypatch.setattr(
        "lake_research_map.ingest.openalex.fetch_openalex_citing_batch", _all_throttled
    )

    stats = pipeline_module._refresh_citation_edges(
        bronze_session,
        argparse.Namespace(
            max_works=20, delay=0.0, max_pages=1, refresh_all=False, commit_every=25, batch_size=1
        ),
    )

    assert len(calls) == CONSECUTIVE_FAILURE_LIMIT
    assert stats["stopped_early"] == "rate_limited"
    assert stats["throttled_works"] == CONSECUTIVE_FAILURE_LIMIT
    # The defect this guards: a throttled work used to be stamped as crawled
    # and truncated, so the resume skipped it forever.
    assert stats["works_crawled"] == 0
    assert all(work.citing_crawled_at is None for work in bronze_session.query(ExternalWork).all())


def test_a_wait_longer_than_the_cap_skips_retrying_entirely(monkeypatch):
    """Sleeping 60s to be refused again is three minutes bought for nothing.

    OpenAlex answered the exhausted quota with `Retry-After: 19587` -- 5.4
    hours. Retrying under that header cannot succeed, and doing it three times
    per DOI is what would make the circuit breaker take fifteen minutes to fire
    instead of five requests.
    """
    from lake_research_map.ingest.openalex import retrying_is_futile

    assert retrying_is_futile(_Response(headers={"Retry-After": "19587"})) is True
    assert retrying_is_futile(_Response(headers={"Retry-After": "5"})) is False
    assert retrying_is_futile(_Response()) is False

    slept: list[float] = []
    monkeypatch.setattr(openalex.time, "sleep", slept.append)
    monkeypatch.setattr(
        openalex.requests,
        "get",
        lambda *a, **k: _Response(status_code=429, headers={"Retry-After": "19587"}),
    )

    result = openalex.fetch_openalex_observation("10.1000/x", max_retries=3)

    assert result["status"] == "rate_limited"
    assert result["retry_count"] == 0
    assert slept == [], "a futile wait must not be slept through"


def test_a_short_wait_is_still_retried(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(openalex.time, "sleep", slept.append)
    monkeypatch.setattr(
        openalex.requests,
        "get",
        lambda *a, **k: _Response(status_code=429, headers={"Retry-After": "3"}),
    )

    result = openalex.fetch_openalex_observation("10.1000/x", max_retries=2)

    assert result["status"] == "rate_limited"
    assert slept == [3.0, 3.0]
