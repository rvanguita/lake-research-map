"""A 3,115-DOI crawl has to survive being run more than once.

The skip compared `observed_at` against *this run's* batch timestamp, which a
previous run can never equal, while the work list was always
`sorted(dois)[:max_fetch]` -- the same leading slice every time. At the default
`max_fetch=100` the crawl re-fetched the same first hundred DOIs on every
invocation and never reached the hundred-and-first.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lake_research_map.db.bronze_models import CitationYearCount, EnrichmentObservation
from lake_research_map.ingest import openalex


@pytest.fixture
def fake_fetch(monkeypatch):
    """Record every DOI requested and answer without touching the network."""
    calls: list[str] = []

    def _fetch(doi, **_kwargs):
        calls.append(doi)
        return {
            "doi": doi,
            "provider_work_id": f"W{doi[-1]}",
            "status": "success",
            "http_status": 200,
            "citation_count": 3,
            "reference_count": 7,
            "response_sha256": "a" * 64,
            "retry_count": 0,
            "payload": {
                "id": f"W{doi[-1]}",
                "display_name": "A work",
                "publication_year": 2020,
                "counts_by_year": [{"year": 2021, "cited_by_count": 2}],
                "referenced_works": [],
                "open_access": {"is_oa": True, "oa_status": "gold"},
            },
            "error_message": None,
        }

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _fetch)
    return calls


# These tests exercise the per-DOI path explicitly (`batch_size=1`). Resume,
# breaker and payload behaviour are orthogonal to how many DOIs share a
# request; the batched default has its own file.
DOIS = [f"10.1000/{index}" for index in range(6)]


def test_a_second_run_continues_instead_of_repeating_the_first_slice(bronze_session, fake_fetch):
    first = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=2, delay=0, batch_size=1
    )
    assert first["fetched"] == 2
    assert first["remaining"] == 4

    second = openalex.refresh_openalex_observations(
        bronze_session, DOIS, max_fetch=2, delay=0, batch_size=1
    )

    assert second["already_observed"] == 2
    assert second["fetched"] == 2
    assert second["remaining"] == 2
    # The crawl advanced; it did not re-request what run one already stored.
    assert fake_fetch == ["10.1000/0", "10.1000/1", "10.1000/2", "10.1000/3"]


def test_the_whole_corpus_is_reachable_across_runs(bronze_session, fake_fetch):
    for _ in range(3):
        openalex.refresh_openalex_observations(
            bronze_session, DOIS, max_fetch=2, delay=0, batch_size=1
        )

    observed = set(bronze_session.scalars(openalex.select(EnrichmentObservation.doi)).all())
    assert observed == set(DOIS)
    assert len(fake_fetch) == len(DOIS)


def test_refresh_all_re_observes_for_a_deliberate_snapshot(bronze_session, fake_fetch):
    openalex.refresh_openalex_observations(
        bronze_session, DOIS[:2], max_fetch=2, delay=0, batch_size=1
    )
    fake_fetch.clear()

    again = openalex.refresh_openalex_observations(
        bronze_session,
        DOIS[:2],
        max_fetch=2,
        delay=0,
        batch_size=1,
        refresh_all=True,
        observed_at=datetime(2027, 1, 1, tzinfo=UTC),
    )

    assert again["already_observed"] == 0
    assert fake_fetch == ["10.1000/0", "10.1000/1"]
    # Append-only: the earlier observation is still there beside the new one.
    assert len(bronze_session.scalars(openalex.select(EnrichmentObservation.id)).all()) == 4


def test_a_failed_observation_is_retried_rather_than_skipped(bronze_session, monkeypatch):
    attempts: list[str] = []

    def _failing(doi, **_kwargs):
        attempts.append(doi)
        return {
            "doi": doi,
            "status": "rate_limited",
            "http_status": 429,
            "retry_count": 3,
            "error_message": "OpenAlex returned HTTP 429",
        }

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _failing)
    openalex.refresh_openalex_observations(
        bronze_session, DOIS[:1], max_fetch=1, delay=0, batch_size=1
    )
    openalex.refresh_openalex_observations(
        bronze_session, DOIS[:1], max_fetch=1, delay=0, batch_size=1
    )

    # A 429 says nothing about the DOI, so it must not permanently exclude it.
    assert attempts == ["10.1000/0", "10.1000/0"]


def test_progress_survives_an_interrupt_midway(bronze_session, monkeypatch):
    calls: list[str] = []

    def _explode_on_the_fourth(doi, **_kwargs):
        calls.append(doi)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return {
            "doi": doi,
            "provider_work_id": f"W{doi[-1]}",
            "status": "success",
            "http_status": 200,
            "citation_count": 1,
            "reference_count": 1,
            "response_sha256": "b" * 64,
            "retry_count": 0,
            "payload": {"id": f"W{doi[-1]}", "counts_by_year": [], "referenced_works": []},
            "error_message": None,
        }

    monkeypatch.setattr(openalex, "fetch_openalex_observation", _explode_on_the_fourth)
    with pytest.raises(KeyboardInterrupt):
        openalex.refresh_openalex_observations(
            bronze_session, DOIS, max_fetch=6, delay=0, batch_size=1, commit_every=3
        )

    bronze_session.rollback()
    # The first committed batch survived; a single commit at the end would have
    # discarded all of it.
    kept = bronze_session.scalars(openalex.select(EnrichmentObservation.doi)).all()
    assert sorted(kept) == ["10.1000/0", "10.1000/1", "10.1000/2"]


def test_the_raw_payload_is_not_stored_unless_asked(bronze_session, fake_fetch):
    openalex.refresh_openalex_observations(
        bronze_session, DOIS[:1], max_fetch=1, delay=0, batch_size=1
    )
    row = bronze_session.scalars(openalex.select(EnrichmentObservation)).first()
    assert row.payload is None
    # The evidence it carries is still extracted into its own table.
    assert bronze_session.scalars(openalex.select(CitationYearCount.year)).all() == [2021]
    # And provenance survives without the body.
    assert row.response_sha256 == "a" * 64


def test_store_payload_keeps_the_body_when_explicitly_requested(bronze_session, fake_fetch):
    openalex.refresh_openalex_observations(
        bronze_session, DOIS[:1], max_fetch=1, delay=0, batch_size=1, store_payload=True
    )
    row = bronze_session.scalars(openalex.select(EnrichmentObservation)).first()
    assert row.payload["display_name"] == "A work"
