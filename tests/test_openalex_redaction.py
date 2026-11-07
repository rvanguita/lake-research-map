"""The OpenAlex API key reaches neither a log line nor the shared bronze database.

`requests` puts the full URL -- `api_key=` included -- into the message of
every exception it raises, and `ingest/openalex.py` logged that message and
persisted it verbatim as `error_message` in `bronze.lit_enrichment_observations`,
a table on a MySQL server shared with unrelated projects (Wave P0-2).
"""

from __future__ import annotations

import logging

import pytest
import requests
from sqlalchemy import select

from db.bronze_models import EnrichmentObservation
from ingest import openalex
from ingest.openalex import (
    _redact,
    fetch_openalex_batch,
    fetch_openalex_citing_batch,
    fetch_openalex_observation,
    refresh_openalex_observations,
)

LEAKY_URL = "https://api.openalex.org/works?filter=x&api_key=SECRET"


def _raise_connection_error(*_args, **_kwargs):
    raise requests.ConnectionError(LEAKY_URL)


@pytest.fixture
def no_sleep(monkeypatch):
    """Retries back off between attempts; the tests must not wait for them."""
    monkeypatch.setattr(openalex.time, "sleep", lambda _seconds: None)


@pytest.fixture
def key_in_environment(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRET")


def test_redact_masks_the_parameter_by_shape_and_the_key_by_value(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    assert _redact(LEAKY_URL) == "https://api.openalex.org/works?filter=x&api_key=<redacted>"
    assert _redact("credential SECRET rejected", "SECRET") == "credential <redacted> rejected"
    assert _redact("api_key=ONE&mailto=a api_key=TWO") == (
        "api_key=<redacted>&mailto=a api_key=<redacted>"
    )
    assert _redact(None) is None
    assert _redact("nothing to mask", None, "") == "nothing to mask"

    monkeypatch.setenv("OPENALEX_API_KEY", "FROMENV")
    assert _redact("token FROMENV") == "token <redacted>"


def test_a_failed_batch_request_logs_and_returns_a_redacted_error(no_sleep, caplog):
    caplog.set_level(logging.WARNING, logger="ingest.openalex")

    results = fetch_openalex_batch(
        ["10.1000/a"], session_factory=_raise_connection_error, api_key="SECRET", max_retries=0
    )

    result = results["10.1000/a"]
    assert result["status"] == "error"
    assert "SECRET" not in result["error_message"]
    assert "api_key=<redacted>" in result["error_message"]
    assert "SECRET" not in caplog.text
    assert "api_key=<redacted>" in caplog.text


def test_a_failed_single_doi_request_logs_and_returns_a_redacted_error(
    monkeypatch, no_sleep, caplog
):
    caplog.set_level(logging.WARNING, logger="ingest.openalex")
    monkeypatch.setattr(openalex.requests, "get", _raise_connection_error)

    result = fetch_openalex_observation("10.1000/a", api_key="SECRET", max_retries=0)

    assert result["status"] == "error"
    assert "SECRET" not in result["error_message"]
    assert "SECRET" not in caplog.text


def test_the_persisted_observation_carries_no_key(
    bronze_session, monkeypatch, no_sleep, key_in_environment, caplog
):
    caplog.set_level(logging.WARNING, logger="ingest.openalex")
    monkeypatch.setattr(openalex.requests, "get", _raise_connection_error)

    stats = refresh_openalex_observations(bronze_session, ["10.1000/a"], max_fetch=1, delay=0)

    row = bronze_session.scalars(select(EnrichmentObservation)).one()
    assert stats["success"] == 0
    assert row.status == "error"
    assert "SECRET" not in row.error_message
    assert "api_key=<redacted>" in row.error_message
    assert "SECRET" not in caplog.text


def test_a_key_outside_the_query_string_is_masked_by_value(no_sleep):
    def _echo_credential(*_args, **_kwargs):
        raise requests.ConnectionError("upstream rejected credential SECRET")

    results = fetch_openalex_batch(
        ["10.1000/a"], session_factory=_echo_credential, api_key="SECRET", max_retries=0
    )

    assert results["10.1000/a"]["error_message"] == "upstream rejected credential <redacted>"


def test_the_citing_crawl_reports_a_redacted_error(no_sleep):
    results = fetch_openalex_citing_batch(
        ["W1"], session_factory=_raise_connection_error, api_key="SECRET", max_pages=1
    )

    assert results["W1"]["truncated"] is True
    assert "SECRET" not in results["W1"]["error"]
    assert "api_key=<redacted>" in results["W1"]["error"]
