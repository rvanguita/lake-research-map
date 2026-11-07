"""Tests for OpenAlex automated bibliographic enrichment client."""

from __future__ import annotations

import pytest

from ingest import openalex
from ingest.openalex import (
    fetch_openalex_observation,
    fetch_openalex_work,
)


class _Response:
    """The slice of `requests.Response` the single-DOI client reads."""

    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._body = body or {}
        self.text = ""

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def answer(monkeypatch):
    """Make every OpenAlex GET return `response`, and never sleep between retries.

    `fetch_openalex_observation` takes no `session_factory` (only the batch and
    citing-works fetchers do), so the injection point is the `requests.get`
    the module calls -- the same one `tests/test_openalex_throttling.py` uses.
    """

    def _answer(response: _Response) -> list[tuple]:
        calls: list[tuple] = []

        def _get(url, **kwargs):
            calls.append((url, kwargs))
            return response

        monkeypatch.setattr(openalex.requests, "get", _get)
        monkeypatch.setattr(openalex.time, "sleep", lambda _seconds: None)
        return calls

    return _answer


def test_fetch_openalex_work_parses_counts(answer):
    calls = answer(
        _Response(200, {"cited_by_count": 42, "referenced_works": ["w1", "w2", "w3", "w4"]})
    )

    res = fetch_openalex_work("10.1109/TPWRS.2020.12345")

    assert res is not None
    assert res["citation_count"] == 42
    assert res["reference_count"] == 4
    assert len(calls) == 1


def test_fetch_openalex_work_handles_404(answer):
    answer(_Response(404))

    res = fetch_openalex_work("10.1109/NONEXISTENT")

    assert res is None


def test_fetch_openalex_observation_distinguishes_rate_limit(answer):
    answer(_Response(429))

    result = fetch_openalex_observation("10.1000/limited", max_retries=0)

    assert result["status"] == "rate_limited"
    assert result["http_status"] == 429
