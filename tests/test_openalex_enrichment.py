"""Tests for OpenAlex automated bibliographic enrichment client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lake_research_map.ingest.openalex import (
    fetch_openalex_observation,
    fetch_openalex_work,
)


def test_fetch_openalex_work_parses_counts():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "cited_by_count": 42,
        "referenced_works": ["w1", "w2", "w3", "w4"],
    }

    with patch("requests.get", return_value=mock_resp):
        res = fetch_openalex_work("10.1109/TPWRS.2020.12345")
        assert res is not None
        assert res["citation_count"] == 42
        assert res["reference_count"] == 4


def test_fetch_openalex_work_handles_404():
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("requests.get", return_value=mock_resp):
        res = fetch_openalex_work("10.1109/NONEXISTENT")
        assert res is None


def test_fetch_openalex_observation_distinguishes_rate_limit():
    mock_resp = MagicMock()
    mock_resp.status_code = 429

    with patch("requests.get", return_value=mock_resp):
        result = fetch_openalex_observation("10.1000/limited", max_retries=0)

    assert result["status"] == "rate_limited"
    assert result["http_status"] == 429
