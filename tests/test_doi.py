"""`ingest/doi.py::normalize_doi`: the one comparison form of a DOI.

Every cross-source join, dedup and enrichment lookup goes through it, so these
are the spellings the two publishers and the enrichment providers actually use
(see CLAUDE.md "The two sources are not interchangeable").
"""

from __future__ import annotations

import pytest

from ingest.doi import normalize_doi


@pytest.mark.parametrize(
    "raw",
    [
        "10.1016/j.ijepes.2020.106042",  # IEEE-style bare DOI
        "https://doi.org/10.1016/j.ijepes.2020.106042",  # ScienceDirect
        "http://doi.org/10.1016/j.ijepes.2020.106042",
        "http://dx.doi.org/10.1016/j.ijepes.2020.106042",
        "https://dx.doi.org/10.1016/j.ijepes.2020.106042",
        "HTTPS://DOI.ORG/10.1016/J.IJEPES.2020.106042",  # prefix case is ignored too
        "  https://doi.org/10.1016/j.ijepes.2020.106042\n",
        "https://doi.org/ 10.1016/j.ijepes.2020.106042 ",  # whitespace after the prefix
    ],
)
def test_every_spelling_of_one_doi_normalizes_to_the_same_key(raw):
    assert normalize_doi(raw) == "10.1016/j.ijepes.2020.106042"


def test_a_bare_doi_is_casefolded():
    assert normalize_doi("10.1109/TPWRS.2024.3418651") == "10.1109/tpwrs.2024.3418651"


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n", "https://doi.org/", "https://doi.org/  "])
def test_no_doi_is_none(raw):
    # One value for "no DOI", which silver counts as `skipped_no_doi`.
    assert normalize_doi(raw) is None


def test_only_a_leading_resolver_prefix_is_stripped():
    # A DOI suffix may legitimately contain "doi.org/"; only the resolver goes.
    assert normalize_doi("10.1000/doi.org/x") == "10.1000/doi.org/x"
    assert normalize_doi("https://example.org/10.1000/x") == "https://example.org/10.1000/x"


def test_normalizing_twice_changes_nothing():
    once = normalize_doi("https://doi.org/10.1109/TPWRS.2024.3418651")

    assert normalize_doi(once) == once


def test_every_layer_uses_the_same_function():
    # The copies this replaced had to be kept in sync by hand.
    from ingest import crossref, enrichment, openalex
    from transform import bronze_articles, duplicate_resolution, screening_calibration

    for module in (bronze_articles, enrichment, openalex, crossref):
        assert module.normalize_doi is normalize_doi
    for module in (duplicate_resolution, screening_calibration):
        assert module.normalize_doi is normalize_doi
