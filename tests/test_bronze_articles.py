from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.raw_models import BibEntry
from lake_research_map.transform.bronze_articles import (
    _split_bibtex_authors,
    _split_ieee_csv_authors,
    _to_int,
    build_bronze_articles,
    normalize_doi,
)


def test_normalize_doi_strips_https_prefix():
    assert (
        normalize_doi("https://doi.org/10.1016/j.ijepes.2020.106042")
        == "10.1016/j.ijepes.2020.106042"
    )


def test_normalize_doi_strips_http_dx_prefix():
    assert (
        normalize_doi("http://dx.doi.org/10.1109/TPWRS.2024.3418651")
        == "10.1109/tpwrs.2024.3418651"
    )


def test_normalize_doi_casefolds_bare_doi():
    assert normalize_doi("10.1109/TPWRS.2024.3418651") == "10.1109/tpwrs.2024.3418651"


def test_normalize_doi_handles_none_and_empty():
    assert normalize_doi(None) is None
    assert normalize_doi("") is None
    assert normalize_doi("   ") is None


def test_split_bibtex_authors_strips_braces_and_splits_on_and():
    assert _split_bibtex_authors("{Doe}, John and Smith, Jane") == ["Doe, John", "Smith, Jane"]


def test_split_bibtex_authors_handles_none():
    assert _split_bibtex_authors(None) == []


def test_split_ieee_csv_authors_splits_on_semicolon():
    assert _split_ieee_csv_authors("Doe, John; Smith, Jane") == ["Doe, John", "Smith, Jane"]


def test_split_ieee_csv_authors_handles_non_string():
    assert _split_ieee_csv_authors(None) == []
    assert _split_ieee_csv_authors(float("nan")) == []


def test_to_int_handles_float_strings_from_ieee_csv():
    assert _to_int("12.0") == 12


def test_to_int_handles_plain_int():
    assert _to_int(7) == 7


def test_to_int_handles_none_and_garbage():
    assert _to_int(None) is None
    assert _to_int("not a number") is None


def test_build_bronze_persists_publication_category_and_basis(raw_session, bronze_session):
    raw_session.add(
        BibEntry(
            source="elsevier",
            bib_key="conference-paper",
            entry_type="inproceedings",
            source_file="data/elsevier/results.bib",
            fields={
                "title": "Distribution planning method",
                "booktitle": "International Conference on Power Systems",
                "author": "Doe, John",
                "year": "2024",
            },
            doi="10.1000/conference-paper",
        )
    )
    raw_session.commit()

    stats = build_bronze_articles(raw_session, bronze_session, dataset_version_id="v1")

    row = bronze_session.query(BronzeArticle).one()
    assert (row.publication_category, row.publication_category_basis) == (
        "conference",
        "conference_record_type",
    )
    assert stats["categories"] == {"conference": 1}
