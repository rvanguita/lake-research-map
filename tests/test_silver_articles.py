from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.raw_models import PdfFile
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.silver_articles import (
    _merge_group,
    build_silver_articles,
    normalize_title,
)


def test_normalize_title_lowercases_and_strips_punctuation():
    assert normalize_title("Hosting Capacity: A Review!") == "hosting capacity a review"


def test_normalize_title_handles_none():
    assert normalize_title(None) == ""


def _bronze(**overrides) -> BronzeArticle:
    fields = dict(
        source="ieee",
        source_id="csv:1",
        record_type="article",
        doi="10.1109/example.1",
        title="Example Title",
        authors=["Doe, John"],
        year=2023,
        venue="IEEE Transactions",
        abstract="",
        keywords=[],
        citation_count=None,
        reference_count=None,
    )
    fields.update(overrides)
    return BronzeArticle(**fields)


def test_merge_group_single_record_is_not_a_duplicate_merge():
    group = [_bronze()]
    merged = _merge_group("10.1109/example.1", group)
    assert merged["sources"] == ["ieee"]
    assert merged["is_duplicate_merge"] is False


def test_merge_group_prefers_longest_abstract_and_flags_duplicate():
    ieee_record = _bronze(source="ieee", abstract="short")
    elsevier_record = _bronze(
        source="elsevier", abstract="a much longer abstract with more content"
    )
    merged = _merge_group("10.1109/example.1", [ieee_record, elsevier_record])

    assert merged["sources"] == ["elsevier", "ieee"]  # sorted
    assert merged["is_duplicate_merge"] is True
    assert merged["abstract"] == "a much longer abstract with more content"


def test_build_silver_articles_dedupes_by_doi_and_skips_missing_doi(
    bronze_session, silver_session, raw_session
):
    bronze_session.add_all(
        [
            _bronze(source="ieee", source_id="csv:1", doi="10.1109/example.1"),
            _bronze(source="elsevier", source_id="bib:1", doi="10.1109/example.1"),
            _bronze(source="ieee", source_id="csv:2", doi=None),
        ]
    )
    bronze_session.commit()

    stats = build_silver_articles(bronze_session, silver_session, raw_session)

    assert stats["written"] == 1
    assert stats["skipped_no_doi"] == 1

    silver_rows = silver_session.query(SilverArticle).all()
    assert len(silver_rows) == 1
    assert silver_rows[0].doi == "10.1109/example.1"
    assert silver_rows[0].is_duplicate_merge is True


def test_build_silver_preserves_highest_precedence_category_for_duplicate_doi(
    bronze_session, silver_session, raw_session
):
    bronze_session.add_all(
        [
            _bronze(
                source="ieee",
                source_id="csv:review",
                title="A systematic review of distribution planning",
                publication_category="review",
                publication_category_basis="title_review",
            ),
            _bronze(
                source="elsevier",
                source_id="bib:journal",
                publication_category="journal",
                publication_category_basis="journal_record_type",
            ),
        ]
    )
    bronze_session.commit()

    build_silver_articles(bronze_session, silver_session, raw_session)

    row = silver_session.query(SilverArticle).one()
    assert (row.publication_category, row.publication_category_basis) == (
        "review",
        "title_review",
    )


def test_build_silver_articles_links_matching_pdf(bronze_session, silver_session, raw_session):
    bronze_session.add(
        _bronze(
            source="ieee",
            source_id="csv:1",
            doi="10.1109/example.2",
            title="Deep Learning for Distribution System Planning",
        )
    )
    bronze_session.commit()

    raw_session.add(
        PdfFile(
            filename="Deep Learning for Distribution System Planning.pdf",
            path="data/articles/Deep Learning for Distribution System Planning.pdf",
            sha256="0" * 64,
            size_bytes=1024,
        )
    )
    raw_session.commit()

    stats = build_silver_articles(bronze_session, silver_session, raw_session)

    assert stats["has_pdf"] == 1
    row = silver_session.query(SilverArticle).one()
    assert row.has_pdf is True
    assert row.pdf_path == "data/articles/Deep Learning for Distribution System Planning.pdf"
