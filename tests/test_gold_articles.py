from lake_research_map.db.gold_models import Article as GoldArticle
from lake_research_map.db.gold_models import Chunk, DuplicateOverride
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.gold_articles import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    _build_abstract_text,
    _chunk_text,
    build_gold_articles,
)


def test_chunk_text_empty_returns_empty_list():
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


def test_chunk_text_short_text_is_a_single_chunk():
    text = "A short paragraph well under the chunking cap."
    assert _chunk_text(text) == [text]


def test_chunk_text_long_text_splits_into_multiple_overlapping_chunks():
    # Long enough to force at least two chunks, made of sentences so a
    # boundary break point exists near the target size.
    sentence = "This is a representative sentence about distribution planning. "
    text = sentence * 40  # well over CHUNK_MAX_CHARS
    chunks = _chunk_text(text, max_chars=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks overlap: the tail of one reappears near the head of the next
    assert chunks[1][:20] in text


def test_chunk_text_respects_default_constants():
    # sanity check the constants imported above are the ones actually used
    assert CHUNK_MAX_CHARS > CHUNK_OVERLAP_CHARS > 0


def _silver(**overrides) -> SilverArticle:
    fields = dict(
        doi="10.1109/example.1",
        sources=["ieee"],
        record_type="article",
        title="Example Title",
        authors=["Doe, John"],
        keywords=[],
        abstract=None,
    )
    fields.update(overrides)
    return SilverArticle(**fields)


def test_build_abstract_text_assembles_title_keywords_abstract_in_order():
    row = _silver(title="A Title", keywords=["grid", "planning"], abstract="An abstract.")
    text = _build_abstract_text(row)
    assert text.startswith("A Title")
    assert "Keywords: grid, planning" in text
    assert text.endswith("An abstract.")


def test_build_abstract_text_skips_missing_pieces():
    row = _silver(title="Only A Title", keywords=[], abstract=None)
    assert _build_abstract_text(row) == "Only A Title"


# ---------------------------------------------------------------------------
# Chunk reconciliation: an embedding costs a full `embed` stage to recompute,
# and losing one also orphans the `lit_semantics` row derived from it, so a
# `gold` rebuild must not throw vectors away for text that didn't change.
# ---------------------------------------------------------------------------


def _embed_everything(gold_session, vector=(0.1, 0.2, 0.3)):
    """Stand in for the `embed` stage without loading the real model."""
    for chunk in gold_session.query(Chunk).all():
        chunk.embedding = list(vector)
        chunk.embed_model = "test-model"
    gold_session.commit()


def test_build_gold_articles_preserves_embeddings_when_text_is_unchanged(
    silver_session, gold_session
):
    silver_session.add(_silver(doi="10.1109/example.1", abstract="An abstract."))
    silver_session.commit()

    build_gold_articles(silver_session, gold_session)
    _embed_everything(gold_session)

    stats = build_gold_articles(silver_session, gold_session)

    assert stats["chunks_unchanged"] == 1
    assert stats["chunks_invalidated"] == 0
    assert stats["chunks_new"] == 0
    assert stats["chunks_removed"] == 0
    assert stats["chunks_missing_embedding"] == 0

    chunk = gold_session.query(Chunk).one()
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert chunk.embed_model == "test-model"


def test_build_gold_articles_drops_the_vector_only_where_the_text_changed(
    silver_session, gold_session
):
    silver_session.add_all(
        [
            _silver(doi="10.1109/example.1", abstract="First abstract."),
            _silver(doi="10.1109/example.2", abstract="Second abstract."),
        ]
    )
    silver_session.commit()

    build_gold_articles(silver_session, gold_session)
    _embed_everything(gold_session)

    edited = silver_session.query(SilverArticle).filter_by(doi="10.1109/example.1").one()
    edited.abstract = "First abstract, revised."
    silver_session.commit()

    stats = build_gold_articles(silver_session, gold_session)

    assert stats["chunks_invalidated"] == 1
    assert stats["chunks_unchanged"] == 1
    assert stats["chunks_missing_embedding"] == 1

    revised = gold_session.query(Chunk).filter_by(doi="10.1109/example.1").one()
    untouched = gold_session.query(Chunk).filter_by(doi="10.1109/example.2").one()
    assert revised.embedding is None
    assert revised.embed_model is None
    assert "revised" in revised.text
    assert revised.char_len == len(revised.text)
    assert untouched.embedding == [0.1, 0.2, 0.3]


def test_build_gold_articles_removes_chunks_whose_article_is_gone(silver_session, gold_session):
    silver_session.add_all(
        [
            _silver(doi="10.1109/example.1", abstract="First abstract."),
            _silver(doi="10.1109/example.2", abstract="Second abstract."),
        ]
    )
    silver_session.commit()

    build_gold_articles(silver_session, gold_session)
    _embed_everything(gold_session)

    silver_session.query(SilverArticle).filter_by(doi="10.1109/example.2").delete()
    silver_session.commit()

    stats = build_gold_articles(silver_session, gold_session)

    assert stats["chunks_removed"] == 1
    assert stats["chunks_unchanged"] == 1
    assert [c.doi for c in gold_session.query(Chunk).all()] == ["10.1109/example.1"]


def test_build_gold_articles_applies_approved_merge_with_conservative_enrichment(
    silver_session, gold_session
):
    silver_session.add_all(
        [
            _silver(
                doi="10.1/canonical",
                sources=["ieee"],
                title="Canonical title",
                authors=["Primary Author"],
                keywords=["planning"],
                abstract=None,
                citation_count=5,
                reference_count=None,
                countries=["Brazil"],
            ),
            _silver(
                doi="10.1/duplicate-z",
                sources=["elsevier"],
                title="Alternate title Z",
                authors=["Primary Author", "Third Author"],
                keywords=["planning", "resilience-z"],
                abstract="The alternate abstract Z.",
                citation_count=12,
                reference_count=30,
                countries=["Portugal"],
            ),
            _silver(
                doi="10.1/duplicate-a",
                sources=["elsevier"],
                title="Alternate title A",
                authors=["Second Author"],
                keywords=["resilience-a"],
                abstract="The deterministic richer abstract.",
                citation_count=8,
                reference_count=20,
                countries=["Spain"],
            ),
        ]
    )
    silver_session.commit()
    build_gold_articles(silver_session, gold_session)
    _embed_everything(gold_session)
    gold_session.add_all(
        [
            DuplicateOverride(
                doi_a="10.1/canonical",
                doi_b="10.1/duplicate-z",
                decision="merge",
                canonical_doi="10.1/canonical",
                reason="Same work Z",
            ),
            DuplicateOverride(
                doi_a="10.1/canonical",
                doi_b="10.1/duplicate-a",
                decision="merge",
                canonical_doi="10.1/canonical",
                reason="Same work A",
            ),
        ]
    )
    gold_session.commit()

    stats = build_gold_articles(silver_session, gold_session)

    assert stats["articles"] == 1
    assert stats["duplicate_records_merged"] == 2
    assert stats["chunks_removed"] == 2
    assert stats["chunks_invalidated"] == 1
    article = gold_session.query(GoldArticle).one()
    assert article.doi == "10.1/canonical"
    assert article.title == "Canonical title"
    assert article.abstract == "The deterministic richer abstract."
    assert article.sources == ["ieee", "elsevier"]
    assert article.authors == ["Primary Author", "Second Author", "Third Author"]
    assert article.keywords == ["planning", "resilience-a", "resilience-z"]
    assert article.citation_count == 12
    assert article.reference_count == 30
    assert article.countries == ["Brazil", "Spain", "Portugal"]
    assert {chunk.doi for chunk in gold_session.query(Chunk).all()} == {"10.1/canonical"}

    _embed_everything(gold_session, vector=(0.4, 0.5, 0.6))
    gold_session.expunge_all()
    repeat_stats = build_gold_articles(silver_session, gold_session)
    repeated_chunk = gold_session.query(Chunk).one()
    assert repeat_stats["chunks_unchanged"] == 1
    assert repeat_stats["chunks_invalidated"] == 0
    assert repeat_stats["chunks_removed"] == 0
    assert repeated_chunk.embedding == [0.4, 0.5, 0.6]


def test_keep_and_undo_restore_duplicate_on_next_gold_build(silver_session, gold_session):
    silver_session.add_all(
        [
            _silver(doi="10.1/a", abstract="Shared abstract."),
            _silver(doi="10.1/b", abstract="Shared abstract."),
        ]
    )
    silver_session.commit()
    override = DuplicateOverride(
        doi_a="10.1/a",
        doi_b="10.1/b",
        decision="merge",
        canonical_doi="10.1/a",
        reason="Same work",
    )
    gold_session.add(override)
    gold_session.commit()
    build_gold_articles(silver_session, gold_session)
    assert gold_session.query(GoldArticle).count() == 1

    gold_session.delete(override)
    gold_session.commit()
    stats = build_gold_articles(silver_session, gold_session)

    assert stats["articles"] == 2
    assert stats["chunks_new"] == 1
    assert {article.doi for article in gold_session.query(GoldArticle).all()} == {
        "10.1/a",
        "10.1/b",
    }

    gold_session.expunge_all()
    gold_session.add(
        DuplicateOverride(
            doi_a="10.1/a",
            doi_b="10.1/b",
            decision="keep",
            canonical_doi=None,
            reason="Related but distinct",
        )
    )
    gold_session.commit()
    keep_stats = build_gold_articles(silver_session, gold_session)
    assert keep_stats["articles"] == 2
    assert keep_stats["duplicate_records_merged"] == 0
