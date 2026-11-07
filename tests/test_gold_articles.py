"""Gold: the per-article chunk/merge rules (`transform/gold_articles.py`) and the
builder that applies them to a candidate version (`versioned_gold.build_dataset_gold`)."""

from collections import Counter
from types import SimpleNamespace

from sqlalchemy import select

from db.gold_models import (
    DatasetArticle,
    DatasetChunk,
    DatasetVersion,
    DuplicateOverride,
    PublicationState,
)
from db.silver_models import Article as SilverArticle
from transform.gold_articles import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    PDF_CLEANING_STATS,
    PDF_STATS,
    _build_abstract_text,
    _chunk_text,
    _extract_pdf_text,
)
from transform.versioned_gold import build_dataset_gold


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
# Rebuilding a candidate. The stats classify every desired chunk against the
# text already stored (the candidate's own rows after a killed run, else the
# active version's): unchanged text is what `embed` copies a vector for instead
# of re-embedding, so a rebuild must report it as reused, not as new work.
# ---------------------------------------------------------------------------


def _candidate(gold_session, version_id: str = "v1", status: str = "candidate") -> None:
    gold_session.add(DatasetVersion(version_id=version_id, status=status))
    gold_session.commit()


def _chunks(gold_session, version_id: str = "v1") -> list[DatasetChunk]:
    return gold_session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == version_id)
        .order_by(DatasetChunk.doi, DatasetChunk.chunk_type, DatasetChunk.seq)
    ).all()


def _articles(gold_session, version_id: str = "v1") -> list[DatasetArticle]:
    return gold_session.scalars(
        select(DatasetArticle)
        .where(DatasetArticle.dataset_version_id == version_id)
        .order_by(DatasetArticle.doi)
    ).all()


def test_unchanged_rebuild_reuses_every_chunk_and_adds_no_rows(silver_session, gold_session):
    silver_session.add(_silver(doi="10.1109/example.1", abstract="An abstract."))
    silver_session.commit()
    _candidate(gold_session)

    first = build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()
    before = [(c.doi, c.chunk_type, c.seq, c.text_sha256) for c in _chunks(gold_session)]

    stats = build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()

    assert first["chunks_new"] == 1
    assert stats["chunks_reused"] == 1
    assert stats["chunks_invalidated"] == 0
    assert stats["chunks_new"] == 0
    # NFR-02: the same logical rows, not a second copy of them.
    assert [(c.doi, c.chunk_type, c.seq, c.text_sha256) for c in _chunks(gold_session)] == before
    assert len(_articles(gold_session)) == 1


def test_new_candidate_reuses_the_active_versions_unchanged_text(silver_session, gold_session):
    silver_session.add(_silver(doi="10.1109/example.1", abstract="An abstract."))
    silver_session.commit()
    _candidate(gold_session, "v1")
    build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.get(DatasetVersion, "v1").status = "active"
    # The build created the publication state row; publish v1 through it.
    gold_session.get(PublicationState, 1).active_version_id = "v1"
    gold_session.commit()
    _candidate(gold_session, "v2")

    stats = build_dataset_gold(silver_session, gold_session, "v2")

    assert (stats["chunks_reused"], stats["chunks_new"], stats["chunks_invalidated"]) == (1, 0, 0)
    # The published snapshot is left exactly as it was.
    assert [c.doi for c in _chunks(gold_session, "v1")] == ["10.1109/example.1"]


def test_rebuild_marks_only_the_chunk_whose_text_changed(silver_session, gold_session):
    silver_session.add_all(
        [
            _silver(doi="10.1109/example.1", abstract="First abstract."),
            _silver(doi="10.1109/example.2", abstract="Second abstract."),
        ]
    )
    silver_session.commit()
    _candidate(gold_session)
    build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()
    untouched_hash = _chunks(gold_session)[1].text_sha256

    edited = silver_session.query(SilverArticle).filter_by(doi="10.1109/example.1").one()
    edited.abstract = "First abstract, revised."
    silver_session.commit()

    stats = build_dataset_gold(silver_session, gold_session, "v1")

    assert stats["chunks_invalidated"] == 1
    assert stats["chunks_reused"] == 1
    revised, untouched = _chunks(gold_session)
    assert "revised" in revised.text
    assert revised.char_len == len(revised.text)
    assert untouched.text_sha256 == untouched_hash
    assert revised.text_sha256 != untouched_hash


def test_rebuild_drops_the_chunks_of_an_article_that_is_gone(silver_session, gold_session):
    silver_session.add_all(
        [
            _silver(doi="10.1109/example.1", abstract="First abstract."),
            _silver(doi="10.1109/example.2", abstract="Second abstract."),
        ]
    )
    silver_session.commit()
    _candidate(gold_session)
    build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()

    silver_session.query(SilverArticle).filter_by(doi="10.1109/example.2").delete()
    silver_session.commit()

    stats = build_dataset_gold(silver_session, gold_session, "v1")

    assert stats["articles"] == 1
    assert stats["chunks_reused"] == 1
    assert [c.doi for c in _chunks(gold_session)] == ["10.1109/example.1"]
    assert [a.doi for a in _articles(gold_session)] == ["10.1109/example.1"]


def test_approved_merge_keeps_the_canonical_record_and_only_fills_its_gaps(
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
    _candidate(gold_session)
    build_dataset_gold(silver_session, gold_session, "v1")
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

    stats = build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()

    assert stats["articles"] == 1
    assert stats["duplicate_records_merged"] == 2
    # The canonical abstract chunk now carries the filled-in abstract.
    assert stats["chunks_invalidated"] == 1
    (article,) = _articles(gold_session)
    assert article.doi == "10.1/canonical"
    assert article.title == "Canonical title"
    # Duplicates are visited in DOI order, so "duplicate-a" fills the gap first.
    assert article.abstract == "The deterministic richer abstract."
    assert article.sources == ["ieee", "elsevier"]
    assert article.authors == ["Primary Author", "Second Author", "Third Author"]
    assert article.keywords == ["planning", "resilience-a", "resilience-z"]
    assert article.citation_count == 12
    assert article.reference_count == 30
    assert article.countries == ["Brazil", "Spain", "Portugal"]
    assert {chunk.doi for chunk in _chunks(gold_session)} == {"10.1/canonical"}

    gold_session.expunge_all()
    repeat_stats = build_dataset_gold(silver_session, gold_session, "v1")
    assert repeat_stats["chunks_reused"] == 1
    assert repeat_stats["chunks_invalidated"] == 0
    assert repeat_stats["chunks_new"] == 0


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
    _candidate(gold_session)
    build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()
    assert [a.doi for a in _articles(gold_session)] == ["10.1/a"]

    gold_session.delete(override)
    gold_session.commit()
    stats = build_dataset_gold(silver_session, gold_session, "v1")
    gold_session.commit()

    assert stats["articles"] == 2
    assert stats["chunks_new"] == 1
    assert [a.doi for a in _articles(gold_session)] == ["10.1/a", "10.1/b"]

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
    keep_stats = build_dataset_gold(silver_session, gold_session, "v1")
    assert keep_stats["articles"] == 2
    assert keep_stats["duplicate_records_merged"] == 0


def test_chunk_text_breaks_on_sentences_once_lines_are_reflowed():
    from transform.pdf_text import reflow_lines

    # pypdf's shape: every sentence ends at a line end and there is no blank line.
    raw = "\n".join(f"Sentence {i} discusses distribution feeder planning." for i in range(60))

    chunks = _chunk_text(reflow_lines(raw), max_chars=300, overlap=40)

    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)


def _fake_pdf(monkeypatch, text: str) -> None:
    from transform import gold_articles

    page = SimpleNamespace(extract_text=lambda: text)
    monkeypatch.setattr(gold_articles, "PdfReader", lambda _path: SimpleNamespace(pages=[page]))


def test_extract_pdf_text_cleans_the_text_and_counts_what_changed(monkeypatch):
    body = "\n".join(f"Line {i} of the body about distri-\nbution planning." for i in range(30))
    _fake_pdf(monkeypatch, body + "\n12\nREFERENCES\n[1] A cited work.")
    stats: Counter = Counter()

    text = _extract_pdf_text("data/articles/paper.pdf", stats)

    assert "cited work" not in text
    assert "distribution planning." in text
    assert "\n" not in text
    assert stats == Counter(
        pdf_texts_cleaned=1,
        pdf_reference_sections_removed=1,
        pdf_hyphens_joined=30,
        pdf_short_lines_dropped=1,
    )


def test_pdf_linked_article_gets_cleaned_fulltext_chunks_and_the_stats_say_so(
    silver_session, gold_session, monkeypatch
):
    _fake_pdf(monkeypatch, "Body sentence.\n" * 300 + "References\n[1] Cited.")
    silver_session.add_all(
        [
            _silver(doi="10.1/pdf", has_pdf=True, pdf_path="data/articles/pdf.pdf"),
            _silver(doi="10.1/plain"),
        ]
    )
    silver_session.commit()
    _candidate(gold_session)

    stats = build_dataset_gold(silver_session, gold_session, "v1")

    assert {key: stats[key] for key in PDF_CLEANING_STATS} == {
        "pdf_texts_cleaned": 1,
        "pdf_reference_sections_removed": 1,
        "pdf_hyphens_joined": 0,
        "pdf_short_lines_dropped": 0,
    }
    chunks = _chunks(gold_session)
    fulltext = [c for c in chunks if c.chunk_type == "fulltext"]
    # One abstract chunk (seq 0) per article; full text only for the linked PDF,
    # numbered from 1, inside the window, and without the reference list.
    assert [(c.doi, c.seq) for c in chunks if c.chunk_type == "abstract"] == [
        ("10.1/pdf", 0),
        ("10.1/plain", 0),
    ]
    assert stats["fulltext_chunks"] == len(fulltext) > 1
    assert {c.doi for c in fulltext} == {"10.1/pdf"}
    assert [c.seq for c in fulltext] == list(range(1, len(fulltext) + 1))
    assert all(c.char_len == len(c.text) <= CHUNK_MAX_CHARS for c in fulltext)
    assert not any("Cited" in c.text for c in fulltext)
    assert stats["pdf_extraction_failures"] == 0


# ---------------------------------------------------------------------------
# No silent input loss (Wave P0-5d): a PDF pypdf cannot read is logged, counted
# in the run stats and leaves the article with its abstract chunk only --
# instead of looking like a corpus with one linked PDF fewer.
# ---------------------------------------------------------------------------


def _unreadable_pdf(monkeypatch) -> None:
    from transform import gold_articles

    def _raise(_path):
        raise OSError("truncated PDF")

    monkeypatch.setattr(gold_articles, "PdfReader", _raise)


def test_extract_pdf_text_counts_a_failed_extraction(monkeypatch):
    _unreadable_pdf(monkeypatch)
    stats: Counter = Counter()

    assert _extract_pdf_text("data/articles/broken.pdf", stats) == ""
    assert stats == Counter(pdf_extraction_failures=1)
    assert "pdf_extraction_failures" in PDF_STATS


def test_a_pdf_that_cannot_be_read_is_counted_and_leaves_only_the_abstract_chunk(
    silver_session, gold_session, monkeypatch
):
    _unreadable_pdf(monkeypatch)
    silver_session.add_all(
        [
            _silver(
                doi="10.1/broken",
                has_pdf=True,
                pdf_path="data/articles/broken.pdf",
                abstract="An abstract.",
            ),
            _silver(doi="10.1/plain", abstract="Another abstract."),
        ]
    )
    silver_session.commit()
    _candidate(gold_session)

    stats = build_dataset_gold(silver_session, gold_session, "v1")

    assert stats["pdf_extraction_failures"] == 1
    assert stats["pdf_texts_cleaned"] == 0
    assert stats["fulltext_chunks"] == 0
    assert [(c.doi, c.chunk_type) for c in _chunks(gold_session)] == [
        ("10.1/broken", "abstract"),
        ("10.1/plain", "abstract"),
    ]
