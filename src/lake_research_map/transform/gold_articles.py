"""Build lit_gold.articles + lit_gold.chunks from lit_silver.articles.

`articles` is the curated table a human/agent scans to pick a reference.
`chunks` is the RAG ingestion unit: every article gets an 'abstract' chunk
(title + abstract + keywords), and articles with a linked PDF (has_pdf) also
get 'fulltext' chunks extracted via pypdf and split into fixed-size windows.
`embedding`/`embed_model` are filled in by `transform/embeddings.py` (the
`embed` stage, run after this one).

Chunks are reconciled, not rebuilt: a chunk whose text is unchanged is left
untouched so it keeps its vector. This used to be a plain delete-and-rebuild,
which silently threw away every embedding on each `gold` run -- and with them
the basis of `lit_semantics`, leaving the semantic signals describing a corpus
state that no longer existed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from sqlalchemy import delete, func, null, select, update
from sqlalchemy.orm import Session

from lake_research_map.config import absolute_path
from lake_research_map.db.gold_models import Article as GoldArticle
from lake_research_map.db.gold_models import Chunk
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.duplicate_resolution import active_merge_plan
from lake_research_map.transform.publication_categories import preferred_classification

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200


def _build_abstract_text(row: SilverArticle) -> str:
    parts = [row.title or ""]
    if row.keywords:
        parts.append("Keywords: " + ", ".join(row.keywords))
    if row.abstract:
        parts.append(row.abstract)
    return "\n\n".join(p for p in parts if p).strip()


def _extract_pdf_text(pdf_path: str) -> str:
    # Paths are stored relative to the repo root (so the same file has the same
    # identity on the host and inside the Airflow container) -- resolve back to
    # an absolute path before actually opening it.
    try:
        reader = PdfReader(absolute_path(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        # pypdf can raise many different exception types on a malformed/
        # encrypted/truncated PDF; log so a failed extraction is diagnosable
        # instead of silently indistinguishable from "PDF has no text".
        logger.warning("_extract_pdf_text: failed to extract text from %r", pdf_path, exc_info=True)
        return ""
    text = "\n\n".join(pages)
    return re.sub(r"[ \t]+", " ", text).strip()


def _chunk_text(
    text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    min_boundary = max_chars // 2  # don't accept a break point too close to `start`
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break on a paragraph/sentence boundary rather than mid-word,
        # but only if that boundary is reasonably close to the target size --
        # otherwise a break right after `start` would produce tiny chunks.
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary - start < min_boundary:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary - start >= min_boundary:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _desired_chunks(row: SilverArticle) -> list[dict]:
    """Every chunk `row` should produce, in `(chunk_type, seq)` order.

    Split out so the reconciliation in `build_gold_articles` can compare the
    desired set against what is already stored without restating the
    abstract/PDF assembly rules.
    """
    chunks: list[dict] = []
    abstract_text = _build_abstract_text(row)
    if abstract_text:
        chunks.append({"seq": 0, "chunk_type": "abstract", "text": abstract_text})

    if row.has_pdf and row.pdf_path:
        full_text = _extract_pdf_text(row.pdf_path)
        for seq, text in enumerate(_chunk_text(full_text), start=1):
            chunks.append({"seq": seq, "chunk_type": "fulltext", "text": text})
    return chunks


def _union_values(rows: list[SilverArticle], field: str) -> list:
    """Union list metadata while preserving canonical-first display order."""
    values = []
    for row in rows:
        for value in getattr(row, field, None) or []:
            if value not in values:
                values.append(value)
    return values


@dataclass(frozen=True, slots=True)
class _GoldArticleSource:
    id: int | None
    doi: str
    sources: list
    publication_category: str
    publication_category_basis: str
    title: str | None
    authors: list
    year: int | None
    venue: str | None
    keywords: list
    abstract: str | None
    citation_count: int | None
    reference_count: int | None
    countries: list
    online_date: Any
    document_type: str | None
    license: str | None
    url: str | None
    has_pdf: bool
    pdf_path: str | None
    is_non_article: bool


def _merge_silver_group(
    primary: SilverArticle, duplicates: list[SilverArticle]
) -> _GoldArticleSource:
    """Build the conservative Gold representation of an approved merge.

    Identity and populated scalar values stay with the reviewer-selected
    canonical row. Duplicate records only fill gaps, contribute list members,
    and raise observed citation/reference counts.
    """
    rows = [primary, *sorted(duplicates, key=lambda row: row.doi)]

    def first_value(field: str):
        for row in rows:
            value = getattr(row, field, None)
            if value is not None and value != "":
                return value
        return None

    def maximum(field: str):
        values = [getattr(row, field, None) for row in rows]
        present = [value for value in values if value is not None]
        return max(present) if present else None

    pdf_row = next((row for row in rows if row.has_pdf and row.pdf_path), None)
    publication_category, publication_category_basis = preferred_classification(rows)
    return _GoldArticleSource(
        id=primary.id,
        doi=primary.doi,
        sources=_union_values(rows, "sources"),
        publication_category=publication_category,
        publication_category_basis=publication_category_basis,
        title=first_value("title"),
        authors=_union_values(rows, "authors"),
        year=first_value("year"),
        venue=first_value("venue"),
        keywords=_union_values(rows, "keywords"),
        abstract=first_value("abstract"),
        citation_count=maximum("citation_count"),
        reference_count=maximum("reference_count"),
        countries=_union_values(rows, "countries"),
        online_date=first_value("online_date"),
        document_type=first_value("document_type"),
        license=first_value("license"),
        url=first_value("url"),
        has_pdf=pdf_row is not None,
        pdf_path=pdf_row.pdf_path if pdf_row is not None else None,
        is_non_article=primary.is_non_article,
    )


# MySQL has a practical cap on how many values fit in one IN (...) clause, and
# a stale-chunk sweep can cover the whole corpus after a silver rebuild.
_DELETE_BATCH = 500


def build_gold_articles(silver_session: Session, gold_session: Session) -> dict:
    silver_rows = silver_session.scalars(select(SilverArticle)).all()
    by_doi = {row.doi: row for row in silver_rows}
    merge_plan = active_merge_plan(gold_session, set(by_doi))
    merged_duplicates = {doi for duplicates in merge_plan.groups.values() for doi in duplicates}

    # Articles are a plain delete-and-rebuild: nothing expensive is derived from
    # them, and `Chunk.doi` is a value-FK, so churning article ids breaks no
    # link. Chunks are reconciled instead -- see the module docstring.
    gold_session.query(GoldArticle).delete()

    # Deliberately without `Chunk.embedding`: that column is a ~384-float JSON
    # per row (tens of MB over the corpus) and the reconciliation only needs to
    # compare text.
    stored: dict[tuple[str, str, int], tuple[int, str]] = {
        (doi, chunk_type, seq): (chunk_id, text)
        for chunk_id, doi, chunk_type, seq, text in gold_session.execute(
            select(Chunk.id, Chunk.doi, Chunk.chunk_type, Chunk.seq, Chunk.text)
        )
    }

    n_articles = 0
    chunk_counts = {"abstract": 0, "fulltext": 0}
    unchanged = invalidated = added = 0
    seen: set[tuple[str, str, int]] = set()

    for silver_row in silver_rows:
        if silver_row.doi in merged_duplicates:
            continue
        duplicate_rows = [by_doi[doi] for doi in merge_plan.groups.get(silver_row.doi, ())]
        row = _merge_silver_group(silver_row, duplicate_rows) if duplicate_rows else silver_row
        gold_session.add(
            GoldArticle(
                doi=row.doi,
                sources=row.sources or [],
                publication_category=row.publication_category,
                publication_category_basis=row.publication_category_basis,
                title=row.title,
                authors=row.authors or [],
                year=row.year,
                venue=row.venue,
                keywords=row.keywords or [],
                abstract=row.abstract,
                citation_count=row.citation_count,
                reference_count=row.reference_count,
                countries=row.countries or [],
                online_date=row.online_date,
                document_type=row.document_type,
                license=row.license,
                url=row.url,
                has_pdf=row.has_pdf,
                pdf_path=row.pdf_path,
                silver_id=row.id,
                is_non_article=getattr(row, "is_non_article", False),
            )
        )
        n_articles += 1

        for chunk in _desired_chunks(row):
            key = (row.doi, chunk["chunk_type"], chunk["seq"])
            seen.add(key)
            chunk_counts[chunk["chunk_type"]] += 1
            existing = stored.get(key)

            if existing is None:
                gold_session.add(
                    Chunk(
                        doi=row.doi,
                        seq=chunk["seq"],
                        chunk_type=chunk["chunk_type"],
                        text=chunk["text"],
                        char_len=len(chunk["text"]),
                    )
                )
                added += 1
            elif existing[1] == chunk["text"]:
                # Left completely untouched -- this is what preserves the vector.
                unchanged += 1
            else:
                # The stored vector describes text that no longer exists, so it
                # has to go with it; `embed` will refill just these rows.
                gold_session.execute(
                    update(Chunk)
                    .where(Chunk.id == existing[0])
                    .values(
                        text=chunk["text"],
                        char_len=len(chunk["text"]),
                        # `null()`, not None: on a JSON column SQLAlchemy
                        # persists a bare None as JSON `null`, which is not SQL
                        # NULL. The JSON mirror is no longer written or read,
                        # but clearing it correctly keeps this legacy live-table
                        # path from leaving a stale vector behind changed text.
                        embedding=null(),
                        embedding_bin=None,
                        embed_model=None,
                    )
                )
                invalidated += 1

    stale_ids = [chunk_id for key, (chunk_id, _) in stored.items() if key not in seen]
    for start in range(0, len(stale_ids), _DELETE_BATCH):
        gold_session.execute(
            delete(Chunk).where(Chunk.id.in_(stale_ids[start : start + _DELETE_BATCH]))
        )

    gold_session.flush()

    # Reported so the caller can tell the operator that `embed` (and therefore
    # `semantic`) has work to do, instead of it being discovered later as a
    # silently stale dashboard.
    missing_embedding = (
        gold_session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
        )
        or 0
    )

    return {
        "articles": n_articles,
        "abstract_chunks": chunk_counts["abstract"],
        "fulltext_chunks": chunk_counts["fulltext"],
        "chunks_unchanged": unchanged,
        "chunks_invalidated": invalidated,
        "chunks_new": added,
        "chunks_removed": len(stale_ids),
        "chunks_missing_embedding": missing_embedding,
        "duplicate_records_merged": len(merged_duplicates),
        "duplicate_overrides_inactive": merge_plan.inactive_overrides,
    }
