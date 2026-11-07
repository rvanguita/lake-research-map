"""What one Silver article becomes in Gold: its chunks and its merged record.

The Gold builder (`transform/versioned_gold.py::build_dataset_gold`) writes a
candidate version's `lit_dataset_articles` / `lit_dataset_chunks` rows; this
module holds the per-article rules it applies, kept pure so they can be tested
without a version:

* chunks are the RAG ingestion unit: every article gets an 'abstract' chunk
  (title + keywords + abstract), and articles with a linked PDF (has_pdf) also
  get 'fulltext' chunks extracted via pypdf, cleaned (`transform/pdf_text.py`:
  reference list cut, page furniture dropped, line-break hyphens rejoined,
  lines reflowed into paragraphs) and split into fixed-size windows;
* an approved duplicate merge keeps the reviewer's canonical record and only
  fills its gaps from the duplicates (`_merge_silver_group`).

Gold stores chunk text and embedding provenance only; the vectors themselves
live in the version's Qdrant collection (ADR-08), filled by the `embed` stage.
Because a vector is a function of the chunk text, any change to these rules
changes chunk text and therefore what `embed` has to compute.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader

from config import absolute_path
from db.silver_models import Article as SilverArticle
from transform.pdf_text import clean_pdf_text
from transform.publication_categories import preferred_classification

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200
# What the PDF cleaning did, summed over a build and reported in its stats.
PDF_CLEANING_STATS = (
    "pdf_texts_cleaned",
    "pdf_reference_sections_removed",
    "pdf_hyphens_joined",
    "pdf_short_lines_dropped",
)
# Everything a build reports about PDFs: the cleaning above plus the PDFs pypdf
# could not read at all. Such an article keeps its abstract chunk only, and a
# run that lost full text has to say so in its stats instead of looking like a
# corpus with fewer linked PDFs.
PDF_STATS = (*PDF_CLEANING_STATS, "pdf_extraction_failures")


def _build_abstract_text(row: SilverArticle) -> str:
    parts = [row.title or ""]
    if row.keywords:
        parts.append("Keywords: " + ", ".join(row.keywords))
    if row.abstract:
        parts.append(row.abstract)
    return "\n\n".join(p for p in parts if p).strip()


def _extract_pdf_text(pdf_path: str, stats: Counter | None = None) -> str:
    # Paths are stored relative to the repo root (so the same file has the same
    # identity in every checkout and container) -- resolve back to
    # an absolute path before actually opening it.
    try:
        reader = PdfReader(absolute_path(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        # pypdf can raise many different exception types on a malformed/
        # encrypted/truncated PDF; log so a failed extraction is diagnosable
        # instead of silently indistinguishable from "PDF has no text".
        logger.warning("_extract_pdf_text: failed to extract text from %r", pdf_path, exc_info=True)
        if stats is not None:
            stats["pdf_extraction_failures"] += 1
        return ""
    text = re.sub(r"[ \t]+", " ", "\n\n".join(pages)).strip()
    cleaned, report = clean_pdf_text(text)
    if stats is not None:
        stats["pdf_texts_cleaned"] += 1
        stats["pdf_reference_sections_removed"] += int(report["reference_section_removed"])
        stats["pdf_hyphens_joined"] += int(report["hyphens_joined"])
        stats["pdf_short_lines_dropped"] += int(report["short_lines_dropped"])
    return cleaned


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


def _desired_chunks(row: SilverArticle, stats: Counter | None = None) -> list[dict]:
    """Every chunk `row` should produce, in `(chunk_type, seq)` order.

    Split out so the Gold builder can compare the desired set against what is
    already stored (for its unchanged / new / changed stats) without restating
    the abstract/PDF assembly rules.
    """
    chunks: list[dict] = []
    abstract_text = _build_abstract_text(row)
    if abstract_text:
        chunks.append({"seq": 0, "chunk_type": "abstract", "text": abstract_text})

    if row.has_pdf and row.pdf_path:
        full_text = _extract_pdf_text(row.pdf_path, stats)
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
        is_non_article=bool(primary.is_non_article),
    )
