"""Build lit_silver.articles from lit_bronze.articles: dedupe by normalized
DOI (the reliable cross-source join key -- see CLAUDE.md), attach quality
flags, and link each article to a PDF in data/articles/ via normalized/fuzzy
title matching (rapidfuzz) rather than exact string equality, since PDF
filenames are a lossy transform of the title (punctuation -> `-`).

Articles with no DOI at all are excluded here (logged, not silently
dropped) -- DOI is silver's primary key and the only dependable cross-source
identifier, so a record without one can't be deduped or safely carried
forward as a distinct row.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.raw_models import PdfFile
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.db.silver_models import RejectedArticle
from lake_research_map.transform.publication_categories import preferred_classification

PDF_MATCH_THRESHOLD = 85.0

_NON_ARTICLE_TYPES = {"incollection", "book", "inbook"}
_SHORT_ABSTRACT_THRESHOLD = 50  # characters


def _is_non_article(record_type: str | None, abstract: str | None) -> bool:
    """Flag records that are book front matter rather than research articles."""
    if record_type and record_type.lower() in _NON_ARTICLE_TYPES:
        if not abstract or len(abstract.strip()) < _SHORT_ABSTRACT_THRESHOLD:
            return True
    return False


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def _merge_group(doi: str, group: list[BronzeArticle]) -> dict:
    # Prefer the record with the most "content" (abstract length as proxy),
    # fall back to the first one deterministically.
    primary = max(group, key=lambda a: len(a.abstract or ""))
    sources = sorted({a.source for a in group})
    publication_category, publication_category_basis = preferred_classification(group)
    return {
        "doi": doi,
        "sources": sources,
        "record_type": primary.record_type,
        "publication_category": publication_category,
        "publication_category_basis": publication_category_basis,
        "title": primary.title,
        "authors": primary.authors or [],
        "year": primary.year,
        "venue": primary.venue,
        "volume": primary.volume,
        "issue": primary.issue,
        "pages": primary.pages,
        "url": primary.url,
        "abstract": primary.abstract,
        "keywords": primary.keywords or [],
        "citation_count": next(
            (a.citation_count for a in group if a.citation_count is not None), None
        ),
        "reference_count": next(
            (a.reference_count for a in group if a.reference_count is not None), None
        ),
        # Only the IEEE side of a merged group carries these, so take the
        # first record that actually has them rather than the primary's.
        "countries": next((a.countries for a in group if a.countries), []),
        "online_date": next((a.online_date for a in group if a.online_date), None),
        "document_type": next((a.document_type for a in group if a.document_type), None),
        "license": next((a.license for a in group if a.license), None),
        "has_abstract": bool(primary.abstract),
        "has_doi": True,
        "is_duplicate_merge": len(group) > 1,
        "bronze_ids": [a.id for a in group],
    }


def _link_pdfs(
    session: Session, silver_rows: list[SilverArticle], pdf_files: list[PdfFile]
) -> None:
    choices = {row.id: normalize_title(row.title) for row in silver_rows if row.title}
    if not choices:
        return

    for pdf in pdf_files:
        pdf_title_guess = normalize_title(pdf.filename.rsplit(".", 1)[0])
        match = process.extractOne(
            pdf_title_guess,
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=PDF_MATCH_THRESHOLD,
        )
        if match is None:
            continue
        _, score, article_id = match
        row = next(r for r in silver_rows if r.id == article_id)
        if row.pdf_match_score is not None and row.pdf_match_score >= score:
            continue
        row.has_pdf = True
        row.pdf_path = pdf.archive_path or pdf.path
        row.pdf_match_score = score


def build_silver_articles(
    bronze_session: Session,
    silver_session: Session,
    raw_session: Session,
    dataset_version_id: str | None = None,
) -> dict:
    bronze_articles = bronze_session.scalars(select(BronzeArticle)).all()

    by_doi: dict[str, list[BronzeArticle]] = defaultdict(list)
    rejected_rows: list[RejectedArticle] = []
    now = dt.datetime.now(dt.UTC)
    for article in bronze_articles:
        if not article.doi:
            rejected_rows.append(
                RejectedArticle(
                    dataset_version_id=dataset_version_id,
                    bronze_id=article.id,
                    source=article.source,
                    source_id=article.source_id,
                    title=article.title,
                    publication_category=article.publication_category,
                    publication_category_basis=article.publication_category_basis,
                    reason="no_doi",
                    rejected_at=now,
                )
            )
            continue
        by_doi[article.doi].append(article)

    # Clear and rebuild -- silver is fully derived from bronze each run.
    silver_session.query(SilverArticle).delete()
    silver_session.query(RejectedArticle).delete()

    silver_rows: list[SilverArticle] = []
    for doi, group in by_doi.items():
        merged = _merge_group(doi, group)
        merged["is_non_article"] = _is_non_article(
            merged.get("record_type"), merged.get("abstract")
        )
        row = SilverArticle(dataset_version_id=dataset_version_id, **merged)
        silver_session.add(row)
        silver_rows.append(row)
    silver_session.add_all(rejected_rows)
    silver_session.flush()  # assign ids for PDF linking

    pdf_files = raw_session.scalars(select(PdfFile)).all()
    _link_pdfs(silver_session, silver_rows, pdf_files)

    silver_session.flush()

    categories: dict[str, int] = defaultdict(int)
    for row in silver_rows:
        categories[str(row.publication_category)] += 1
    return {
        "written": len(silver_rows),
        "skipped_no_doi": len(rejected_rows),
        "rejected_persisted": len(rejected_rows),
        "non_articles": sum(1 for r in silver_rows if r.is_non_article),
        "has_pdf": sum(1 for r in silver_rows if r.has_pdf),
        "categories": dict(categories),
    }
