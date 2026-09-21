"""SILVER layer models (lit_silver database) -- cleaned, conformed and
deduplicated. One row per normalized DOI (the reliable cross-source join
key -- see CLAUDE.md), with quality flags and a fuzzy-matched link to a PDF
in data/articles/ where one exists.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lake_research_map.db.time import naive_utc_now


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "lit_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    doi: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)  # ['ieee'] | ['elsevier'] | both
    record_type: Mapped[str] = mapped_column(String(32))

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    volume: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issue: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pages: Mapped[str | None] = mapped_column(String(64), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # IEEE-only (see bronze_models) -- carried through so the dashboard can
    # read them from the layer it actually renders.
    countries: Mapped[list] = mapped_column(JSON, default=list)
    online_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # quality flags
    has_abstract: Mapped[bool] = mapped_column(Boolean, default=False)
    has_doi: Mapped[bool] = mapped_column(Boolean, default=True)
    is_duplicate_merge: Mapped[bool] = mapped_column(Boolean, default=False)

    # PDF link (fuzzy title match against data/articles/*.pdf)
    has_pdf: Mapped[bool] = mapped_column(Boolean, default=False)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_non_article: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0"
    )

    bronze_ids: Mapped[list] = mapped_column(
        JSON, default=list
    )  # source bronze.articles ids merged

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class RejectedArticle(Base):
    """Bronze records excluded at the silver stage, persisted for SLR audit.

    An SLR has to document what it threw away and why -- previously the count
    was returned in the stats dict and then lost.
    """

    __tablename__ = "lit_rejected"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    bronze_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. 'no_doi'
    rejected_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
