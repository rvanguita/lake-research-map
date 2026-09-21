"""BRONZE layer models (lit_bronze database) -- this is where cross-source
consolidation begins: IEEE (csv+bib) and Elsevier (bib) are unioned into one
common, typed schema. Pure pagination duplicates within a source are
collapsed, but there is no cross-source dedup or quality filtering yet --
that happens in silver.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Date, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lake_research_map.db.time import naive_utc_now


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "lit_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    source: Mapped[str] = mapped_column(String(32), index=True)  # 'ieee' | 'elsevier'
    source_id: Mapped[str] = mapped_column(String(255))  # bib key or csv row ref
    record_type: Mapped[str] = mapped_column(String(32))  # article, incollection, book...
    publication_category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    publication_category_basis: Mapped[str | None] = mapped_column(String(64), nullable=True)

    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list] = mapped_column(JSON, default=list)  # list[str]
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)  # journal/booktitle
    volume: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issue: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pages: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issn: Mapped[str | None] = mapped_column(String(64), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)  # list[str]
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # IEEE-only: the Elsevier .bib carries none of these. Always report them
    # against the IEEE subset (~17% of the corpus), never the whole thing.
    countries: Mapped[list] = mapped_column(JSON, default=list)  # from Author Affiliations
    online_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)  # monthly resolution
    document_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license: Mapped[str | None] = mapped_column(String(64), nullable=True)

    raw_bib_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_csv_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)

    __table_args__ = (UniqueConstraint("source", "source_id", name="uq_source_record"),)
