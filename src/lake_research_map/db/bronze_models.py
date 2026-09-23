"""BRONZE layer models (lit_bronze database) -- this is where cross-source
consolidation begins: IEEE (csv+bib) and Elsevier (bib) are unioned into one
common, typed schema. Pure pagination duplicates within a source are
collapsed, but there is no cross-source dedup or quality filtering yet --
that happens in silver.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Date, DateTime, Integer, String, Text, UniqueConstraint
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


class EnrichmentObservation(Base):
    """Append-only external metadata response with reproducible time semantics."""

    __tablename__ = "lit_enrichment_observations"
    __table_args__ = (
        UniqueConstraint("provider", "doi", "observed_at", name="uq_enrichment_provider_doi_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    provider_work_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ExternalWork(Base):
    __tablename__ = "lit_external_works"
    __table_args__ = (UniqueConstraint("provider", "provider_work_id", name="uq_external_work"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    last_observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    # WP-24: when the forward (`cites:`) crawl last ran for this work, and
    # whether it hit its page budget. Edge presence cannot answer either: a
    # work with genuinely no citing papers produces no edge and is
    # indistinguishable from one never crawled, which makes forward coverage
    # unmeasurable and lets a resumable crawl restart from the beginning.
    citing_crawled_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    citing_truncated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class ReferenceWork(Base):
    """Publication year of a work the corpus *cites* but does not contain.

    Price's index and reference-age analyses need the year of every cited
    reference, and 56,330 of the 57,454 distinct works the corpus cites are
    outside it, so `lit_external_works` -- which holds corpus works only --
    cannot answer. Kept in its own table because putting cited works into
    `lit_external_works` would silently change the population every coverage
    audit divides by.
    """

    __tablename__ = "lit_reference_works"
    __table_args__ = (UniqueConstraint("provider", "provider_work_id", name="uq_reference_work"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # 'success' or 'not_found'. Throttled attempts are not stored: they say
    # nothing about the work and must be retried.
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class CrossrefReferenceList(Base):
    """Whether Crossref holds a deposited reference list for a corpus DOI.

    A second source for WP-23's cited-reference years, chosen because Crossref
    has no per-window request quota, and because a publisher's own deposit
    carries the year of most references inline. Recorded per observation so a
    work with no deposit is a stated fact rather than a missing row.
    """

    __tablename__ = "lit_crossref_reference_lists"
    __table_args__ = (UniqueConstraint("doi", "observed_at", name="uq_crossref_list_doi_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # success | not_found
    deposited: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CrossrefReference(Base):
    """One deposited reference, as the citing work's publisher wrote it."""

    __tablename__ = "lit_crossref_references"
    __table_args__ = (
        UniqueConstraint("citing_doi", "position", "observed_at", name="uq_crossref_reference"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    citing_doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    reference_doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    # The year as deposited inline, parsed; None when the deposit carries none.
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class DoiYear(Base):
    """Publication year looked up for a DOI that a reference named without a year.

    Kept apart from `lit_crossref_references` so a lookup never rewrites an
    observation: the deposit said "no year", and that remains true.
    """

    __tablename__ = "lit_doi_years"
    __table_args__ = (UniqueConstraint("provider", "doi", name="uq_doi_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # success | not_found
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class CitationYearCount(Base):
    __tablename__ = "lit_citation_year_counts"
    __table_args__ = (
        UniqueConstraint(
            "provider", "provider_work_id", "year", "observed_at", name="uq_citation_year_count"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)


class CitationEdge(Base):
    __tablename__ = "lit_citation_edges"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "citing_work_id",
            "cited_work_id",
            "observed_at",
            name="uq_citation_edge_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    citing_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    cited_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    # How the edge was seen, which is not the same question as its direction.
    # `referenced_works` is the citing work's own complete reference list, so a
    # missing edge means the reference is absent. A `cites:` query is paginated
    # and can be cut short, so a missing edge there may only mean the crawl
    # stopped. Coverage cannot be computed without telling the two apart.
    discovered_via: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


class AccessObservation(Base):
    __tablename__ = "lit_access_observations"
    __table_args__ = (
        UniqueConstraint(
            "provider", "provider_work_id", "observed_at", name="uq_access_observation"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider_work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    is_oa: Mapped[bool | None] = mapped_column(nullable=True)
    oa_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    license: Mapped[str | None] = mapped_column(String(128), nullable=True)
    landing_page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
