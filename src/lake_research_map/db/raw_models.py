"""RAW layer models (lit_raw database) -- verbatim ingestion, one table per
source artifact type. Nothing here is normalized or deduplicated; it exists
so every downstream layer can be rebuilt without re-touching the filesystem.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lake_research_map.db.time import naive_utc_now


class Base(DeclarativeBase):
    pass


class SourceFile(Base):
    """Manifest of every file ingested, for idempotent re-runs."""

    __tablename__ = "lit_source_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 512 chars * 4 bytes (utf8mb4) = 2048 bytes, under MySQL's 3072-byte max key
    # length; real corpus paths top out at ~220 chars, so there's ample headroom.
    path: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(32))  # 'ieee' | 'elsevier' | 'articles'
    kind: Mapped[str] = mapped_column(String(32))  # 'csv' | 'bib' | 'pdf' | 'config'
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    mtime: Mapped[dt.datetime] = mapped_column(DateTime)
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class SourceBlob(Base):
    """One immutable, content-addressed input object."""

    __tablename__ = "lit_source_blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    archive_path: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class SourceRevision(Base):
    """Immutable identity of one logical path at one content revision."""

    __tablename__ = "lit_source_revisions"
    __table_args__ = (UniqueConstraint("path", "sha256", name="uq_source_revision_path_hash"),)

    revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(String(512), index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mtime: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DatasetSourceFile(Base):
    """One present source path in an immutable dataset-version manifest."""

    __tablename__ = "lit_dataset_source_files"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "path", name="uq_dataset_source_path"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    source_revision_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class SourceChange(Base):
    """Per-file reconciliation event retained for audit and provenance."""

    __tablename__ = "lit_source_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    change_type: Mapped[str] = mapped_column(String(16), nullable=False)
    previous_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class Config(Base):
    """Parsed provenance from data/{ieee,elsevier}/config.csv (free text)."""

    __tablename__ = "lit_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), unique=True)
    query_string: Mapped[str | None] = mapped_column(Text, nullable=True)
    filters: Mapped[str | None] = mapped_column(Text, nullable=True)
    year_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    search_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    source_file: Mapped[str] = mapped_column(String(512))
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class IeeeCsvRow(Base):
    """One row per line of data/ieee/export*.csv, columns kept as text."""

    __tablename__ = "lit_ieee_csv_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    row_index: Mapped[int] = mapped_column(Integer)
    # part of a 2-column unique key below -- 512*4 = 2048 bytes, under the 3072 cap
    source_file: Mapped[str] = mapped_column(String(512))
    fields: Mapped[dict] = mapped_column(JSON)  # {csv column name: value}
    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (UniqueConstraint("source_file", "row_index"),)


class BibEntry(Base):
    """One row per BibTeX entry, either source, fields kept as raw JSON."""

    __tablename__ = "lit_bib_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), index=True)  # 'ieee' | 'elsevier'
    bib_key: Mapped[str] = mapped_column(String(255), index=True)
    entry_type: Mapped[str] = mapped_column(String(32))  # article, incollection, book...
    # part of a 3-column unique key below -- (32+255+255)*4 = 2168 bytes, under 3072
    source_file: Mapped[str] = mapped_column(String(255))
    fields: Mapped[dict] = mapped_column(JSON)  # {bibtex field name: value}
    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (UniqueConstraint("source", "bib_key", "source_file"),)


class PdfFile(Base):
    """Inventory of data/articles/*.pdf."""

    __tablename__ = "lit_pdf_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True)  # 255*4=1020 bytes
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    archive_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
