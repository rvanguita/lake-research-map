"""GOLD layer models (lit_gold database) -- curated, RAG-ready.

`articles` is the table a human or agent scans to decide which paper to cite.
`chunks` is the RAG ingestion unit: one row per passage of text. `embedding`/
`embed_model` are filled in by the `embed` pipeline stage
(`transform/embeddings.py`, via `fastembed`) -- they stay NULL only until
that stage has been run at least once for a given chunk.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lake_research_map.db.time import naive_utc_now


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "lit_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    doi: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)  # ['ieee'] / ['elsevier'] / both
    publication_category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    publication_category_basis: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # IEEE-only (see bronze_models).
    countries: Mapped[list] = mapped_column(JSON, default=list)
    online_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license: Mapped[str | None] = mapped_column(String(64), nullable=True)

    has_pdf: Mapped[bool] = mapped_column(Boolean, default=False)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    silver_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_non_article: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0"
    )

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class Chunk(Base):
    __tablename__ = "lit_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    doi: Mapped[str] = mapped_column(String(255), index=True)  # value-FK to lit_articles.doi
    seq: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(32))  # 'abstract' | 'fulltext'
    text: Mapped[str] = mapped_column(Text)
    char_len: Mapped[int] = mapped_column(Integer)

    embedding: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # filled by --stage embed
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_bin: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    text_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embed_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_dtype: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_normalized: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    embedded_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class Semantics(Base):
    """Per-article signals derived from the abstract embedding, by `--stage semantic`.

    Kept in its own table rather than as columns on `Article` so the semantic
    stage can truncate and rebuild everything it owns without touching the
    curated article rows.
    """

    __tablename__ = "lit_semantics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    doi: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    # Cosine similarity to the topic anchor (see transform/semantics.py). Both
    # vectors are L2-normalized, so this is in [-1, 1] and in practice ~0.4-0.9.
    relevance_score: Mapped[float] = mapped_column(Float)

    # Cosine similarity to the *other* reading of "distribution system planning"
    # (logistics/supply chain). Nullable because rows written before the
    # contrastive anchor existed don't have it. The screening signal is the
    # margin `relevance_score - offtopic_score`, whose zero means "closer to
    # logistics than to the review's topic" -- a threshold a reviewer can
    # defend, unlike a percentile of a distribution that overlaps.
    offtopic_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    theme_id: Mapped[int] = mapped_column(Integer, index=True)
    theme_label: Mapped[str] = mapped_column(String(255))

    # 2D projection for the semantic map -- only meaningful relative to the
    # other rows of the same run, never as an absolute coordinate.
    map_x: Mapped[float] = mapped_column(Float)
    map_y: Mapped[float] = mapped_column(Float)

    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DuplicatePair(Base):
    """Near-identical abstracts that survived DOI deduplication as separate rows.

    DOI is this corpus's only reliable dedup key (see CLAUDE.md), so two
    printings of the same work under different DOIs stay separate. This table
    flags those for human review -- it never merges anything on its own.
    """

    __tablename__ = "lit_duplicate_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    doi_a: Mapped[str] = mapped_column(String(255), index=True)
    doi_b: Mapped[str] = mapped_column(String(255), index=True)
    similarity: Mapped[float] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DuplicateOverride(Base):
    """Persistent human decision for a detected cross-DOI duplicate pair.

    The pair columns are stored in lexical order so one unordered pair has one
    durable audit row. ``canonical_doi`` is populated only for ``merge``
    decisions; ``keep`` means the two publications were reviewed and must stay
    distinct.
    """

    __tablename__ = "lit_duplicate_overrides"
    __table_args__ = (
        UniqueConstraint("doi_a", "doi_b", name="uq_duplicate_override_pair"),
        CheckConstraint("doi_a < doi_b", name="ck_duplicate_override_order"),
        CheckConstraint("decision IN ('merge', 'keep')", name="ck_duplicate_override_decision"),
        CheckConstraint(
            "(decision = 'merge' AND canonical_doi IN (doi_a, doi_b)) OR "
            "(decision = 'keep' AND canonical_doi IS NULL)",
            name="ck_duplicate_override_canonical",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi_a: Mapped[str] = mapped_column(String(255), index=True)
    doi_b: Mapped[str] = mapped_column(String(255), index=True)
    decision: Mapped[str] = mapped_column(String(16))  # 'merge' | 'keep'
    canonical_doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=naive_utc_now, onupdate=naive_utc_now
    )


class DatasetVersion(Base):
    """A reproducible logical corpus version and its publication state."""

    __tablename__ = "lit_dataset_versions"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parent_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    code_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    curation_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    failed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class PipelineExecution(Base):
    """Parent execution shared by all stage attempts in one workflow."""

    __tablename__ = "lit_pipeline_executions"

    execution_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    requested_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class PublicationState(Base):
    """Singleton pointer separating the published Gold from working layers."""

    __tablename__ = "lit_publication_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    active_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    working_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    execution_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=naive_utc_now, onupdate=naive_utc_now
    )


class QualityResult(Base):
    """Persisted result of one executable data-contract check."""

    __tablename__ = "lit_quality_results"
    __table_args__ = (UniqueConstraint("stage_run_id", "check_id", name="uq_stage_quality_check"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    stage_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    check_id: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    observed: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    expected: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    checked_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DatasetArticle(Base):
    """Immutable Gold article snapshot keyed by dataset version and DOI."""

    __tablename__ = "lit_dataset_articles"
    __table_args__ = (UniqueConstraint("dataset_version_id", "doi", name="uq_dataset_article_doi"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    publication_category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    publication_category_basis: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    countries: Mapped[list] = mapped_column(JSON, default=list)
    online_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license: Mapped[str | None] = mapped_column(String(64), nullable=True)
    has_pdf: Mapped[bool] = mapped_column(Boolean, default=False)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    silver_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_non_article: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DatasetChunk(Base):
    """Versioned chunk and embedding snapshot."""

    __tablename__ = "lit_dataset_chunks"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "doi", "chunk_type", "seq", name="uq_dataset_chunk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_len: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_bin: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    text_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embed_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_dtype: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_normalized: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    embedded_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DatasetSemantics(Base):
    __tablename__ = "lit_dataset_semantics"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "doi", name="uq_dataset_semantics_doi"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    doi: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    offtopic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    theme_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    theme_label: Mapped[str] = mapped_column(String(255), nullable=False)
    map_x: Mapped[float] = mapped_column(Float, nullable=False)
    map_y: Mapped[float] = mapped_column(Float, nullable=False)
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class DatasetDuplicatePair(Base):
    __tablename__ = "lit_dataset_duplicate_pairs"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "doi_a", "doi_b", name="uq_dataset_duplicate_pair"),
        CheckConstraint("doi_a < doi_b", name="ck_dataset_duplicate_pair_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    doi_a: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    doi_b: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class SemanticRun(Base):
    """Immutable manifest for one semantic derivation of a dataset version."""

    __tablename__ = "lit_semantic_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embed_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embed_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Cluster/projection stability for this run. Kept on the run rather than
    # recomputed on the page so a past map's caveats stay attached to it: the
    # diagnostics describe the run that produced the layout, not whatever the
    # dashboard happens to be showing now.
    stability: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class ReviewProtocol(Base):
    """Versioned instructions shared by human-evidence workflows."""

    __tablename__ = "lit_review_protocols"
    __table_args__ = (
        UniqueConstraint("workflow", "protocol_version", name="uq_review_protocol_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False)
    instructions_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class ReviewAssignment(Base):
    """Deterministic reviewer assignment for an immutable dataset version."""

    __tablename__ = "lit_review_assignments"
    __table_args__ = (
        UniqueConstraint(
            "workflow",
            "dataset_version_id",
            "subject_id",
            "reviewer_id",
            name="uq_review_assignment",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    protocol_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stratum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class ReviewLabel(Base):
    """Append-only raw human label; adjudication never overwrites this row."""

    __tablename__ = "lit_review_labels"
    __table_args__ = (
        UniqueConstraint("assignment_id", "label_revision", name="uq_review_label_revision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assignment_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    label_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    label: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    imported_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class ReviewAdjudication(Base):
    __tablename__ = "lit_review_adjudications"
    __table_args__ = (
        UniqueConstraint("workflow", "dataset_version_id", "subject_id", name="uq_adjudication"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    final_label: Mapped[str] = mapped_column(String(128), nullable=False)
    adjudicator_id: Mapped[str] = mapped_column(String(128), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class ModelApproval(Base):
    """Durable approval gate for a measured threshold or retrieval default."""

    __tablename__ = "lit_model_approvals"
    __table_args__ = (
        UniqueConstraint(
            "workflow", "dataset_version_id", "model_sha256", name="uq_model_approval"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    label_set_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    approved_by: Mapped[str] = mapped_column(String(128), nullable=False)
    approved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=naive_utc_now)


class PipelineRun(Base):
    """One row per pipeline stage execution, written by `pipeline.py`.

    Turns the per-stage stats dict (which each `run_*` already returns) into a
    persistent record, so the "Camadas & Pipeline" page can answer "what changed
    between the last two runs" instead of only showing live row counts.
    """

    __tablename__ = "lit_pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    execution_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    input_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    output_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempt: Mapped[int | None] = mapped_column(Integer, nullable=True)

    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # 'success' | 'error'
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
