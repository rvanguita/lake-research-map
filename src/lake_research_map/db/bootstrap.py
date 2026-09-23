"""Create the four medallion databases (`raw`/`bronze`/`silver`/`gold`) on
the MySQL server if they don't exist yet, then create every layer's tables.

Note: `raw`/`bronze`/`silver` are typically already present on the server
(shared with unrelated tables from other projects) -- `CREATE DATABASE IF
NOT EXISTS` is a no-op for them. `gold` is created fresh.

Usage:
    uv run python -m lake_research_map.db.bootstrap
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from lake_research_map.config import LAYERS, get_settings
from lake_research_map.db import bronze_models, gold_models, raw_models, silver_models
from lake_research_map.db.engines import get_engine

_MODELS = {
    "raw": raw_models,
    "bronze": bronze_models,
    "silver": silver_models,
    "gold": gold_models,
}


def create_databases() -> None:
    settings = get_settings()
    server_engine = create_engine(settings.server_url(), future=True)
    with server_engine.connect() as conn:
        for layer in LAYERS:
            db_name = settings.database_name(layer)
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
        conn.commit()
    server_engine.dispose()


# Additive migrations for `lit_articles`, applied where the column is missing.
# `create_all` only ever creates absent *tables*, so a column added to a model
# after a table already exists needs this. Every entry must be nullable and
# purely additive -- this runs unattended on every pipeline start, so it must
# never be able to drop or rewrite existing data.
#
# Keyed by table: `lit_articles` was the only one that ever grew a column until
# the semantic stage started scoring against a second anchor, and a migration
# map that can only reach one table silently isn't a migration map.
_ADDITIVE_COLUMNS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "lit_articles": {
        "reference_count": ("INT NULL", ("bronze", "silver", "gold")),
        "sources": ("JSON NULL", ("gold",)),
        # IEEE-only enrichment, see transform/bronze_articles.py.
        "countries": ("JSON NULL", ("bronze", "silver", "gold")),
        "online_date": ("DATE NULL", ("bronze", "silver", "gold")),
        "document_type": ("VARCHAR(128) NULL", ("bronze", "silver", "gold")),
        "license": ("VARCHAR(64) NULL", ("bronze", "silver", "gold")),
        # ROADMAP #4: flag non-article records (book front matter, etc.).
        "is_non_article": ("BOOLEAN NOT NULL DEFAULT FALSE", ("silver", "gold")),
        "dataset_version_id": ("VARCHAR(64) NULL", ("bronze", "silver")),
        "publication_category": ("VARCHAR(32) NULL", ("bronze", "silver", "gold")),
        "publication_category_basis": ("VARCHAR(64) NULL", ("bronze", "silver", "gold")),
    },
    # Cosine to the logistics anchor, see transform/semantics.py.
    "lit_semantics": {"offtopic_score": ("FLOAT NULL", ("gold",))},
    # WP-17: cluster/projection stability recorded with the run that produced it.
    "lit_semantic_runs": {"stability": ("JSON NULL", ("gold",))},
    # WP-24: how a citation edge was observed (complete reference list vs. a
    # paginated cites: crawl), which the coverage audit needs.
    "lit_citation_edges": {"discovered_via": ("VARCHAR(32) NULL", ("bronze",))},
    # WP-24: whether a work's forward crawl ran and whether it was truncated.
    # Edge presence cannot say: an uncited work and an uncrawled one look alike.
    "lit_external_works": {
        "citing_crawled_at": ("DATETIME NULL", ("bronze",)),
        "citing_truncated": ("BOOLEAN NULL", ("bronze",)),
    },
    # ROADMAP #1: binary embedding storage (migration from JSON to BLOB).
    "lit_chunks": {
        "embedding_bin": ("LONGBLOB NULL", ("gold",)),
        "text_sha256": ("VARCHAR(64) NULL", ("gold",)),
        "embed_revision": ("VARCHAR(128) NULL", ("gold",)),
        "embedding_dim": ("INT NULL", ("gold",)),
        "embedding_dtype": ("VARCHAR(32) NULL", ("gold",)),
        "embedding_normalized": ("BOOLEAN NULL", ("gold",)),
        "embedded_at": ("DATETIME NULL", ("gold",)),
    },
    "lit_rejected": {
        "dataset_version_id": ("VARCHAR(64) NULL", ("silver",)),
        "publication_category": ("VARCHAR(32) NULL", ("silver",)),
        "publication_category_basis": ("VARCHAR(64) NULL", ("silver",)),
    },
    "lit_dataset_articles": {
        "publication_category": ("VARCHAR(32) NULL", ("gold",)),
        "publication_category_basis": ("VARCHAR(64) NULL", ("gold",)),
    },
    "lit_source_files": {
        "dataset_version_id": ("VARCHAR(64) NULL", ("raw",)),
        "source_revision_id": ("VARCHAR(64) NULL", ("raw",)),
    },
    "lit_config": {
        "dataset_version_id": ("VARCHAR(64) NULL", ("raw",)),
        "source_revision_id": ("VARCHAR(64) NULL", ("raw",)),
    },
    "lit_ieee_csv_rows": {
        "dataset_version_id": ("VARCHAR(64) NULL", ("raw",)),
        "source_revision_id": ("VARCHAR(64) NULL", ("raw",)),
    },
    "lit_bib_entries": {
        "dataset_version_id": ("VARCHAR(64) NULL", ("raw",)),
        "source_revision_id": ("VARCHAR(64) NULL", ("raw",)),
    },
    "lit_pdf_files": {
        "archive_path": ("VARCHAR(512) NULL", ("raw",)),
        "dataset_version_id": ("VARCHAR(64) NULL", ("raw",)),
        "source_revision_id": ("VARCHAR(64) NULL", ("raw",)),
    },
    "lit_pipeline_runs": {
        "execution_id": ("VARCHAR(255) NULL", ("gold",)),
        "dataset_version_id": ("VARCHAR(64) NULL", ("gold",)),
        "input_version_id": ("VARCHAR(64) NULL", ("gold",)),
        "output_version_id": ("VARCHAR(64) NULL", ("gold",)),
        "sequence": ("INT NULL", ("gold",)),
        "attempt": ("INT NULL", ("gold",)),
    },
    "lit_pipeline_executions": {
        "heartbeat_at": ("DATETIME NULL", ("gold",)),
    },
    "lit_dataset_chunks": {
        "text_sha256": ("VARCHAR(64) NULL", ("gold",)),
        "embed_revision": ("VARCHAR(128) NULL", ("gold",)),
        "embedding_dim": ("INT NULL", ("gold",)),
        "embedding_dtype": ("VARCHAR(32) NULL", ("gold",)),
        "embedding_normalized": ("BOOLEAN NULL", ("gold",)),
        "embedded_at": ("DATETIME NULL", ("gold",)),
    },
}


def _migrate_config_uniqueness(engine) -> None:
    """Key search reports by file instead of collapsing them by publisher.

    Early versions allowed one ``lit_config`` row per source. That loses the
    provenance of later searches from the same publisher. This constraint-only
    migration does not rewrite rows: the former source uniqueness guarantees
    that existing ``source_file`` values are already distinct.
    """
    if engine.dialect.name != "mysql":
        return
    inspector = inspect(engine)
    if not inspector.has_table("lit_config"):
        return
    unique_constraints = inspector.get_unique_constraints("lit_config")
    indexes = inspector.get_indexes("lit_config")
    source_unique = next(
        (
            constraint
            for constraint in unique_constraints
            if constraint.get("column_names") == ["source"]
        ),
        None,
    )
    file_unique = next(
        (
            constraint
            for constraint in unique_constraints
            if constraint.get("column_names") == ["source_file"]
        ),
        None,
    )
    source_index = next(
        (
            index
            for index in indexes
            if index.get("column_names") == ["source"] and not index.get("unique")
        ),
        None,
    )
    if source_unique is None and file_unique is not None and source_index is not None:
        return

    clauses = []
    if source_unique is not None:
        clauses.append(f"DROP INDEX `{source_unique['name']}`")
    if file_unique is None:
        clauses.append("ADD UNIQUE INDEX `uq_lit_config_source_file` (`source_file`)")
    if source_index is None:
        clauses.append("ADD INDEX `ix_lit_config_source` (`source`)")
    if clauses:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE `lit_config` " + ", ".join(clauses)))


def create_tables() -> None:
    for layer, module in _MODELS.items():
        engine = get_engine(layer)
        module.Base.metadata.create_all(engine)

        inspector = inspect(engine)
        for table, columns in _ADDITIVE_COLUMNS.items():
            if not inspector.has_table(table):
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            missing = [
                (name, ddl)
                for name, (ddl, layers) in columns.items()
                if layer in layers and name not in existing
            ]
            if not missing:
                continue
            with engine.connect() as conn:
                for name, ddl in missing:
                    conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{name}` {ddl}"))
                conn.commit()
        if layer == "raw":
            _migrate_config_uniqueness(engine)


def bootstrap() -> None:
    create_databases()
    create_tables()


def main() -> None:
    bootstrap()
    settings = get_settings()
    print("Bootstrapped databases:")
    for layer in LAYERS:
        print(f"  {settings.database_name(layer)}")


if __name__ == "__main__":
    main()
