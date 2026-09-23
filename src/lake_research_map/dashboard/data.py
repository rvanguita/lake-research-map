"""Data-loading helpers for the Streamlit dashboard.

Reads directly from the medallion MySQL databases (via the same per-layer
engines the pipeline uses) and returns plain pandas DataFrames. Every
function tolerates a layer/table that doesn't exist yet -- the dashboard is
meant to be usable even before the pipeline has been run end to end.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import MetaData, Table, and_, exists, inspect, or_, select, text
from sqlalchemy.exc import SQLAlchemyError

from lake_research_map.db.engines import get_engine

logger = logging.getLogger(__name__)

LAYER_TABLES = {
    "raw": [
        "lit_source_files",
        "lit_config",
        "lit_ieee_csv_rows",
        "lit_bib_entries",
        "lit_pdf_files",
    ],
    "bronze": ["lit_articles", "lit_enrichment_observations"],
    "silver": ["lit_articles"],
    "gold": [
        "lit_articles",
        "lit_chunks",
        "lit_semantics",
        "lit_duplicate_pairs",
        "lit_duplicate_overrides",
    ],
}

GOLD_ANALYTICAL_COLUMNS = frozenset(
    {
        "doi",
        "sources",
        "title",
        "authors",
        "year",
        "venue",
        "keywords",
        "abstract",
        "citation_count",
        "reference_count",
        "has_pdf",
        "is_non_article",
        "publication_category",
    }
)


def table_exists(layer: str, table: str) -> bool:
    try:
        engine = get_engine(layer)
        return inspect(engine).has_table(table)
    except SQLAlchemyError as exc:
        logger.warning("table_exists(%r, %r): database unreachable (%s)", layer, table, exc)
        return False


def layer_row_counts() -> pd.DataFrame:
    """One row per (layer, table) with its row count, or an error note."""
    rows = []
    for layer, tables in LAYER_TABLES.items():
        for table in tables:
            count = None
            status = "ok"
            try:
                engine = get_engine(layer)
                if inspect(engine).has_table(table):
                    with engine.connect() as conn:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
                else:
                    count = 0
                    status = "no table yet"
            except SQLAlchemyError as exc:
                logger.warning(
                    "layer_row_counts(%r, %r): database unreachable (%s)", layer, table, exc
                )
                status = f"unreachable ({type(exc).__name__})"
            rows.append({"layer": layer, "table": table, "rows": count, "status": status})
    return pd.DataFrame(rows)


def load_articles(layer: str) -> pd.DataFrame:
    if layer == "gold":
        return load_active_dataset_table("lit_dataset_articles")
    if not table_exists(layer, "lit_articles"):
        return pd.DataFrame()
    engine = get_engine(layer)
    return pd.read_sql_table("lit_articles", engine)


def active_dataset_version() -> str | None:
    """Return the immutable version currently published to dashboard readers."""
    if not table_exists("gold", "lit_publication_state"):
        return None
    engine = get_engine("gold")
    state = Table("lit_publication_state", MetaData(), autoload_with=engine)
    if "active_version_id" not in state.c:
        return None
    value = pd.read_sql_query(
        select(state.c.active_version_id).where(state.c.id == 1).limit(1), engine
    )
    if value.empty or pd.isna(value.iloc[0, 0]):
        return None
    return str(value.iloc[0, 0])


def load_active_dataset_table(table_name: str) -> pd.DataFrame:
    """Load one immutable Gold snapshot table bound to the publication pointer."""
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", table_name):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table(table_name, MetaData(), autoload_with=engine)
    if "dataset_version_id" not in table.c:
        return pd.DataFrame()
    return pd.read_sql_query(select(table).where(table.c.dataset_version_id == version_id), engine)


def load_search_configs() -> pd.DataFrame:
    """Per-source search provenance from `raw.lit_config` (query, filters, year range, URL)."""
    if not table_exists("raw", "lit_config"):
        return pd.DataFrame()
    engine = get_engine("raw")
    return pd.read_sql_table("lit_config", engine)


def load_enrichment_observation_times() -> pd.DataFrame:
    """Observation timestamps for time-varying bibliometric metadata."""
    if not table_exists("bronze", "lit_enrichment_observations"):
        return pd.DataFrame(columns=["observed_at"])
    engine = get_engine("bronze")
    table = Table("lit_enrichment_observations", MetaData(), autoload_with=engine)
    if "observed_at" not in table.c:
        return pd.DataFrame(columns=["observed_at"])
    return pd.read_sql_query(select(table.c.observed_at), engine)


def assess_gold_articles(df: pd.DataFrame) -> tuple[str, ...]:
    """Return reasons why a Gold frame is not safe as the analytical population."""
    if df.empty:
        return ("The Gold layer is empty or unavailable.",)

    reasons: list[str] = []
    missing_columns = sorted(GOLD_ANALYTICAL_COLUMNS.difference(df.columns))
    if missing_columns:
        reasons.append("Gold is missing required fields: " + ", ".join(missing_columns))
    if "doi" in df.columns:
        dois = df["doi"].fillna("").astype(str).str.strip().str.lower()
        if dois.eq("").any():
            reasons.append("Gold contains blank DOI values.")
        if dois[dois.ne("")].duplicated().any():
            reasons.append("Gold contains duplicate DOI values.")
    return tuple(reasons)


def active_version_blocking_failures(version_id: str | None) -> tuple[str, ...]:
    """Blocking quality checks whose latest recorded outcome for this version failed.

    `assess_gold_articles` only checks the frame's shape. The pipeline has
    already evaluated a full contract for the version and persisted every
    outcome in `lit_quality_results`; a version activated by hand, or one
    re-checked after publication, can carry a recorded `error` failure that the
    shape check would never see. Only the latest run of each check counts: a
    failure that a later stage run fixed must not keep the version degraded.
    """
    if not version_id or not table_exists("gold", "lit_quality_results"):
        return ()
    engine = get_engine("gold")
    try:
        table = Table("lit_quality_results", MetaData(), autoload_with=engine)
        results = pd.read_sql_query(
            select(table.c.check_id, table.c.stage_run_id, table.c.passed).where(
                table.c.dataset_version_id == version_id, table.c.severity == "error"
            ),
            engine,
        )
    except SQLAlchemyError as exc:
        logger.warning("active_version_blocking_failures: %s", exc)
        return ()
    return blocking_failures(results)


def blocking_failures(results: pd.DataFrame) -> tuple[str, ...]:
    """Reasons for the checks whose most recent run did not pass."""
    if results.empty:
        return ()
    latest = results.sort_values("stage_run_id").drop_duplicates("check_id", keep="last")
    failed = latest.loc[~latest["passed"].astype(bool), "check_id"].astype(str)
    return tuple(
        f"Recorded blocking check failed for this version: {check}." for check in sorted(failed)
    )


def select_articles_layer() -> tuple[str, pd.DataFrame, dict[str, object]]:
    """Select Gold when its contract passes, otherwise degrade explicitly."""
    version_id = active_dataset_version()
    gold_df = load_articles("gold")
    gold_issues = assess_gold_articles(gold_df)
    if not gold_df.empty:
        gold_issues = gold_issues + active_version_blocking_failures(version_id)
    if not gold_issues:
        return (
            "gold",
            gold_df,
            {
                "layer": "gold",
                "is_canonical": True,
                "dataset_version_id": version_id,
                "fallback_reasons": (),
            },
        )

    for layer in ("silver", "bronze"):
        df = load_articles(layer)
        if not df.empty:
            return (
                layer,
                df,
                {
                    "layer": layer,
                    "is_canonical": False,
                    "dataset_version_id": None,
                    "fallback_reasons": gold_issues,
                },
            )
    return (
        "none",
        pd.DataFrame(),
        {
            "layer": "none",
            "is_canonical": False,
            "dataset_version_id": None,
            "fallback_reasons": gold_issues,
        },
    )


def load_articles_all_layers() -> dict[str, pd.DataFrame]:
    """Load `articles` from bronze, silver, and gold in one call.

    Used by the pipeline/layers page to compare the medallion stages
    directly instead of only seeing the single "best" layer the rest of the
    dashboard uses.
    """
    return {layer: load_articles(layer) for layer in ("bronze", "silver", "gold")}


_CHUNK_LIGHT_COLUMNS = ("id", "doi", "seq", "chunk_type", "char_len", "embed_model", "created_at")


def load_chunks() -> pd.DataFrame:
    """Lightweight chunk metadata from `gold.lit_chunks`: everything the
    dashboard's aggregate stats/charts need, without the `text` and
    `embedding` columns. Both are large per row -- `embedding` is a ~768-float
    JSON vector -- and unused outside the on-demand search box in
    `pages/quality.py`, which pulls them separately via
    `load_chunk_search_data` only once a query is actually submitted.
    Deserializing them here for every row dominated render time on every page
    that touches chunk counts (~7s for ~6k rows just for this one query).
    """
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", "lit_dataset_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_dataset_chunks", MetaData(), autoload_with=engine)
    columns = [table.c[name] for name in _CHUNK_LIGHT_COLUMNS if name in table.c]
    columns.append(table.c.embedding_bin.is_not(None).label("has_embedding"))
    return pd.read_sql_query(
        select(*columns).where(table.c.dataset_version_id == version_id), engine
    )


def load_semantics() -> pd.DataFrame:
    """Per-article semantic signals from `gold.lit_semantics` (`--stage semantic`)."""
    return load_active_dataset_table("lit_dataset_semantics")


def load_active_semantic_run() -> pd.DataFrame:
    """Latest semantic-run manifest attached to the active Gold version."""
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", "lit_semantic_runs"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_semantic_runs", MetaData(), autoload_with=engine)
    if "dataset_version_id" not in table.c:
        return pd.DataFrame()
    statement = (
        select(table)
        .where(table.c.dataset_version_id == version_id)
        .order_by(table.c.created_at.desc())
        .limit(1)
    )
    return pd.read_sql_query(statement, engine)


def load_duplicate_pairs() -> pd.DataFrame:
    """Unresolved near-duplicate pairs, excluding persistent review decisions."""
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", "lit_dataset_duplicate_pairs"):
        return pd.DataFrame()
    engine = get_engine("gold")
    if not inspect(engine).has_table("lit_duplicate_overrides"):
        return load_active_dataset_table("lit_dataset_duplicate_pairs")

    metadata = MetaData()
    pairs = Table("lit_dataset_duplicate_pairs", metadata, autoload_with=engine)
    overrides = Table("lit_duplicate_overrides", metadata, autoload_with=engine)
    reviewed = exists(
        select(1)
        .select_from(overrides)
        .where(
            or_(
                and_(
                    overrides.c.doi_a == pairs.c.doi_a,
                    overrides.c.doi_b == pairs.c.doi_b,
                ),
                and_(
                    overrides.c.doi_a == pairs.c.doi_b,
                    overrides.c.doi_b == pairs.c.doi_a,
                ),
            )
        )
    )
    return pd.read_sql_query(
        select(pairs).where(
            pairs.c.dataset_version_id == version_id,
            ~reviewed,
        ),
        engine,
    )


def load_duplicate_overrides() -> pd.DataFrame:
    """Persistent human decisions for cross-DOI near-duplicate pairs."""
    if not table_exists("gold", "lit_duplicate_overrides"):
        return pd.DataFrame()
    return pd.read_sql_table("lit_duplicate_overrides", get_engine("gold"))


def load_pipeline_runs(limit: int = 50) -> pd.DataFrame:
    """Return recent pipeline audit rows, or an empty frame before bootstrap."""
    if not table_exists("gold", "lit_pipeline_runs"):
        return pd.DataFrame()
    table = Table("lit_pipeline_runs", MetaData(), autoload_with=get_engine("gold"))
    query = select(table).order_by(table.c.finished_at.desc()).limit(limit)
    return pd.read_sql_query(query, get_engine("gold"))


def load_pipeline_executions(limit: int = 25) -> pd.DataFrame:
    """Return parent executions that correlate individual stage attempts."""
    if not table_exists("gold", "lit_pipeline_executions"):
        return pd.DataFrame()
    table = Table("lit_pipeline_executions", MetaData(), autoload_with=get_engine("gold"))
    return pd.read_sql_query(
        select(table).order_by(table.c.started_at.desc()).limit(limit), get_engine("gold")
    )


def load_dataset_versions(limit: int = 25) -> pd.DataFrame:
    if not table_exists("gold", "lit_dataset_versions"):
        return pd.DataFrame()
    table = Table("lit_dataset_versions", MetaData(), autoload_with=get_engine("gold"))
    return pd.read_sql_query(
        select(table).order_by(table.c.created_at.desc()).limit(limit), get_engine("gold")
    )


def load_publication_state() -> pd.DataFrame:
    if not table_exists("gold", "lit_publication_state"):
        return pd.DataFrame()
    return pd.read_sql_table("lit_publication_state", get_engine("gold"))


def load_quality_results(limit: int = 500) -> pd.DataFrame:
    if not table_exists("gold", "lit_quality_results"):
        return pd.DataFrame()
    table = Table("lit_quality_results", MetaData(), autoload_with=get_engine("gold"))
    return pd.read_sql_query(
        select(table).order_by(table.c.checked_at.desc()).limit(limit), get_engine("gold")
    )


def load_source_changes(limit: int = 500) -> pd.DataFrame:
    if not table_exists("raw", "lit_source_changes"):
        return pd.DataFrame()
    table = Table("lit_source_changes", MetaData(), autoload_with=get_engine("raw"))
    return pd.read_sql_query(
        select(table).order_by(table.c.recorded_at.desc()).limit(limit), get_engine("raw")
    )


def load_rejected_records() -> pd.DataFrame:
    """Return the silver rejection audit without assuming the table exists."""
    if not table_exists("silver", "lit_rejected"):
        return pd.DataFrame()
    table = Table("lit_rejected", MetaData(), autoload_with=get_engine("silver"))
    query = select(table).order_by(table.c.rejected_at.desc())
    return pd.read_sql_query(query, get_engine("silver"))


def load_chunk_search_data() -> pd.DataFrame:
    """Binary-first chunk rows for the on-demand search box in
    `pages/quality.py` -- loaded lazily, only once a query is actually
    submitted, never on a plain page render.
    """
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", "lit_dataset_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_dataset_chunks", MetaData(), autoload_with=engine)
    names = ("id", "doi", "seq", "chunk_type", "text", "embedding_bin", "embed_model")
    columns = [table.c[name] for name in names if name in table.c]
    return pd.read_sql_query(
        select(*columns).where(table.c.dataset_version_id == version_id), engine
    )


def raw_funnel_counts() -> dict[str, dict[str, int]]:
    """Per-source row counts for the raw-layer tables that feed bronze.

    Used by the pipeline/layers page's funnel chart -- `ieee_csv_rows` has no
    `source` column of its own (the CSV is IEEE-only), so it's reported as a
    single IEEE bucket alongside the two `bib_entries` sources.
    """
    counts = {
        "ieee": {"csv_rows": 0, "bib_entries": 0},
        "elsevier": {"csv_rows": 0, "bib_entries": 0},
    }
    try:
        engine = get_engine("raw")
        with engine.connect() as conn:
            if inspect(engine).has_table("lit_ieee_csv_rows"):
                counts["ieee"]["csv_rows"] = (
                    conn.execute(text("SELECT COUNT(*) FROM `lit_ieee_csv_rows`")).scalar() or 0
                )
            if inspect(engine).has_table("lit_bib_entries"):
                for source, n in conn.execute(
                    text("SELECT source, COUNT(*) FROM `lit_bib_entries` GROUP BY source")
                ):
                    counts.setdefault(source, {"csv_rows": 0, "bib_entries": 0})
                    counts[source]["bib_entries"] = n
    except SQLAlchemyError as exc:
        logger.warning("raw_funnel_counts: database unreachable (%s)", exc)
    return counts


def bronze_doi_dropped_counts() -> dict[str, int]:
    """Bronze rows with no DOI, per source -- these never make it into silver.

    `silver_articles.py` counts this drop as `skipped_no_doi` but only prints
    it; this reconstructs the per-source breakdown live from bronze, which is
    never truncated.
    """
    result = {"ieee": 0, "elsevier": 0}
    try:
        engine = get_engine("bronze")
        if not inspect(engine).has_table("lit_articles"):
            return result
        with engine.connect() as conn:
            for source, n in conn.execute(
                text(
                    "SELECT source, COUNT(*) FROM `lit_articles` WHERE doi IS NULL GROUP BY source"
                )
            ):
                result[source] = n
    except SQLAlchemyError as exc:
        logger.warning("bronze_doi_dropped_counts: database unreachable (%s)", exc)
    return result


def load_abstract_embeddings_data() -> pd.DataFrame:
    """Load abstract chunk vectors (doi, embedding_bin) for projections and novelty."""
    version_id = active_dataset_version()
    if version_id is None or not table_exists("gold", "lit_dataset_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_dataset_chunks", MetaData(), autoload_with=engine)
    if "embedding_bin" not in table.c:
        return pd.DataFrame()

    # Binary only. Selecting the JSON mirror alongside it was what made this
    # read cost 11.6s instead of 2.3s on the reference corpus, and the mirror
    # is no longer written for new versions.
    query = select(table.c.doi, table.c.embedding_bin).where(
        table.c.dataset_version_id == version_id,
        table.c.chunk_type == "abstract",
        table.c.embedding_bin.is_not(None),
    )
    return pd.read_sql_query(query, engine)


def load_review_labels(workflow: str) -> pd.DataFrame:
    """Reviewed labels for one human-evidence workflow, newest revision per pair.

    Labels are append-only: a reviewer changing their mind adds a row rather
    than overwriting one, so reading the raw table double-counts. The latest
    revision per (assignment, reviewer) is the decision that stands, and the
    superseded rows remain the audit trail.
    """
    if not table_exists("gold", "lit_review_labels") or not table_exists(
        "gold", "lit_review_assignments"
    ):
        return pd.DataFrame()
    engine = get_engine("gold")
    try:
        return pd.read_sql_query(
            text(
                # The reviewer lives on the assignment, not the label; a label
                # only knows which assignment it answers.
                "SELECT a.workflow, a.subject_id, a.reviewer_id, l.label, l.rationale, "
                "       l.label_revision, l.imported_at "
                "FROM lit_review_labels l "
                "JOIN lit_review_assignments a ON a.id = l.assignment_id "
                "WHERE a.workflow = :workflow "
                "  AND l.label_revision = ("
                "      SELECT MAX(l2.label_revision) FROM lit_review_labels l2 "
                "      WHERE l2.assignment_id = l.assignment_id"
                "  )"
            ),
            engine,
            params={"workflow": workflow},
        )
    except SQLAlchemyError as exc:
        logger.warning("load_review_labels(%r): %s", workflow, exc)
        return pd.DataFrame()
