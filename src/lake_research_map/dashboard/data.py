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
    "bronze": ["lit_articles"],
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
    if not table_exists(layer, "lit_articles"):
        return pd.DataFrame()
    engine = get_engine(layer)
    return pd.read_sql_table("lit_articles", engine)


def load_search_configs() -> pd.DataFrame:
    """Per-source search provenance from `raw.lit_config` (query, filters, year range, URL)."""
    if not table_exists("raw", "lit_config"):
        return pd.DataFrame()
    engine = get_engine("raw")
    return pd.read_sql_table("lit_config", engine)


def assess_gold_articles(df: pd.DataFrame) -> tuple[str, ...]:
    """Return reasons why a Gold frame is not safe as the analytical population."""
    if df.empty:
        return ("A camada Gold está vazia ou indisponível.",)

    reasons: list[str] = []
    missing_columns = sorted(GOLD_ANALYTICAL_COLUMNS.difference(df.columns))
    if missing_columns:
        reasons.append("Gold não possui campos obrigatórios: " + ", ".join(missing_columns))
    if "doi" in df.columns:
        dois = df["doi"].fillna("").astype(str).str.strip().str.lower()
        if dois.eq("").any():
            reasons.append("Gold contém DOI vazio.")
        if dois[dois.ne("")].duplicated().any():
            reasons.append("Gold contém DOI duplicado.")
    return tuple(reasons)


def select_articles_layer() -> tuple[str, pd.DataFrame, dict[str, object]]:
    """Select Gold when its minimum contract passes, otherwise degrade explicitly."""
    gold_df = load_articles("gold")
    gold_issues = assess_gold_articles(gold_df)
    if not gold_issues:
        return (
            "gold",
            gold_df,
            {
                "layer": "gold",
                "is_canonical": True,
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
                    "fallback_reasons": gold_issues,
                },
            )
    return (
        "none",
        pd.DataFrame(),
        {
            "layer": "none",
            "is_canonical": False,
            "fallback_reasons": gold_issues,
        },
    )


def pick_best_articles_layer() -> tuple[str, pd.DataFrame]:
    """Backward-compatible wrapper around the canonical layer selector."""
    layer, df, _ = select_articles_layer()
    return layer, df


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
    if not table_exists("gold", "lit_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_chunks", MetaData(), autoload_with=engine)
    columns = [table.c[name] for name in _CHUNK_LIGHT_COLUMNS if name in table.c]
    columns.append(table.c.embedding.is_not(None).label("has_embedding"))
    return pd.read_sql_query(select(*columns), engine)


def load_semantics() -> pd.DataFrame:
    """Per-article semantic signals from `gold.lit_semantics` (`--stage semantic`)."""
    if not table_exists("gold", "lit_semantics"):
        return pd.DataFrame()
    return pd.read_sql_table("lit_semantics", get_engine("gold"))


def load_duplicate_pairs() -> pd.DataFrame:
    """Unresolved near-duplicate pairs, excluding persistent review decisions."""
    if not table_exists("gold", "lit_duplicate_pairs"):
        return pd.DataFrame()
    engine = get_engine("gold")
    if not inspect(engine).has_table("lit_duplicate_overrides"):
        return pd.read_sql_table("lit_duplicate_pairs", engine)

    metadata = MetaData()
    pairs = Table("lit_duplicate_pairs", metadata, autoload_with=engine)
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
    return pd.read_sql_query(select(pairs).where(~reviewed), engine)


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
    """Full chunk rows (`text` + `embedding`), for the search box in
    `pages/quality.py` -- loaded lazily, only once a query is actually
    submitted, never on a plain page render.
    """
    if not table_exists("gold", "lit_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    return pd.read_sql_table("lit_chunks", engine)


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
    """Load only abstract chunk embeddings (doi, embedding_bin, embedding) for vector projections and novelty."""
    if not table_exists("gold", "lit_chunks"):
        return pd.DataFrame()
    engine = get_engine("gold")
    table = Table("lit_chunks", MetaData(), autoload_with=engine)
    cols = [table.c.doi]
    if "embedding_bin" in table.c:
        cols.append(table.c.embedding_bin)
    if "embedding" in table.c:
        cols.append(table.c.embedding)

    query = select(*cols).where(table.c.chunk_type == "abstract")
    if "embedding_bin" in table.c and "embedding" in table.c:
        query = query.where((table.c.embedding_bin.is_not(None)) | (table.c.embedding.is_not(None)))
    elif "embedding_bin" in table.c:
        query = query.where(table.c.embedding_bin.is_not(None))
    elif "embedding" in table.c:
        query = query.where(table.c.embedding.is_not(None))

    return pd.read_sql_query(query, engine)
