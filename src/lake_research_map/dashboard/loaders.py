"""Cached data access for the dashboard pages.

Every page calls these instead of touching `data.py` directly, so the MySQL
round-trips are shared across pages through Streamlit's global cache, and the
normalization each page depends on (list columns, scalar `source`) happens in
exactly one place.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import streamlit as st

from lake_research_map.dashboard import qualis
from lake_research_map.dashboard.data import (
    bronze_doi_dropped_counts,
    layer_row_counts,
    load_abstract_embeddings_data,
    load_articles_all_layers,
    load_chunk_search_data,
    load_chunks,
    load_dataset_versions,
    load_duplicate_overrides,
    load_duplicate_pairs,
    load_pipeline_executions,
    load_pipeline_runs,
    load_publication_state,
    load_quality_results,
    load_rejected_records,
    load_search_configs,
    load_semantics,
    load_source_changes,
    raw_funnel_counts,
    select_articles_layer,
)


@st.cache_data(ttl=60)
def pipeline_runs() -> pd.DataFrame:
    """Recent pipeline executions for the operational dashboard page."""
    return load_pipeline_runs()


@st.cache_data(ttl=30)
def pipeline_executions() -> pd.DataFrame:
    return load_pipeline_executions()


@st.cache_data(ttl=30)
def dataset_versions() -> pd.DataFrame:
    return load_dataset_versions()


@st.cache_data(ttl=10)
def publication_state() -> pd.DataFrame:
    return load_publication_state()


@st.cache_data(ttl=30)
def quality_results() -> pd.DataFrame:
    return load_quality_results()


@st.cache_data(ttl=30)
def source_changes() -> pd.DataFrame:
    return load_source_changes()


@st.cache_data(ttl=60)
def rejected_records() -> pd.DataFrame:
    """Silver rejection audit rows."""
    return load_rejected_records()


def _to_list(value):
    """Normalize a JSON/list-ish column value into a python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError, TypeError):
                continue
    return []


def filter_signature() -> tuple:
    """Cheap, stable key for the current global filter state.

    Page-level `@st.cache_data` helpers that derive from `filtered_articles()`
    take this as their argument instead of the frame itself: hashing three
    small tuples costs nothing, while hashing the frame meant serializing it
    (~3.5MB, ~20ms) on every cache *lookup*, several times per rerun.
    """
    return (
        st.session_state.get("global_year_range"),
        tuple(st.session_state.get("global_sources", ())),
        tuple(st.session_state.get("global_venues", ())),
        st.session_state.get("global_min_margin"),
        tuple(st.session_state.get("global_publication_categories", ())),
    )


@st.cache_data(ttl=60)
def row_counts() -> pd.DataFrame:
    return layer_row_counts()


@st.cache_data(ttl=60)
def chunks() -> pd.DataFrame:
    return load_chunks()


@st.cache_data(ttl=60)
def chunk_search_data() -> pd.DataFrame:
    """Binary-first chunk rows for the on-demand search box in
    `pages/quality.py` -- callers should only invoke this once a query is
    actually submitted, not on a plain page render (see `data.load_chunk_search_data`).
    """
    return load_chunk_search_data()


@st.cache_data(ttl=60)
def search_configs() -> pd.DataFrame:
    return load_search_configs()


@st.cache_data(ttl=60)
def semantics() -> pd.DataFrame:
    """Per-article semantic signals, keyed by DOI.

    Callers join it on `doi` rather than expecting semantic signals to be
    embedded in the active article table.
    """
    return load_semantics()


@st.cache_data(ttl=60)
def duplicate_pairs() -> pd.DataFrame:
    return load_duplicate_pairs()


@st.cache_data(ttl=60)
def duplicate_overrides() -> pd.DataFrame:
    return load_duplicate_overrides()


def with_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the semantic signals onto an article frame, by DOI.

    Returns `df` untouched when the `semantic` stage has never run, so every
    page keeps working on a database that only has the older stages.
    """
    signals = semantics()
    if df.empty or signals.empty or "doi" not in df.columns:
        return df
    columns = ["doi", "relevance_score", "theme_id", "theme_label", "map_x", "map_y"]
    if "offtopic_score" in signals.columns:
        columns.append("offtopic_score")
    merged = df.merge(signals[columns], on="doi", how="left")
    if "offtopic_score" in merged.columns:
        # The screening signal, derived once here so no page repeats the
        # subtraction: how much closer an abstract sits to the review's topic
        # than to the logistics reading of the same query. Zero is the
        # meaningful threshold -- see transform/semantics.py.
        merged["relevance_margin"] = merged["relevance_score"] - merged["offtopic_score"]
    return merged


def _normalize_article_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("authors", "keywords", "sources"):
        if col in df.columns:
            df[col] = df[col].apply(_to_list)
    if "source" not in df.columns and "sources" in df.columns:
        df["source"] = df["sources"].apply(lambda s: s[0] if s else "unknown")
    if "has_abstract" not in df.columns:
        if "abstract" in df.columns:
            df["has_abstract"] = df["abstract"].fillna("").astype(str).str.strip().ne("")
        else:
            df["has_abstract"] = False
    return df


@st.cache_data(ttl=60)
def articles_by_layer() -> dict[str, pd.DataFrame]:
    """Bronze/silver/gold `articles`, each normalized the same way as `articles()`.

    Used by the pipeline/layers page to compare stages directly, rather than
    only seeing the single "best" layer the rest of the dashboard reads.
    """
    return {layer: _normalize_article_frame(df) for layer, df in load_articles_all_layers().items()}


@st.cache_data(ttl=60)
def layer_funnel() -> pd.DataFrame:
    """Raw -> bronze -> silver -> gold article counts, split by source.

    One row per (layer, source) with a `total` row per layer too -- shaped
    for the pipeline/layers funnel chart.
    """
    raw_counts = raw_funnel_counts()
    dropped = bronze_doi_dropped_counts()
    by_layer = articles_by_layer()

    rows = []
    for source in ("ieee", "elsevier"):
        raw_n = raw_counts.get(source, {}).get("csv_rows", 0) or raw_counts.get(source, {}).get(
            "bib_entries", 0
        )
        # IEEE raw volume is CSV-row-driven (the authoritative record list);
        # Elsevier has no CSV, so its raw count is bib entries.
        if source == "ieee":
            raw_n = raw_counts.get("ieee", {}).get("csv_rows", 0)
        else:
            raw_n = raw_counts.get("elsevier", {}).get("bib_entries", 0)

        bronze_df = by_layer.get("bronze", pd.DataFrame())
        silver_df = by_layer.get("silver", pd.DataFrame())
        gold_df = by_layer.get("gold", pd.DataFrame())

        bronze_n = (
            int((bronze_df["source"] == source).sum()) if "source" in bronze_df.columns else 0
        )
        silver_n = (
            int(silver_df["sources"].apply(lambda s, source=source: source in s).sum())
            if "sources" in silver_df.columns
            else 0
        )
        gold_n = (
            int(gold_df["sources"].apply(lambda s, source=source: source in s).sum())
            if "sources" in gold_df.columns
            else 0
        )
        rows.append(
            {
                "source": source,
                "raw": raw_n,
                "bronze": bronze_n,
                "dropped_no_doi": dropped.get(source, 0),
                "silver": silver_n,
                "gold": gold_n,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def _article_bundle() -> tuple[str, pd.DataFrame, dict[str, object]]:
    layer, df, status = select_articles_layer()
    return layer, _normalize_article_frame(df), status


def articles() -> tuple[str, pd.DataFrame]:
    """Canonical Gold articles, or an explicitly reported degraded fallback."""
    layer, df, _ = _article_bundle()
    return layer, df


def article_population_status() -> dict[str, object]:
    """Describe the selected layer without changing the established article API."""
    _, _, status = _article_bundle()
    return status


@st.cache_data(ttl=60)
def filter_articles(
    year_range: tuple[int, int] | None = None,
    sources: tuple[str, ...] = (),
    venues: tuple[str, ...] = (),
    min_margin: float | None = None,
    publication_categories: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply the dashboard-wide filters to the best article layer.

    The filter is cached separately from the database read so changing a
    widget never causes another MySQL round-trip. Empty source/venue tuples
    mean "all", which also keeps the state valid when a partial layer lacks a
    column. The source frame is read from `articles()` (itself cached) rather
    than taken as an argument, so the cache key stays a few small values
    instead of a serialized copy of the whole frame.

    `min_margin` drops articles that sit closer to the logistics reading of
    "distribution system planning" than to the review's own topic -- the
    contrastive margin from `transform/semantics.py`, where 0 is the natural
    cut. Articles with no score yet are always kept: a missing signal must
    never silently shrink the corpus.
    """
    _, df = articles()
    if df.empty:
        return df
    if min_margin is not None:
        scored = with_semantics(df)
        # `relevance_margin` needs `offtopic_score`, which a database whose
        # last `semantic` run predates the contrastive anchor doesn't have.
        column = "relevance_margin" if "relevance_margin" in scored.columns else "relevance_score"
        if column in scored.columns:
            df = scored[scored[column].isna() | scored[column].ge(min_margin)]
    filtered = df.copy()
    if year_range and "year" in filtered.columns:
        if isinstance(year_range, (int, float)):
            year_range = (int(year_range), int(year_range))
        years = pd.to_numeric(filtered["year"], errors="coerce")
        filtered = filtered.loc[years.ge(year_range[0]) & years.le(year_range[1])]
    if sources and "source" in filtered.columns:
        filtered = filtered[filtered["source"].isin(sources)]
    if venues and "venue" in filtered.columns:
        filtered = filtered[filtered["venue"].isin(venues)]
    if publication_categories and "publication_category" in filtered.columns:
        filtered = filtered[filtered["publication_category"].isin(publication_categories)]
    return filtered.reset_index(drop=True)


def filtered_articles() -> tuple[str, pd.DataFrame]:
    """Return the best available article layer after global sidebar filters."""
    layer, df = articles()
    if df.empty:
        return layer, df
    return layer, filter_articles(*filter_signature())


@st.cache_data(ttl=60)
def author_table(filter_sig: tuple) -> pd.DataFrame:
    """Canonicalized author rows for the active global-filter signature."""
    from lake_research_map.dashboard.analytics import (
        author_display_name,
        canonical_author,
        explode_authors_with_position,
    )

    _, articles_df = filtered_articles()
    exploded = explode_authors_with_position(articles_df)
    if exploded.empty:
        return exploded
    exploded["author_key"] = exploded["author"].apply(canonical_author)
    exploded = exploded[exploded["author_key"] != ""]
    display_names = exploded.groupby("author_key")["author"].apply(author_display_name)
    exploded["author_display"] = exploded["author_key"].map(display_names)
    return exploded


@st.cache_data(ttl=60)
def author_year_matrix_cached(filter_sig: tuple) -> pd.DataFrame:
    """Author-by-year output matrix for the active global-filter signature."""
    from lake_research_map.dashboard.analytics import author_year_matrix

    _, articles_df = filtered_articles()
    return author_year_matrix(articles_df)


@st.cache_data(ttl=60)
def volume_forecast(source: str | None):
    """Cached publication-volume forecast for one source or the full corpus."""
    from lake_research_map.dashboard.forecasting import fit_and_forecast, yearly_counts

    _, articles_df = articles()
    return fit_and_forecast(yearly_counts(articles_df, source=source))


@st.cache_data(ttl=60)
def keyword_forecasts(min_occurrences: int = 20):
    """Fit and cache temporal models for sufficiently frequent keywords."""
    from lake_research_map.dashboard.analytics import explode_keywords
    from lake_research_map.dashboard.forecasting import TRAIN_END_YEAR, fit_and_forecast

    _, articles_df = articles()
    exploded = explode_keywords(articles_df)
    if exploded.empty or "year" not in exploded.columns:
        return "no_keywords", [], {}, None

    counts = exploded["keyword"].value_counts()
    eligible = counts[counts >= min_occurrences].index.tolist()
    if not eligible:
        return "none_eligible", [], {}, None

    rows = []
    results = {}
    final_year = None
    for keyword in eligible:
        keyword_df = exploded[exploded["keyword"] == keyword]
        series = keyword_df.groupby(keyword_df["year"].astype("Int64")).size()
        series.index = series.index.astype(int)
        result = fit_and_forecast(series)
        if result.insufficient_data:
            continue
        results[keyword] = result
        final_year = result.forecast_years[-1]
        # The comparison baseline is the last COMPLETE year, which moves with
        # TRAIN_END_YEAR. Hard-coding it meant the "observed" column silently
        # became a partial year as soon as the calendar rolled over.
        observed = float(series.get(TRAIN_END_YEAR, series.tail(1).iloc[0] if len(series) else 0))
        forecast = float(result.forecast_values[-1])
        rows.append(
            {
                "keyword": keyword,
                f"{TRAIN_END_YEAR} (actual)": observed,
                f"{final_year} (forecast)": forecast,
                "variation": forecast - observed,
                "model": result.chosen_model,
            }
        )
    return "ok", rows, results, final_year


@st.cache_data(ttl=60)
def filter_chunks(dois: tuple[str, ...]) -> pd.DataFrame:
    """Keep RAG chunks belonging to the currently filtered article set."""
    chunks_df = chunks()
    if chunks_df.empty or "doi" not in chunks_df.columns or not dois:
        return chunks_df.iloc[0:0].copy() if not dois else chunks_df.copy()
    return chunks_df[chunks_df["doi"].isin(dois)].reset_index(drop=True)


def filtered_chunks() -> pd.DataFrame:
    """Return chunks scoped to the globally filtered article DOI set."""
    _, article_df = filtered_articles()
    if article_df.empty or "doi" not in article_df.columns:
        return chunks().iloc[0:0].copy()
    dois = tuple(article_df["doi"].dropna().astype(str).unique())
    return filter_chunks(dois)


def require_articles() -> pd.DataFrame:
    """Return the articles frame, or render the empty state and stop the page."""
    _, all_articles = articles()
    if all_articles.empty:
        st.warning(
            "No data found in the `lit_bronze`, `lit_silver`, or `lit_gold` layers yet.\n\n"
            "Perform the pipeline (handbar or sidebar buttons) "
            "`uv run lake-research-map --stage all`) and reload this page."
        )
        st.stop()
    status = article_population_status()
    if not status["is_canonical"]:
        reasons = " ".join(status["fallback_reasons"])
        st.warning(
            f"Degraded mode: analyses use the `{status['layer']}` layer instead of the curated "
            f"curada Gold. {reasons}"
        )
    _, df = filtered_articles()
    if df.empty:
        st.warning(
            "No article corresponds to global filters. "
            "Expand the year, the source or the journal in the sidebar."
        )
        st.stop()
    return df


@st.cache_data(ttl=None)
def _qualis_reference() -> pd.DataFrame:
    """Cached wrapper over `qualis.load_qualis_reference()`.

    Parsing the national CAPES xlsx (171k rows across every evaluation area,
    filtered down to ENGENHARIAS IV) takes ~9s via openpyxl; without this it
    reran on every `venue_qualis_map` cache miss, since that cache is keyed
    by the venues tuple rather than by this file.
    """
    return qualis.load_qualis_reference()


@st.cache_data(ttl=None)
def venue_qualis_map(venues: tuple[str, ...]) -> pd.DataFrame:
    """CAPES/Qualis (ENGENHARIAS IV) classification for each of `venues`.

    Cached indefinitely (the reference file doesn't change during a session) --
    see `dashboard.qualis` for the fuzzy-matching rationale. Prefer
    `all_venue_qualis_map()` from page code: matching against the full corpus
    once means toggling a global filter never re-triggers rapidfuzz matching.
    """
    return qualis.match_venues_to_qualis(list(venues), _qualis_reference())


@st.cache_data(ttl=60)
def all_venue_qualis_map() -> pd.DataFrame:
    """CAPES/Qualis classification for every venue in the (unfiltered) best
    article layer.

    Matching against the full corpus's venues once, rather than whatever
    subset survives the current global filters, means switching a
    year/source/venue filter never re-triggers `venue_qualis_map`'s fuzzy
    matching -- callers should filter the result by venue membership locally
    instead of calling `venue_qualis_map` with a filtered venue tuple.
    """
    _, df = articles()
    if df.empty or "venue" not in df.columns:
        return pd.DataFrame(columns=["venue", "matched_title", "estrato", "score"])
    venues = tuple(sorted(df["venue"].dropna().unique()))
    return venue_qualis_map(venues)


@st.cache_data(ttl=120)
def abstract_embeddings() -> tuple[list[str], np.ndarray] | None:
    """Load abstract chunk embeddings matrix and DOIs for vector projections and novelty."""
    import numpy as np

    from lake_research_map.dashboard.search import _parse_embedding

    df = load_abstract_embeddings_data()
    if df.empty:
        return None
    dois = []
    vecs = []
    for _, r in df.iterrows():
        vec = _parse_embedding(r.get("embedding_bin"))
        if vec is not None:
            dois.append(r["doi"])
            vecs.append(vec)
    if not vecs:
        return None
    return dois, np.stack(vecs)


@st.cache_data(ttl=300)
def semantic_stability(points: tuple[tuple[str, str, float, float], ...]) -> dict | None:
    """Bootstrap-ARI for the theme solution and trustworthiness for the shown map.

    Cached here, not in the page, for two reasons. It refits KMeans 30 times,
    which is seconds of work that must not repeat on every widget interaction;
    and the key is `(doi, theme, x, y)` tuples rather than the embedding matrix,
    so Streamlit hashes a few thousand small values instead of a 3115x384 array.

    Clustering stability is measured in the SAME PCA space the themes were built
    in (`transform/semantics.reduced_space`), not in the raw 384-dimensional
    space: measuring the stability of a geometry nothing was clustered in would
    answer a different question, and it is an order of magnitude cheaper.
    Trustworthiness is measured against the coordinates actually on screen,
    whichever projection the user selected, for the same reason.
    """
    import numpy as np

    from lake_research_map.dashboard.analytics import semantic_stability_diagnostics
    from lake_research_map.transform.semantics import reduced_space

    if not points:
        return None
    embeddings = abstract_embeddings()
    if embeddings is None:
        return None
    dois, matrix = embeddings

    by_doi = {doi: (theme, x, y) for doi, theme, x, y in points}
    rows = [index for index, doi in enumerate(dois) if doi in by_doi]
    if len(rows) < 10:
        return None
    labels = np.array([by_doi[dois[index]][0] for index in rows])
    projection = np.array([by_doi[dois[index]][1:] for index in rows], dtype=float)

    reduced = reduced_space(matrix[rows])
    if reduced.ndim != 2 or reduced.shape[1] < 2:
        return None
    result = semantic_stability_diagnostics(reduced, labels, projection)
    if result.get("valid"):
        # The sweep runs on the same reduced space the themes were found in,
        # so the numbers on the page describe the search that actually chose k.
        from lake_research_map.transform.semantics import theme_sweep

        result["k_sweep"] = theme_sweep(reduced)
    return result


@st.cache_data(ttl=300)
def alternative_projections() -> dict[str, pd.DataFrame]:
    """Compute available named projections without relabeling fallback algorithms."""
    from lake_research_map.transform.semantics import project_pca_2d, project_umap

    embs = abstract_embeddings()
    if embs is None:
        return {}
    dois, matrix = embs
    if len(dois) == 0:
        return {}

    pca_coords = project_pca_2d(matrix)
    projections = {
        "PCA 2D": pd.DataFrame({"doi": dois, "map_x": pca_coords[:, 0], "map_y": pca_coords[:, 1]}),
    }
    try:
        umap_coords = project_umap(matrix)
    except ImportError:
        pass
    else:
        projections["UMAP"] = pd.DataFrame(
            {"doi": dois, "map_x": umap_coords[:, 0], "map_y": umap_coords[:, 1]}
        )
    return projections


@st.cache_data(ttl=300)
def semantic_novelty_scores() -> pd.DataFrame:
    """Compute semantic isolation (k-NN distance) for all articles."""
    from lake_research_map.transform.semantics import compute_semantic_novelty

    embs = abstract_embeddings()
    if embs is None:
        return pd.DataFrame(columns=["doi", "novelty_score"])
    dois, matrix = embs
    scores = compute_semantic_novelty(matrix)
    return pd.DataFrame({"doi": dois, "novelty_score": scores})
