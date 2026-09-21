"""Corpus overview: volume, sources, time span, and publication categories."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    RECENT_WINDOW_YEARS,
    valid_years,
)
from lake_research_map.dashboard.components import (
    hero_banner,
    metric_row,
    page_header,
    render_chart,
)
from lake_research_map.dashboard.theme import (
    PUBLICATION_CATEGORY_COLORS,
    PUBLICATION_CATEGORY_LABELS,
    SOURCE_COLORS,
)


def render() -> None:
    page_header(
        ":material/dashboard:",
        "Overview",
        'Systematic literature review pipeline for "electric power distribution network planning" — '
        "IEEE Xplore and ScienceDirect/Elsevier consolidated through the "
        "Raw → Bronze → Silver → Gold layers.",
    )

    layer, _ = loaders.filtered_articles()
    articles_df = loaders.require_articles()
    chunks_df = loaders.filtered_chunks()

    years_df = _years(articles_df)
    active_year_range = st.session_state.get("global_year_range")
    active_sources = st.session_state.get("global_sources", [])
    active_venues = st.session_state.get("global_venues", [])
    active_categories = st.session_state.get("global_publication_categories", [])
    filter_summary = []
    if active_year_range:
        filter_summary.append(f"years {active_year_range[0]}–{active_year_range[1]}")
    if active_sources:
        filter_summary.append(f"{len(active_sources)} sources")
    if active_venues:
        filter_summary.append(f"{len(active_venues)} venues")
    if active_categories:
        filter_summary.append(f"{len(active_categories)} publication categories")
    summary_text = ", ".join(filter_summary) if filter_summary else "no active filters"
    hero_banner(
        "Executive summary",
        f"The corpus contains <b>{len(articles_df):,}</b> articles across <b>{len(years_df):,}</b> years, "
        f"from <b>{len(articles_df['source'].unique()) if 'source' in articles_df.columns else 0}</b> sources. "
        f"Filter scope: <b>{summary_text}</b> · active layer: <b>{layer}</b>.",
    )

    st.divider()
    _charts_grid(articles_df)

    if not years_df.empty:
        st.divider()
        st.subheader("Corpus recency")
        last_year = int(years_df["year"].max())
        recent_share = (years_df["year"] >= last_year - RECENT_WINDOW_YEARS + 1).mean()
        metric_row(
            [
                (
                    "Time span",
                    f"{int(years_df['year'].min())}–{last_year}",
                    None,
                ),
                (
                    f"Published in the last {RECENT_WINDOW_YEARS} years",
                    f"{recent_share:.1%}",
                    None,
                ),
                ("Articles with a known year", f"{len(years_df):,}", None),
            ]
        )
        st.caption(
            f"The {RECENT_WINDOW_YEARS}-year window is also used on the Researchers page and shows "
            "whether the review is grounded in recent literature or concentrated on foundational work."
        )

    st.divider()
    _numbers_summary(articles_df, years_df, chunks_df)


def _charts_grid(articles_df: pd.DataFrame) -> None:
    source_col, category_col = st.columns(2)
    with source_col:
        _source_distribution_pie(articles_df)
    with category_col:
        _publication_category_distribution(articles_df)


def _source_distribution_pie(articles_df: pd.DataFrame) -> None:
    st.subheader("Distribution by source")
    if "source" not in articles_df.columns:
        st.info("Column 'source' is unavailable in this layer.")
        return

    by_source = articles_df["source"].value_counts().rename_axis("source").reset_index(name="count")
    fig = px.pie(
        by_source,
        names="source",
        values="count",
        color="source",
        color_discrete_map=SOURCE_COLORS,
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br><b>%{value:,} (%{percent})</b>",
        hovertemplate="<b>%{label}</b>: %{value:,} articles (%{percent})<extra></extra>",
    )
    render_chart(
        fig,
        caption="Counts use the normalized primary source after DOI-level consolidation.",
    )


def _publication_category_distribution(articles_df: pd.DataFrame) -> None:
    st.subheader("Distribution by publication category")
    if "publication_category" not in articles_df.columns:
        st.info("Publication category is unavailable in this layer.")
        return
    counts = (
        articles_df["publication_category"]
        .value_counts()
        .reindex(PUBLICATION_CATEGORY_LABELS, fill_value=0)
        .rename_axis("publication_category")
        .reset_index(name="articles")
    )
    counts["category_label"] = counts["publication_category"].map(PUBLICATION_CATEGORY_LABELS)
    fig = px.bar(
        counts,
        x="articles",
        y="category_label",
        color="publication_category",
        color_discrete_map=PUBLICATION_CATEGORY_COLORS,
        orientation="h",
        labels={"articles": "Articles", "category_label": "Publication category"},
    )
    fig.update_traces(
        texttemplate="%{x:,}",
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:,} articles<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    render_chart(
        fig,
        caption="Review is evaluated first, followed by conference, journal, and other publication types.",
    )


def _numbers_summary(
    articles_df: pd.DataFrame, years_df: pd.DataFrame, chunks_df: pd.DataFrame
) -> None:
    st.subheader("Summary metrics")

    n_total = len(articles_df)
    mean_refs = (
        articles_df["reference_count"].dropna().mean()
        if "reference_count" in articles_df and articles_df["reference_count"].notna().any()
        else None
    )
    mean_citations = (
        articles_df["citation_count"].dropna().mean()
        if "citation_count" in articles_df and articles_df["citation_count"].notna().any()
        else None
    )
    n_doi = (
        int(articles_df["doi"].fillna("").astype(str).str.strip().ne("").sum())
        if "doi" in articles_df
        else n_total
    )
    n_abstract = (
        int(articles_df["abstract"].fillna("").astype(str).str.strip().ne("").sum())
        if "abstract" in articles_df
        else 0
    )
    n_pdf = (
        int(articles_df["has_pdf"].fillna(False).astype(bool).sum())
        if "has_pdf" in articles_df
        else 0
    )

    st.markdown("**Coverage**")
    metric_row(
        [
            ("Corpus articles", f"{n_total:,}", None),
            ("With DOI", f"{n_doi:,}", f"{n_doi / n_total:.0%}"),
            ("With abstract", f"{n_abstract:,}", f"{n_abstract / n_total:.0%}"),
            ("With linked PDF", f"{n_pdf:,}", f"{n_pdf / n_total:.0%}"),
        ]
    )

    st.markdown("**Bibliometric quality**")
    metric_row(
        [
            (
                "References per article",
                f"{mean_refs:,.1f}" if mean_refs is not None else "N/A",
                None,
            ),
            (
                "Mean citations",
                f"{mean_citations:,.1f}" if mean_citations is not None else "N/A",
                None,
            ),
            ("RAG chunks", f"{len(chunks_df):,}", None),
            ("Full-text coverage", f"{n_pdf / n_total:.0%}" if n_total else "N/A", None),
        ]
    )

    st.markdown("**Vocabulary**")
    n_venues = articles_df["venue"].nunique() if "venue" in articles_df else 0
    n_authors = _unique_list_values(articles_df, "authors")
    n_keywords = _unique_list_values(articles_df, "keywords")
    metric_row(
        [
            ("Years covered", f"{len(years_df):,}", None),
            ("Venues / events", f"{n_venues:,}", None),
            ("Identified authors", f"{n_authors:,}", None),
            ("Unique keywords", f"{n_keywords:,}", None),
        ]
    )


def _years(articles_df: pd.DataFrame) -> pd.DataFrame:
    years = valid_years(articles_df).dropna()
    if years.empty:
        return pd.DataFrame()
    years_df = articles_df.loc[years.index].copy()
    years_df["year"] = years.astype(int)
    return years_df


def _unique_list_values(articles_df: pd.DataFrame, column: str) -> int:
    if column not in articles_df.columns:
        return 0
    return len(
        {
            str(value).strip().casefold()
            for values in articles_df[column]
            if isinstance(values, list)
            for value in values
            if str(value).strip()
        }
    )
