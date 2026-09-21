"""Publication output over time, cumulative volume, and venue strata."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    OTHERS_LABEL,
    cumulative_by_source,
    cumulative_by_venue,
    publication_category_counts_by_year,
    publication_category_totals,
    source_counts_by,
    valid_years,
)
from lake_research_map.dashboard.charts import (
    publication_category_bars,
    source_bars,
    source_lines,
    stacked_area,
)
from lake_research_map.dashboard.components import page_header, render_chart, require_columns
from lake_research_map.dashboard.qualis import ESTRATO_ORDER, NOT_CLASSIFIED, QUALIS_AREA
from lake_research_map.dashboard.theme import (
    venue_color_map,
)

TOP_VENUES_PER_SOURCE = 8
TOP_VENUES_CUMULATIVE = 10


def render() -> None:
    page_header(
        ":material/calendar_month:",
        "Production and venues",
        "Annual volume, cumulative growth, and the temporal evolution of classified research output.",
    )

    articles_df = loaders.require_articles()

    if not require_columns(articles_df, ["year"], "Column 'year' is unavailable in this layer."):
        return
    if not articles_df["year"].notna().any():
        st.info("Column 'year' is empty in this layer.")
        return

    _summary_section(articles_df)
    st.divider()
    _publication_types_section(articles_df)
    st.divider()
    _venues_section(articles_df)


def _summary_section(articles_df: pd.DataFrame) -> None:
    st.subheader("Corpus summary and total trend")
    totals = publication_category_totals(articles_df)
    with st.container(horizontal=True, gap="small"):
        for label, value in (
            ("Total", totals.sum()),
            ("Articles", totals["journal"]),
            ("Conference", totals["conference"]),
            ("Review", totals["review"]),
            ("Other", totals["other"]),
        ):
            st.metric(label, f"{int(value):,}", border=True)
    _volume_by_year(articles_df)


def _publication_types_section(articles_df: pd.DataFrame) -> None:
    st.subheader("Publication types")
    _volume_by_year_category(articles_df)


def _venues_section(articles_df: pd.DataFrame) -> None:
    st.subheader("Venues and CAPES/Qualis")
    _cumulative_production(articles_df)
    _volume_by_year_qualis(articles_df)


def _volume_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("Publication volume by year")
    years_df = articles_df.copy()
    years_df["year"] = valid_years(years_df)
    years_df = years_df.dropna(subset=["year"]).astype({"year": int})

    by_year = source_counts_by(years_df, "year").sort_values("year")
    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Publication year",
        yaxis_title="Articles",
    )
    render_chart(
        fig,
        caption="Stacked bars show the contribution of each source; the Total line sums both.",
    )
    _volume_by_year_category(years_df)


def _volume_by_year_category(years_df: pd.DataFrame) -> None:
    if "publication_category" not in years_df.columns:
        st.info("Publication category is unavailable in this layer.")
        return
    grouped = publication_category_counts_by_year(years_df)
    if grouped.empty:
        st.info("No valid publication years are available for the category breakdown.")
        return
    fig = publication_category_bars(
        grouped,
        title="Annual output by publication type",
        x_title="Publication year",
        y_title="Articles",
        category_labels={
            "journal": "Articles",
            "conference": "Conference",
            "review": "Review",
            "other": "Other",
        },
    )
    fig.update_layout(legend_title_text="Publication type")
    render_chart(
        fig,
        caption="Bars separate Articles, Conference, Review, and Other; the Total line reconciles all four types.",
    )


def _volume_by_year_qualis(articles_df: pd.DataFrame) -> None:
    st.subheader("Publication volume by year — CAPES/Qualis classification through B2")
    if not require_columns(articles_df, ["venue"]) or not articles_df["venue"].notna().any():
        return

    # Matches against the full corpus's venues (cached indefinitely), not a
    # filtered subset -- see `loaders.all_venue_qualis_map` -- so toggling a
    # global filter never re-triggers the rapidfuzz matching pass. `.map()`
    # below simply ignores venues absent from the current filter scope.
    match_df = loaders.all_venue_qualis_map()
    venue_to_estrato = dict(zip(match_df["venue"], match_df["estrato"], strict=True))

    # ESTRATO_ORDER is best-to-worst with the unclassified bucket last; "up to
    # B2" is everything from A1 through B2 in that ranking.
    allowed = ESTRATO_ORDER[: ESTRATO_ORDER.index("B2") + 1]

    years_df = articles_df.copy()
    years_df["year"] = valid_years(years_df)
    years_df = years_df.dropna(subset=["year"]).astype({"year": int})
    years_df["estrato"] = years_df["venue"].map(venue_to_estrato)
    years_df = years_df[years_df["estrato"].isin(allowed)]

    if years_df.empty:
        st.info(
            f"No article in a venue classified through B2 (CAPES/Qualis area {QUALIS_AREA}) "
            "is available in this layer or filter."
        )
        return

    by_year_estrato = (
        years_df.groupby(["year", "estrato"]).size().reset_index(name="count").sort_values("year")
    )
    estrato_order = [e for e in allowed if e in years_df["estrato"].unique()]
    fig = px.bar(
        by_year_estrato,
        x="year",
        y="count",
        color="estrato",
        category_orders={"estrato": estrato_order},
        color_discrete_map=venue_color_map(estrato_order, others_label=NOT_CLASSIFIED),
        barmode="stack",
        labels={
            "year": "Publication year",
            "count": "Articles",
            "estrato": "Classification",
        },
    )
    fig.update_traces(hovertemplate="Year %{x}<br>%{data.name}: %{y:,} articles<extra></extra>")
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Publication year",
        yaxis_title="Articles",
        legend_title_text="Classification",
    )
    render_chart(
        fig,
        caption=f"Includes only venues classified A1 through B2 in CAPES/Qualis (area {QUALIS_AREA}, "
        "2017–2020 cycle); B3 or lower and unclassified venues are excluded.",
    )


def _cumulative_production(articles_df: pd.DataFrame) -> None:
    st.caption(
        "Year-by-year cumulative publications, overall and by source, followed by cumulative venue composition."
    )
    cum = cumulative_by_source(articles_df)
    if cum.empty:
        st.info("No valid years are available for cumulative output.")
    else:
        fig = source_lines(
            cum,
            "year",
            title="Cumulative publications by year",
            x_title="Publication year",
            y_title="Cumulative articles",
        )
        render_chart(
            fig,
            caption=f"At the end of the period, the corpus contains {int(cum['total'].iloc[-1]):,} articles "
            f"({int(cum['ieee'].iloc[-1]):,} IEEE, {int(cum['elsevier'].iloc[-1]):,} Elsevier).",
        )

    scope_label = (
        st.segmented_control(
            "Venue scope",
            options=["Total", "IEEE", "Elsevier"],
            default="Total",
            key="prod_cumulative_scope",
        )
        or "Total"
    )
    scope = {"Total": "total", "IEEE": "ieee", "Elsevier": "elsevier"}[scope_label]
    venue_cum = cumulative_by_venue(articles_df, top_n=TOP_VENUES_CUMULATIVE, scope=scope)
    if venue_cum.empty:
        st.info("Insufficient venue data for cumulative composition.")
    else:
        venue_order = (
            venue_cum.groupby("venue")["cumulative"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        venue_order = [v for v in venue_order if v != OTHERS_LABEL] + (
            [OTHERS_LABEL] if OTHERS_LABEL in venue_order else []
        )
        fig = stacked_area(
            venue_cum,
            x="year",
            y="cumulative",
            color="venue",
            color_map=venue_color_map(venue_order, others_label=OTHERS_LABEL),
            category_orders={"venue": venue_order},
            title=f"Cumulative composition by venue ({scope_label})",
        )
        fig.update_traces(
            hovertemplate="Year %{x}<br>%{data.name}: %{y:,.0f} cumulative articles<extra></extra>"
        )
        fig.update_layout(
            xaxis_title="Publication year",
            yaxis_title="Cumulative articles",
            legend_title_text="Venue",
        )
        render_chart(
            fig,
            caption=f"Top {TOP_VENUES_CUMULATIVE} venues in the selected scope; all remaining venues "
            f"are grouped as '{OTHERS_LABEL}'.",
        )
