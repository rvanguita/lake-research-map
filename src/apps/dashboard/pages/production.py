"""Publication output over time, cumulative volume, and venue strata."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard import loaders
from apps.dashboard.analytics.counts import (
    OTHERS_LABEL,
    cumulative_by_category,
    cumulative_by_source,
    cumulative_by_venue,
    publication_category_counts_by_year,
    publication_category_totals,
    source_counts_by,
    valid_years,
)
from apps.dashboard.charts import (
    publication_category_bars,
    source_bars,
    source_lines,
    stacked_area,
    topn_hbar,
)
from apps.dashboard.components import metric_row, page_header, render_chart, require_columns
from apps.dashboard.qualis import (
    MATCH_THRESHOLD,
    NOT_CLASSIFIED,
    QUALIS_AREA,
    SOURCE_UNAVAILABLE_MESSAGE,
    TOP_TIERS,
    QualisStrata,
    qualis_strata,
    strata_volume_by_year,
    top_tier_summary,
)
from apps.dashboard.theme import (
    venue_color_map,
)

TOP_VENUES_PER_SOURCE = 8
TOP_VENUES_CUMULATIVE = 10
TOP_TIER_VENUES = 15
QUALIS_VIEWS = (
    "Detailed table",
    "Totals by stratum",
    "Cumulative evolution",
    "A1–A3 subset",
)


def render() -> None:
    page_header(
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
    st.subheader("Publication volume by year — CAPES/Qualis classification through B2")
    if not require_columns(articles_df, ["venue"]) or not articles_df["venue"].notna().any():
        return
    # One guard for every Qualis view below: the dashboard container ships
    # without `data/`, so a missing workbook is an expected state, said once.
    if not loaders.qualis_available():
        st.info(SOURCE_UNAVAILABLE_MESSAGE)
        return

    # Matched once against the full corpus's venues (`all_venue_qualis_map`)
    # and narrowed to the filter scope in `qualis_strata`, so toggling a global
    # filter never re-triggers the rapidfuzz matching pass.
    strata = qualis_strata(articles_df, loaders.all_venue_qualis_map())
    _volume_by_year_qualis(strata)
    st.divider()
    _qualis_views(articles_df, strata)


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
        key="production_annual_source_volume",
    )


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
        key="production_annual_category_volume",
    )


def _volume_by_year_qualis(strata: QualisStrata) -> None:
    by_year_estrato, estrato_order = strata_volume_by_year(strata.with_estrato, through="B2")
    if by_year_estrato.empty:
        st.info(
            f"No article in a venue classified through B2 (CAPES/Qualis area {QUALIS_AREA}) "
            "is available in this layer or filter."
        )
        return

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
        key="production_qualis_volume",
    )


def _qualis_views(articles_df: pd.DataFrame, strata: QualisStrata) -> None:
    qualis_view = (
        st.segmented_control(
            "CAPES/Qualis visualization format",
            options=list(QUALIS_VIEWS),
            default=QUALIS_VIEWS[0],
            key="qualis_view_selector",
        )
        or QUALIS_VIEWS[0]
    )
    if qualis_view == "Detailed table":
        _qualis_table(articles_df, strata)
    elif qualis_view == "Totals by stratum":
        _qualis_totals_chart(strata)
    elif qualis_view == "Cumulative evolution":
        _qualis_cumulative_chart(strata)
    else:
        _qualis_top_tiers(strata)


def _qualis_table(articles_df: pd.DataFrame, strata: QualisStrata) -> None:
    st.subheader("🎓 CAPES/Qualis classification")
    match_df = strata.match_table
    n_classified = int((match_df["estrato"] != NOT_CLASSIFIED).sum())
    n_a1_venues = int((match_df["estrato"] == "A1").sum())
    n_a1_articles = int((strata.with_estrato["estrato"] == "A1").sum())
    pct_a1 = (n_a1_articles / len(articles_df) * 100) if len(articles_df) else 0.0

    metric_row(
        [
            ("📚 Periodicals classified", f"{n_classified}/{len(match_df)}", None),
            ("🥇 A1 journals", f"{n_a1_venues}", None),
            (
                "📄 Articles in A1 journals",
                f"{n_a1_articles:,}",
                f"{pct_a1:.1f}% of the corpus",
            ),
        ]
    )

    st.caption(
        f"Official CAPES/Qualis classification (2017–2020 cycle, area **{QUALIS_AREA}** — the "
        "last per-journal evaluation; from 2025-2028 CAPES evaluates by article rather than by "
        "venue). Venues are matched to the official list by normalised title similarity "
        f"(threshold {MATCH_THRESHOLD:.0f}%), because venue spellings vary across sources "
        "(e.g. `&` vs. `and`, `(Print)`/`(Online)` suffixes). A venue with no confident match "
        f'appears as "{NOT_CLASSIFIED}" -- never as a guessed grade.'
    )
    st.dataframe(
        match_df.rename(
            columns={
                "venue": "Venue (corpus)",
                "matched_title": "Matched title (CAPES)",
                "estrato": "Tier",
                "score": "Similarity (%)",
                "articles": "Articles",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _qualis_totals_chart(strata: QualisStrata) -> None:
    st.subheader("Total publications by classification")
    if strata.totals.empty:
        st.info("No article in this layer/filter has a CAPES/Qualis classification.")
        return
    fig = px.bar(
        x=strata.totals.index,
        y=strata.totals.to_numpy(),
        category_orders={"x": strata.order},
        color=strata.totals.index,
        color_discrete_map=venue_color_map(strata.order, others_label=NOT_CLASSIFIED),
        labels={"x": "Classification", "y": "Articles"},
    )
    fig.update_layout(showlegend=False, barmode="relative")
    render_chart(
        fig,
        caption="Total corpus articles by CAPES/Qualis classification, ordered from best "
        f'("A1") to worst, with "{NOT_CLASSIFIED}" last.',
        key="production_qualis_totals",
    )


def _qualis_cumulative_chart(strata: QualisStrata) -> None:
    st.subheader("Cumulative by classification")
    cum_by_estrato = cumulative_by_category(strata.with_estrato, "estrato", top_n=10)
    if cum_by_estrato.empty:
        st.info("No valid years for the cumulative by classification.")
        return

    fig = stacked_area(
        cum_by_estrato,
        x="year",
        y="cumulative",
        color="estrato",
        color_map=venue_color_map(strata.order, others_label=NOT_CLASSIFIED),
        category_orders={"estrato": strata.order},
        title="Articles cumulative by CAPES/Qualis classification",
    )
    fig.update_traces(
        hovertemplate="Year %{x}<br>%{data.name}: %{y:,.0f} cumulative articles<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Publication year",
        yaxis_title="Cumulative articles",
        # px names the legend after the raw `estrato` column otherwise.
        legend_title_text="Classification",
    )
    render_chart(
        fig,
        caption="Cumulative corpus composition by CAPES/Qualis stratum (area "
        f"{QUALIS_AREA}). Journals without a reliable match remain in "
        f'"{NOT_CLASSIFIED}" and are not mixed into a real stratum.',
        key="production_qualis_cumulative",
    )


def _qualis_top_tiers(strata: QualisStrata) -> None:
    st.subheader("A1–A3 journals")
    summary = top_tier_summary(strata.with_estrato, top_n=TOP_TIER_VENUES)
    if summary["top_venues"].empty:
        st.info("No journal articles classified A1, A2 or A3 in this layer/filter.")
        return

    tier_palette = venue_color_map(list(TOP_TIERS))
    col_volume, col_ranking = st.columns(2)
    with col_volume:
        if summary["by_year"].empty:
            st.info("No valid years for this graph.")
        else:
            fig = source_bars(
                summary["by_year"],
                "year",
                total_line=True,
                title="Annual volume of articles in A1–A3 journals, by source",
            )
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Publication year",
                yaxis_title="Articles",
            )
            render_chart(
                fig,
                caption="Annual volume of articles published in A1, A2, or A3 journals, by source.",
                key="production_qualis_top_tier_volume",
            )
    with col_ranking:
        # One ranking, coloured by tier, with the IEEE/Elsevier split in the
        # hover. It used to be drawn twice -- once stacked by source and once
        # coloured by tier -- over the same 15 journals.
        top = summary["top_venues"].set_index("venue")
        fig = topn_hbar(
            top["articles"],
            color_by=top["estrato"],
            palette=tier_palette,
            title=f"Top {TOP_TIER_VENUES} A1–A3 journals by article count",
            x_title="Articles",
            y_title="Journal",
        )
        for trace in fig.data:
            trace.customdata = top.reindex(trace.y)[["ieee", "elsevier"]].to_numpy()
            trace.hovertemplate = (
                "<b>%{y}</b><br>%{x:,} articles (" + str(trace.name) + ")"
                "<br>IEEE: %{customdata[0]:,} · Elsevier: %{customdata[1]:,}<extra></extra>"
            )
        fig.update_layout(legend_title_text="Classification", barmode="relative")
        render_chart(
            fig,
            caption="Most-published journals across the A1, A2 and A3 tiers, coloured by "
            "CAPES/Qualis classification; hover a bar for its IEEE/Elsevier split.",
            key="production_qualis_top_tier_ranking",
        )

    by_source_tier = summary["by_source_tier"]
    if by_source_tier.empty:
        st.info("Source column not available in this layer.")
        return
    fig = px.bar(
        by_source_tier,
        x="source",
        y="count",
        color="estrato",
        category_orders={"estrato": list(TOP_TIERS), "source": ["ieee", "elsevier"]},
        color_discrete_map=tier_palette,
        barmode="stack",
        title="Distribution of A1/A2/A3 by source (IEEE vs. Elsevier)",
        labels={"source": "Source", "count": "Articles", "estrato": "Classification"},
    )
    fig.update_traces(hovertemplate="<b>%{data.name}</b><br>%{x}: %{y:,} articles<extra></extra>")
    fig.update_layout(
        xaxis_title="Source",
        yaxis_title="Articles",
        legend_title_text="Classification",
    )
    render_chart(
        fig,
        caption="Distribution of A1/A2/A3 articles by source (IEEE vs. Elsevier).",
        key="production_qualis_top_tier_by_source",
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
            key="production_cumulative_source_volume",
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
            key="production_cumulative_venue_composition",
        )
