"""📅 Produção ao Longo do Tempo — volume, acumulado por periódico e colaboração."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    OTHERS_LABEL,
    cumulative_by_source,
    cumulative_by_venue,
    source_counts_by,
    valid_years,
)
from lake_research_map.dashboard.charts import source_bars, source_lines, stacked_area
from lake_research_map.dashboard.components import page_header, render_chart, require_columns
from lake_research_map.dashboard.qualis import ESTRATO_ORDER, NOT_CLASSIFIED, QUALIS_AREA
from lake_research_map.dashboard.theme import (
    venue_color_map,
)

TOP_VENUES_PER_SOURCE = 8
TOP_VENUES_CUMULATIVE = 10


def render() -> None:
    page_header(
        "📅",
        "Produção e periódicos",
        "Volume anual, crescimento acumulado e evolução temporal da produção científica qualificada.",
    )

    articles_df = loaders.require_articles()

    if not require_columns(articles_df, ["year"], "Coluna 'year' não disponível nesta camada."):
        return
    if not articles_df["year"].notna().any():
        st.info("Coluna 'year' vazia nesta camada.")
        return

    tab_volume, tab_acumulado, tab_qualis = st.tabs(
        ["📅 Volume Anual", "📈 Crescimento Acumulado", "🎓 Estratos CAPES/Qualis"]
    )

    with tab_volume:
        _volume_by_year(articles_df)

    with tab_acumulado:
        _cumulative_production(articles_df)

    with tab_qualis:
        _volume_by_year_qualis(articles_df)


def _volume_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("Volume de publicações por ano")
    years_df = articles_df.copy()
    years_df["year"] = valid_years(years_df)
    years_df = years_df.dropna(subset=["year"]).astype({"year": int})

    by_year = source_counts_by(years_df, "year").sort_values("year")
    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Ano de publicação",
        yaxis_title="Quantidade de artigos",
    )
    render_chart(
        fig,
        caption="A altura empilhada mostra a contribuição de cada base (IEEE e Elsevier); a linha Total "
        "soma as duas.",
    )


def _volume_by_year_qualis(articles_df: pd.DataFrame) -> None:
    st.subheader("🎓 Volume de publicações por ano — classificação CAPES/Qualis (até B2)")
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
            f"Nenhum artigo em periódico classificado até B2 (CAPES/Qualis, área {QUALIS_AREA}) "
            "nesta camada/filtro."
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
            "year": "Ano de publicação",
            "count": "Quantidade de artigos",
            "estrato": "Classificação",
        },
    )
    fig.update_traces(hovertemplate="Ano %{x}<br>%{data.name}: %{y:,} artigos<extra></extra>")
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Ano de publicação",
        yaxis_title="Quantidade de artigos",
        legend_title_text="Classificação",
    )
    render_chart(
        fig,
        caption=f"Inclui apenas periódicos classificados de A1 até B2 no CAPES/Qualis (área {QUALIS_AREA}, "
        "quadriênio 2017-2020); periódicos B3 ou piores, e os não classificados, ficam fora deste "
        "gráfico.",
    )


def _cumulative_production(articles_df: pd.DataFrame) -> None:
    st.subheader("📈 Crescimento acumulado")
    st.caption(
        "Publicações acumuladas ano a ano — total e por base — seguidas da composição acumulada por "
        "periódico."
    )
    sub_total, sub_venue = st.tabs(["🌐 Total", "📰 Por Periódico"])

    with sub_total:
        cum = cumulative_by_source(articles_df)
        if cum.empty:
            st.info("Sem anos válidos para o acumulado.")
        else:
            fig = source_lines(
                cum,
                "year",
                title="Publicações acumuladas por ano",
                y_title="Artigos acumulados",
            )
            fig.update_layout(xaxis_title="Ano de publicação")
            render_chart(
                fig,
                caption=f"Ao final do período, o corpus acumula {int(cum['total'].iloc[-1]):,} artigos "
                f"({int(cum['ieee'].iloc[-1]):,} IEEE, {int(cum['elsevier'].iloc[-1]):,} Elsevier).",
            )

    with sub_venue:
        scope_label = st.segmented_control(
            "Escopo",
            options=["Total", "IEEE", "Elsevier"],
            default="Total",
            key="prod_cumulative_scope",
        )
        # segmented_control allows deselecting the current pill, returning
        # None -- fall back to "Total" for both the lookup and anything
        # displayed, so a deselect never renders a literal "None" in a title.
        scope_label = scope_label or "Total"
        scope = {"Total": "total", "IEEE": "ieee", "Elsevier": "elsevier"}[scope_label]
        venue_cum = cumulative_by_venue(articles_df, top_n=TOP_VENUES_CUMULATIVE, scope=scope)
        if venue_cum.empty:
            st.info("Sem dados de periódico suficientes para o acumulado por revista.")
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
                title=f"Composição acumulada por periódico ({scope_label})",
            )
            fig.update_traces(
                hovertemplate="Ano %{x}<br>%{data.name}: %{y:,.0f} artigos acumulados<extra></extra>"
            )
            fig.update_layout(
                xaxis_title="Ano de publicação",
                yaxis_title="Artigos acumulados",
                legend_title_text="Periódico",
            )
            render_chart(
                fig,
                caption=f"Top {TOP_VENUES_CUMULATIVE} periódicos no escopo selecionado; o restante é agrupado "
                f"em '{OTHERS_LABEL}'.",
            )
