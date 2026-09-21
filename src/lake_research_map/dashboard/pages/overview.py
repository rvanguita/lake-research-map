"""📊 Visão Geral — o corpus em resumo: volume, fontes, período temporal e crescimento."""

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
from lake_research_map.dashboard.theme import SOURCE_COLORS


def render() -> None:
    page_header(
        "📊",
        "Visão geral",
        'Pipeline de revisão sistemática de literatura sobre "planejamento de redes de distribuição de energia" — '
        "IEEE Xplore e ScienceDirect/Elsevier consolidados através das "
        "camadas raw → bronze → silver → gold.",
    )

    layer, _ = loaders.filtered_articles()
    articles_df = loaders.require_articles()
    chunks_df = loaders.filtered_chunks()

    years_df = _years(articles_df)
    active_year_range = st.session_state.get("global_year_range")
    active_sources = st.session_state.get("global_sources", [])
    active_venues = st.session_state.get("global_venues", [])
    filter_summary = []
    if active_year_range:
        filter_summary.append(f"ano {active_year_range[0]}–{active_year_range[1]}")
    if active_sources:
        filter_summary.append(f"{len(active_sources)} fontes")
    if active_venues:
        filter_summary.append(f"{len(active_venues)} periódicos")
    summary_text = ". ".join(filter_summary) if filter_summary else "sem filtros ativos"
    hero_banner(
        "Resumo executivo",
        f"O corpus mostra <b>{len(articles_df):,}</b> artigos em <b>{len(years_df):,}</b> anos, com "
        f"<b>{len(articles_df['source'].unique()) if 'source' in articles_df.columns else 0}</b> fontes "
        f"e filtros ativos em <b>{summary_text}</b> · camada ativa: <b>{layer}</b>.",
    )

    st.divider()
    _charts_grid(articles_df)

    if not years_df.empty:
        st.divider()
        st.subheader("Recência do Corpus")
        last_year = int(years_df["year"].max())
        recent_share = (years_df["year"] >= last_year - RECENT_WINDOW_YEARS + 1).mean()
        metric_row(
            [
                (
                    "🗓️ Período temporal coberto",
                    f"{int(years_df['year'].min())}–{last_year}",
                    None,
                ),
                (
                    f"🆕 Publicados nos últimos {RECENT_WINDOW_YEARS} anos",
                    f"{recent_share:.1%}",
                    None,
                ),
                ("📚 Artigos com ano identificado", f"{len(years_df):,}", None),
            ]
        )
        st.caption(
            f"Janela de {RECENT_WINDOW_YEARS} anos (mesma usada na página Pesquisadores) — indica se a "
            "revisão se apoia em literatura recente ou se concentra em trabalhos pioneiros clássicos."
        )

    st.divider()
    _numbers_summary(articles_df, years_df, chunks_df)


def _charts_grid(articles_df: pd.DataFrame) -> None:
    _source_distribution_pie(articles_df)


def _source_distribution_pie(articles_df: pd.DataFrame) -> None:
    st.subheader("Distribuição por Base / Fonte")
    if "source" not in articles_df.columns:
        st.info("Coluna 'source' não disponível nesta camada.")
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
        hovertemplate="<b>%{label}</b>: %{value:,} artigos (%{percent})<extra></extra>",
    )
    render_chart(
        fig,
        caption="As duas bases não possuem sobreposição: nenhum DOI se repete entre elas, de modo "
        "que cada artigo pertence exclusivamente a uma editora.",
    )


def _numbers_summary(
    articles_df: pd.DataFrame, years_df: pd.DataFrame, chunks_df: pd.DataFrame
) -> None:
    st.subheader("📋 Resumo em números")

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

    st.markdown("**Cobertura**")
    metric_row(
        [
            ("📄 Artigos no Corpus", f"{n_total:,}", None),
            ("🔗 Com DOI", f"{n_doi:,}", f"{n_doi / n_total:.0%}"),
            ("📝 Com Resumo", f"{n_abstract:,}", f"{n_abstract / n_total:.0%}"),
            ("📎 Com PDF Vinculado", f"{n_pdf:,}", f"{n_pdf / n_total:.0%}"),
        ]
    )

    st.markdown("**Qualidade bibliométrica**")
    metric_row(
        [
            (
                "📚 Refs. por artigo",
                f"{mean_refs:,.1f}" if mean_refs is not None else "N/D",
                None,
            ),
            (
                "⭐ Citações médias",
                f"{mean_citations:,.1f}" if mean_citations is not None else "N/D",
                None,
            ),
            ("🧩 Chunks RAG", f"{len(chunks_df):,}", None),
            ("🧠 Cobertura de texto", f"{n_pdf / n_total:.0%}" if n_total else "N/D", None),
        ]
    )

    st.markdown("**Vocabulário**")
    n_venues = articles_df["venue"].nunique() if "venue" in articles_df else 0
    n_authors = _unique_list_values(articles_df, "authors")
    n_keywords = _unique_list_values(articles_df, "keywords")
    metric_row(
        [
            ("🗓️ Anos cobertos", f"{len(years_df):,}", None),
            ("📰 Periódicos / eventos", f"{n_venues:,}", None),
            ("👥 Autores identificados", f"{n_authors:,}", None),
            ("🏷️ Keywords únicas", f"{n_keywords:,}", None),
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
