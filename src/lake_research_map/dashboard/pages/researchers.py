"""👥 Pesquisadores — quem publica, com quem, e quem lidera cada linha de pesquisa."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    RECENT_WINDOW_YEARS,
    analyze_coauthorship_partners,
    author_count_series,
    author_impact_advanced_indices,
    author_m_quotient_analysis,
    author_productivity_trend,
    coauthorship_community_detection,
    cumulative_researchers,
    explode_keywords,
    gini_coefficient,
    graph_advanced_metrics,
    lorenz_curve,
    output_impact_correlation,
    researchers_by_year,
    source_means,
    valid_years,
)
from lake_research_map.dashboard.charts import (
    lorenz_chart,
    source_bars,
    source_lines,
    source_topn_hbar,
    topn_hbar,
)
from lake_research_map.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
)
from lake_research_map.dashboard.theme import CATEGORICAL_PALETTE, theme_tokens

TOP_AUTHORS = 25
MIN_PAPERS_FOR_NETWORK = 4
TOP_NETWORK_AUTHORS = 18


def render() -> None:
    page_header(
        "👥",
        "Pesquisadores e colaboração",
        "Produção, colaboração e linhas de pesquisa dos autores do corpus.",
    )

    articles_df = loaders.require_articles()
    author_rows = loaders.author_table(loaders.filter_signature())

    if author_rows.empty:
        st.info("Coluna 'authors' não disponível ou vazia nesta camada.")
        return

    hero_banner(
        "⚠️ Nomes canonicalizados, não identidades verificadas",
        "Os nomes são normalizados para <b>inicial + sobrenome</b> (ex.: <code>Junyong Liu</code> e "
        "<code>J. Liu</code> viram a mesma chave) porque o IEEE exporta iniciais e a Elsevier nomes "
        "completos. Isso funde grafias do mesmo pesquisador, mas também pode fundir <b>homônimos "
        "diferentes</b> que compartilham inicial e sobrenome — trate os números como uma aproximação, não "
        "como identidade confirmada.",
    )

    n_authors = author_rows["author_key"].nunique()
    per_author_counts = (
        author_rows.groupby("author_key")["doi"].nunique()
        if "doi" in author_rows.columns
        else author_rows.groupby("author_key").size()
    )
    n_5plus = int((per_author_counts >= 5).sum())
    n_3plus = int((per_author_counts >= 3).sum())
    metric_row(
        [
            ("👥 Autores distintos (canonicalizados)", f"{n_authors:,}", None),
            ("🏅 Com ≥5 artigos", f"{n_5plus:,}", None),
            ("📗 Com ≥3 artigos", f"{n_3plus:,}", None),
            (
                "✍️ Média de autores/artigo",
                f"{author_count_series(articles_df).mean():.1f}",
                None,
            ),
        ]
    )

    st.divider()
    section = st.selectbox(
        "Área de análise",
        [
            "Produtividade e ranking",
            "Impacto no corpus",
            "Trajetória temporal",
            "Colaboração e redes",
            "Linhas de pesquisa",
            "Leis bibliométricas",
        ],
    )

    if section == "Produtividade e ranking":
        sub_prolific, sub_lead = st.tabs(["✍️ Mais Prolíficos", "🥇 1º/2º Autor"])
        with sub_prolific:
            _top_authors(author_rows)
        with sub_lead:
            _lead_authors_ranking(author_rows)

    elif section == "Impacto no corpus":
        _render_scientific_leadership_tab(articles_df)

    elif section == "Trajetória temporal":
        (
            sub_active,
            sub_heatmap,
            sub_emerging,
        ) = st.tabs(
            [
                "👥 Volume Anual & Acumulado",
                "🗓️ Heatmap Top Autores",
                "🌱 Emergentes vs. Consolidados",
            ]
        )
        with sub_active:
            active_view = (
                st.segmented_control(
                    "Métrica de Atividade",
                    options=[
                        "Pesquisadores/Ano",
                        "Pesquisadores Acumulado",
                        "1º/2º Autores/Ano",
                        "1º/2º Acumulado",
                    ],
                    default="Pesquisadores/Ano",
                    key="res_active_view_selector",
                )
                or "Pesquisadores/Ano"
            )
            if active_view == "Pesquisadores/Ano":
                _researchers_by_year(articles_df)
            elif active_view == "Pesquisadores Acumulado":
                _cumulative_researchers_chart(articles_df)
            elif active_view == "1º/2º Autores/Ano":
                _lead_authors_by_year(articles_df)
            else:
                _cumulative_lead_authors_chart(articles_df)

        with sub_heatmap:
            _production_heatmap(author_rows)
        with sub_emerging:
            _emerging_vs_established(author_rows)

    elif section == "Colaboração e redes":
        sub_teams, sub_network, sub_cognitive = st.tabs(
            [
                "👥 Equipes & Tamanho",
                "🕸️ Rede de Coautoria (Louvain)",
                "🧠 Distância Cognitiva vs. Impacto",
            ]
        )
        with sub_teams:
            _render_team_collaboration_stats(articles_df)
        with sub_network:
            _coauthorship_network(author_rows)
        with sub_cognitive:
            _cognitive_distance_analysis(articles_df, author_rows)

    elif section == "Linhas de pesquisa":
        (
            sub_leaders,
            sub_trend,
            sub_kw_year,
            sub_kw_cum,
            sub_kw_profile,
            sub_kw_shift,
        ) = st.tabs(
            [
                "🔎 Líderes da Linha",
                "📈 Trajetória Anual",
                "👥 Pesquisadores/Ano",
                "📈 Pesquisadores Acumulados",
                "🏷️ Perfil de Palavras-Chave",
                "🔀 Mudança de Foco",
            ]
        )
        selected_kw, scoped_authors, dois_with_kw = _research_line_selector(
            author_rows, articles_df
        )
        with sub_leaders:
            if selected_kw:
                _research_line_top_authors(selected_kw, scoped_authors)
        with sub_trend:
            if selected_kw:
                _research_line_trend(selected_kw, scoped_authors)
        with sub_kw_year:
            if selected_kw:
                _research_line_researchers_by_year(selected_kw, articles_df, dois_with_kw)
        with sub_kw_cum:
            if selected_kw:
                _research_line_researchers_cumulative(selected_kw, articles_df, dois_with_kw)
                _research_line_articles(articles_df, scoped_authors, selected_kw)

        selected_author = _author_keyword_selector(author_rows)
        working_kw = (
            _author_keyword_working(selected_author, author_rows, articles_df)
            if selected_author
            else None
        )
        with sub_kw_profile:
            if working_kw is not None:
                _author_keyword_overview(selected_author, working_kw)
        with sub_kw_shift:
            if working_kw is not None:
                _author_keyword_shift(working_kw)

    elif section == "Leis bibliométricas":
        sub_table, sub_concentration, sub_trend_table, sub_vs_impact = st.tabs(
            [
                "📋 Tabela Completa",
                "📐 Concentração (Gini/Lorenz)",
                "📈 Tendência de Produtividade",
                "📊 Volume × Impacto",
            ]
        )
        with sub_table:
            matrix = _full_output_table(articles_df)
        with sub_concentration:
            _concentration_analysis(matrix)
        with sub_trend_table:
            _productivity_trend(matrix)
        with sub_vs_impact:
            _volume_vs_impact(author_rows)


def _render_scientific_leadership_tab(articles_df: pd.DataFrame) -> None:
    st.markdown("### Indicadores de impacto no corpus")
    st.caption(
        "Compara pesquisadores somente pelos artigos presentes neste corpus. Os indicadores não representam "
        "a carreira completa. A leitura cruza três métricas "
        "bibliométricas canônicas: o **$h$-index** (consistência de produção e citação), o **$g$-index de Egghe** "
        "(que pontua artigos de impacto desproporcional ou 'blockbusters'), o **$e$-index de Zhang** "
        "(que mede o excesso citacional acumulado além do núcleo $h$), e o "
        "**$m$-quotient de Hirsch** ($m = h / \\text{anos observados no corpus}$)."
    )

    auth_df = author_impact_advanced_indices(articles_df, min_papers=2)
    m_df = author_m_quotient_analysis(articles_df, min_papers=2)

    if auth_df.empty:
        st.info("Autores com produção mínima de 2 artigos não encontrados.")
        return

    top_g = auth_df.iloc[0]
    top_m = m_df.iloc[0] if not m_df.empty else None

    metric_row(
        [
            (
                "🥇 Maior g-index",
                f"{top_g['author']} (g={top_g['g_index']})",
                f"h-index: {top_g['h_index']}",
            ),
            (
                "🔥 Maior Excesso Citacional (e)",
                f"{auth_df.sort_values(by='e_index', ascending=False).iloc[0]['author']}",
                f"e={auth_df.sort_values(by='e_index', ascending=False).iloc[0]['e_index']:.1f}",
            ),
            (
                "⚡ Maior Velocidade (m-quotient)",
                f"{top_m['author']} (m={top_m['m_quotient']})" if top_m is not None else "N/A",
                "h-index por ano de carreira",
            ),
            ("👥 Pesquisadores Analisados", str(len(auth_df)), "≥ 2 artigos no corpus"),
        ]
    )

    col_scatter, col_ebar = st.columns([1.1, 0.9])

    with col_scatter:
        st.markdown("#### 🎯 Dispersão h-index vs. g-index (Egghe)")
        max_val = max(auth_df["g_index"].max(), auth_df["h_index"].max()) + 2

        t = theme_tokens()
        ref_line = t.get("reference_line_subtle", "rgba(180, 180, 180, 0.6)")
        border_color = t.get("point_border", "#FFFFFF")

        fig_hg = go.Figure()
        fig_hg.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="g = h (Produção Uniforme)",
                line={"dash": "dash", "color": ref_line, "width": 1.5},
            )
        )
        hover_hg = [
            f"<b>{r['author']}</b><br>• g-index: {r['g_index']}<br>• h-index: {r['h_index']}<br>• Artigos: {r['papers']}<br>• Citações: {r['total_citations']:,}<br>• e-index: {r['e_index']}"
            for _, r in auth_df.iterrows()
        ]
        fig_hg.add_trace(
            go.Scatter(
                x=auth_df["h_index"],
                y=auth_df["g_index"],
                mode="markers",
                marker={
                    "size": auth_df["papers"].clip(lower=6, upper=24),
                    "color": auth_df["g_h_diff"],
                    "colorscale": "Plasma",
                    "colorbar": {"title": "g - h"},
                    "opacity": 0.85,
                    "line": {"color": border_color, "width": 1},
                },
                hoverinfo="text",
                hovertext=hover_hg,
                name="Pesquisadores",
            )
        )
        fig_hg.update_layout(
            xaxis_title="h-index (Consistência)",
            yaxis_title="g-index de Egghe (Impacto com Blockbusters)",
            height=480,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_hg)

    with col_ebar:
        st.markdown("#### 🚀 Top 12 por Excesso Citacional (e-index de Zhang)")
        top_e = auth_df.sort_values(by="e_index", ascending=True).tail(12)
        fig_e = px.bar(
            top_e,
            x="e_index",
            y="author",
            orientation="h",
            color="total_citations",
            color_continuous_scale="Magma",
            labels={
                "e_index": "e-index de Zhang",
                "author": "Pesquisador",
                "total_citations": "Total Citações",
            },
        )
        fig_e.update_layout(
            xaxis_title="e-index de Zhang",
            yaxis_title="Pesquisador",
            height=480,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_e)

    if not m_df.empty:
        st.markdown(
            "#### ⚡ Top 12 em Velocidade Citacional por Ano de Carreira (m-quotient de Hirsch)"
        )
        top_m_plot = m_df.head(12).sort_values(by="m_quotient", ascending=True)
        fig_m = px.bar(
            top_m_plot,
            x="m_quotient",
            y="author",
            orientation="h",
            color="papers",
            color_continuous_scale="Tealgrn",
            labels={
                "m_quotient": "m-quotient (h-index / Anos de Carreira)",
                "author": "Pesquisador",
                "papers": "Artigos no Corpus",
            },
        )
        fig_m.update_layout(
            xaxis_title="m-quotient (h / Anos de Carreira)",
            yaxis_title="Pesquisador",
            height=420,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_m)

    st.markdown("#### 📋 Ranking Completo de Liderança Científica")
    disp_auth = auth_df.copy()
    if not m_df.empty:
        disp_auth = disp_auth.merge(
            m_df[["author", "first_year", "career_span_years", "m_quotient"]],
            on="author",
            how="left",
        )
        disp_auth.columns = [
            "Pesquisador",
            "Artigos",
            "Total Citações",
            "Média Citações",
            "h-index",
            "g-index (Egghe)",
            "e-index (Zhang)",
            "Excesso (g - h)",
            "1º Ano Publicação",
            "Anos Carreira",
            "m-quotient",
        ]
    else:
        disp_auth.columns = [
            "Pesquisador",
            "Artigos",
            "Total Citações",
            "Média Citações",
            "h-index",
            "g-index (Egghe)",
            "e-index (Zhang)",
            "Excesso (g - h)",
        ]
    st.dataframe(disp_auth, hide_index=True, width="stretch")


def _render_team_collaboration_stats(articles_df: pd.DataFrame) -> None:
    st.subheader("👥 Tamanho e Distribuição das Equipes de Autores")
    with_authors = articles_df.assign(n_authors=author_count_series(articles_df))
    with_authors = with_authors[with_authors["n_authors"] > 0]
    if with_authors.empty:
        st.info("Coluna 'authors' vazia nesta camada.")
        return

    means = source_means(with_authors, "n_authors")
    solo_pct = float((with_authors["n_authors"] == 1).mean())

    metric_row(
        [
            (
                "📊 Média Geral",
                f"{means['total']:.1f} autores/artigo",
                f"Mediana: {with_authors['n_authors'].median():.0f}",
            ),
            (
                "📘 Média IEEE",
                f"{means['ieee']:.1f}" if means["ieee"] is not None else "N/D",
                "Base IEEE Xplore",
            ),
            (
                "📙 Média Elsevier",
                f"{means['elsevier']:.1f}" if means["elsevier"] is not None else "N/D",
                "Base ScienceDirect",
            ),
            (
                "👤 Autor Único (Solo)",
                f"{solo_pct:.1%}",
                f"{len(with_authors):,} artigos analisados",
            ),
        ]
    )

    col_dist, col_trend = st.columns(2)
    with col_dist:
        dist = (
            with_authors["n_authors"]
            .value_counts()
            .sort_index()
            .rename_axis("authors")
            .reset_index(name="articles")
        )
        fig_dist = px.bar(
            dist,
            x="authors",
            y="articles",
            title="Distribuição do Número de Autores por Artigo",
            labels={"authors": "Autores por artigo", "articles": "Quantidade de artigos"},
            color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
        )
        fig_dist.update_traces(hovertemplate="%{x} autores: %{y:,} artigos<extra></extra>")
        fig_dist.update_layout(
            xaxis_title="Autores por artigo", yaxis_title="Quantidade de artigos"
        )
        render_chart(
            fig_dist,
            caption="Assimetria típica da colaboração científica em engenharia elétrica.",
        )

    with col_trend:
        trend = with_authors.copy()
        trend["year"] = valid_years(trend)
        trend = trend.dropna(subset=["year"]).astype({"year": int})
        trend = trend[trend["year"] >= 2000]
        if not trend.empty:
            if "source" in trend.columns:
                by_year_auth = (
                    trend.groupby(["year", "source"])["n_authors"].mean().reset_index(name="mean")
                )
                fig_trend = px.line(
                    by_year_auth,
                    x="year",
                    y="mean",
                    color="source",
                    markers=True,
                    title="Evolução do Tamanho Médio da Equipe ao Longo dos Anos",
                    labels={"year": "Ano", "mean": "Média de autores", "source": "Base"},
                )
            else:
                by_year_auth = (
                    trend.groupby("year")["n_authors"].mean().rename("mean").reset_index()
                )
                fig_trend = px.line(
                    by_year_auth,
                    x="year",
                    y="mean",
                    markers=True,
                    title="Evolução do Tamanho Médio da Equipe ao Longo dos Anos",
                    labels={"year": "Ano", "mean": "Média de autores"},
                )
            fig_trend.update_layout(xaxis_title="Ano", yaxis_title="Média de autores/artigo")
            render_chart(
                fig_trend,
                caption="Tendência temporal de expansão do tamanho médio das equipes.",
            )


def _top_authors(author_rows: pd.DataFrame) -> None:
    st.subheader("✍️ Autores mais prolíficos (canonicalizado)")
    count_col = "doi" if "doi" in author_rows.columns else "author_display"
    agg = "nunique" if count_col == "doi" else "size"
    counts = author_rows.groupby("author_display")[count_col].agg(agg)
    top_index = counts.sort_values(ascending=False).head(15).index

    if "source" in author_rows.columns:
        scoped = author_rows[author_rows["author_display"].isin(top_index)]
        by_source = (
            scoped.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in by_source.columns:
                by_source[src] = 0
        by_source["total"] = counts.reindex(top_index)
        by_source = by_source.reindex(top_index).reset_index()
        fig = source_topn_hbar(
            by_source, "author_display", x_title="Quantidade de artigos", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
    else:
        fig = topn_hbar(counts.reindex(top_index), x_title="Quantidade de artigos", y_title="Autor")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
    render_chart(fig)


def _lead_authors_ranking(author_rows: pd.DataFrame) -> None:
    st.subheader("🥇 Mais frequentes como 1º ou 2º autor")
    if "position" not in author_rows.columns:
        st.info("Posição do autor na publicação não disponível nesta camada.")
        return

    lead_rows = author_rows[author_rows["position"] <= 1]
    if lead_rows.empty:
        st.info("Nenhum autor em 1ª/2ª posição identificado nesta camada.")
        return

    count_col = "doi" if "doi" in lead_rows.columns else "author_display"
    agg = "nunique" if count_col == "doi" else "size"
    counts = lead_rows.groupby("author_display")[count_col].agg(agg)
    top_index = counts.sort_values(ascending=False).head(15).index

    if "source" in lead_rows.columns:
        scoped = lead_rows[lead_rows["author_display"].isin(top_index)]
        by_source = (
            scoped.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in by_source.columns:
                by_source[src] = 0
        by_source["total"] = counts.reindex(top_index)
        by_source = by_source.reindex(top_index).reset_index()
        fig = source_topn_hbar(
            by_source, "author_display", x_title="Artigos como 1º/2º autor", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
    else:
        fig = topn_hbar(
            counts.reindex(top_index), x_title="Artigos como 1º/2º autor", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
    render_chart(
        fig,
        caption="Contagem combinada de artigos em que o autor aparece na 1ª OU 2ª posição da lista de "
        "autores, na ordem registrada pela fonte (não é alfabética). Posição na publicação é apenas "
        "isso -- uma posição; não indica um papel de autoria específico (a convenção sobre o que 1ª/2ª "
        "posição significa varia por área, e este corpus não registra papéis). Vale o mesmo aviso de "
        "canonicalização do topo da página.",
    )


def _researchers_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("👥 Pesquisadores ativos por ano")
    by_year = researchers_by_year(articles_df)
    if by_year.empty:
        st.info("Sem anos válidos para este gráfico.")
        return

    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Ano de publicação",
        yaxis_title="Pesquisadores distintos",
    )
    render_chart(
        fig,
        caption="Autores canonicalizados distintos que publicaram em cada ano, por base. Um autor "
        "publicando nas duas bases no mesmo ano conta uma vez em cada base, mas apenas uma vez no "
        "Total — por isso o Total pode ser menor que a soma de IEEE + Elsevier.",
    )


def _cumulative_researchers_chart(articles_df: pd.DataFrame) -> None:
    st.subheader("📈 Pesquisadores acumulados")
    cum = cumulative_researchers(articles_df)
    if cum.empty:
        st.info("Sem anos válidos para o acumulado.")
        return

    fig = source_lines(
        cum,
        "year",
        title="Pesquisadores distintos acumulados por ano",
        y_title="Pesquisadores acumulados",
    )
    fig.update_layout(xaxis_title="Ano de publicação")
    render_chart(
        fig,
        caption=f"Cada pesquisador é contado uma única vez, no ano da sua primeira publicação "
        f"identificada no corpus (por base, e no geral para o Total). Ao final do período, o corpus "
        f"acumula {int(cum['total'].iloc[-1]):,} pesquisadores distintos "
        f"({int(cum['ieee'].iloc[-1]):,} IEEE, {int(cum['elsevier'].iloc[-1]):,} Elsevier).",
    )


def _lead_authors_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("🥇 1º/2º autores ativos por ano")
    by_year = researchers_by_year(articles_df, max_position=1)
    if by_year.empty:
        st.info("Sem anos válidos para este gráfico.")
        return

    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Ano de publicação",
        yaxis_title="Pesquisadores distintos (1º/2º autor)",
    )
    render_chart(
        fig,
        caption="Autores canonicalizados distintos que apareceram como 1º ou 2º autor em cada ano, "
        'por base -- não é o mesmo universo do gráfico "Pesquisadores ativos por ano" acima, que '
        "conta qualquer posição na lista de autores.",
    )


def _cumulative_lead_authors_chart(articles_df: pd.DataFrame) -> None:
    st.subheader("📈 1º/2º autores acumulados")
    cum = cumulative_researchers(articles_df, max_position=1)
    if cum.empty:
        st.info("Sem anos válidos para o acumulado.")
        return

    fig = source_lines(
        cum,
        "year",
        title="1º/2º autores distintos acumulados por ano",
        y_title="Pesquisadores acumulados",
    )
    fig.update_layout(xaxis_title="Ano de publicação")
    render_chart(
        fig,
        caption=f"Cada pesquisador é contado uma única vez, no ano da sua primeira aparição como 1º "
        f"ou 2º autor. Ao final do período, o corpus acumula {int(cum['total'].iloc[-1]):,} "
        f"pesquisadores distintos nessa condição ({int(cum['ieee'].iloc[-1]):,} IEEE, "
        f"{int(cum['elsevier'].iloc[-1]):,} Elsevier).",
    )


def _production_heatmap(author_rows: pd.DataFrame) -> None:
    st.subheader("🗓️ Produção por ano — top autores")
    working = author_rows.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).astype({"year": int})
    if working.empty:
        st.info("Sem anos válidos para o heatmap.")
        return

    top_authors = (
        working.groupby("author_display")["doi"]
        .nunique()
        .sort_values(ascending=False)
        .head(TOP_AUTHORS)
        .index
        if "doi" in working.columns
        else working.groupby("author_display")
        .size()
        .sort_values(ascending=False)
        .head(TOP_AUTHORS)
        .index
    )
    scoped = working[working["author_display"].isin(top_authors)]
    pivot = scoped.groupby(["author_display", "year"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(top_authors)

    t = theme_tokens()
    zero_color = t.get("heatmap_zero", "#0b1725")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[zero_color, CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[3]],
        labels={"x": "Ano de publicação", "y": "Autor", "color": "Artigos"},
    )
    fig.update_layout(
        xaxis_title="Ano de publicação",
        yaxis_title="Autor",
        height=max(420, 22 * len(pivot)),
    )
    render_chart(
        fig,
        caption="Linhas com atividade recente indicam pesquisadores ativos; linhas concentradas em anos "
        "antigos indicam quem parou de publicar nesta linha de pesquisa.",
    )


def _emerging_vs_established(author_rows: pd.DataFrame) -> None:
    st.subheader("🌱 Emergentes vs. consolidados")
    working = author_rows.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).astype({"year": int})
    if working.empty:
        st.info("Sem anos válidos para esta análise.")
        return

    last_year = int(working["year"].max())
    by_author = working.groupby("author_display").agg(
        first_year=("year", "min"),
        total_papers=("year", "size"),
        recent_papers=("year", lambda s: (s >= last_year - RECENT_WINDOW_YEARS + 1).sum()),
    )
    by_author = by_author[by_author["total_papers"] >= 2]
    if by_author.empty:
        st.info("Sem autores com produção suficiente para o comparativo.")
        return

    fig = px.scatter(
        by_author.reset_index(),
        x="first_year",
        y="recent_papers",
        size="total_papers",
        color="total_papers",
        color_continuous_scale=["#4a3aa7", "#eda100", "#1baf7a"],
        hover_name="author_display",
        labels={
            "first_year": "Ano da primeira publicação no corpus",
            "recent_papers": f"Artigos nos últimos {RECENT_WINDOW_YEARS} anos",
            "total_papers": "Total de artigos",
        },
    )
    fig.update_layout(coloraxis_showscale=False)
    render_chart(
        fig,
        caption="Quadrante superior direito = pesquisadores recentes já com produção alta (em ascensão); "
        "quadrante inferior esquerdo (ano de estreia antigo, poucos artigos recentes) = atividade "
        "concentrada no passado nesta linha de pesquisa.",
    )


def _correlation_by_source(author_rows: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/N for IEEE, Elsevier, and Total, as one small table."""
    sources: list[tuple[str, str | None]] = [("Total", None)]
    if "source" in author_rows.columns:
        sources = [("IEEE", "ieee"), ("Elsevier", "elsevier"), *sources]
    rows = []
    for label, key in sources:
        subset = author_rows[author_rows["source"] == key] if key else author_rows
        stats = output_impact_correlation(subset, "citation_count")
        rows.append(
            {
                "Base": label,
                "Pearson": f"{stats['pearson']:.2f}" if stats["pearson"] is not None else "—",
                "Spearman": f"{stats['spearman']:.2f}" if stats["spearman"] is not None else "—",
                "Autores": stats["n"],
            }
        )
    return pd.DataFrame(rows)


def _volume_vs_impact(author_rows: pd.DataFrame) -> None:
    st.subheader("📊 Volume × impacto")
    if "citation_count" not in author_rows.columns:
        st.info("Coluna 'citation_count' não disponível nesta camada.")
        return
    stats = output_impact_correlation(author_rows, "citation_count")
    by_author = author_rows.groupby("author_display").agg(
        articles=("author_display", "size"),
        mean_citations=("citation_count", "mean"),
        total_citations=("citation_count", "sum"),
    )
    by_author = by_author[by_author["articles"] >= 2].dropna(subset=["mean_citations"])
    if by_author.empty or stats["pearson"] is None:
        st.info("Sem dados de citação suficientes para autores com ≥2 artigos.")
        return

    st.dataframe(_correlation_by_source(author_rows), hide_index=True, width="stretch")
    fig = px.scatter(
        by_author.reset_index(),
        x="articles",
        y="mean_citations",
        size="total_citations",
        color="mean_citations",
        color_continuous_scale=["#4a3aa7", "#eda100", "#1baf7a"],
        hover_name="author_display",
        title=f"Volume × impacto (Pearson r = {stats['pearson']:.2f})",
        labels={"articles": "Artigos no corpus", "mean_citations": "Citações médias por artigo"},
    )
    fig.update_layout(coloraxis_showscale=False)
    render_chart(
        fig,
        caption="`citation_count` nulo é tratado como 'não coletado' e excluído da média — não como zero. "
        "O tamanho da bolha é o total de citações acumuladas pelo autor. A correlação de Spearman é "
        "incluída por ser mais robusta a distribuições de cauda longa (poucos autores com produção ou "
        "citações muito acima da média), comuns em dados bibliométricos.",
    )


def _full_output_table(articles_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📋 Produção completa por autor e ano")
    matrix = loaders.author_year_matrix_cached(loaders.filter_signature())
    if matrix.empty:
        st.info("Sem anos válidos para montar a tabela.")
        return matrix

    st.caption(
        "Uma linha por autor canonicalizado (ver aviso no topo da página), uma coluna por ano de "
        "publicação válido, mais `total`, `ieee_total` e `elsevier_total` (quebra do total histórico "
        "por base). Contagem por DOI distinto quando disponível, para não contar duas vezes um artigo "
        "em coautoria assinado pelo mesmo autor. **Ordenado do maior para o menor total.**"
    )
    st.dataframe(matrix, hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Baixar tabela completa (CSV)",
        data=matrix.to_csv(index=False).encode("utf-8"),
        file_name="producao_por_autor_ano.csv",
        mime="text/csv",
        key="dl_author_year_matrix",
    )
    return matrix


def _gini_interpretation(gini: float) -> str:
    if gini < 0.3:
        return "baixa concentração — a produção é relativamente distribuída entre os autores"
    if gini > 0.6:
        return "alta concentração — a produção está dominada por poucos autores muito prolíficos"
    return "concentração moderada"


def _concentration_analysis(matrix: pd.DataFrame) -> None:
    st.subheader("📐 Concentração da produção (Gini / curva de Lorenz)")
    if matrix.empty:
        st.info("Sem dados suficientes para esta análise.")
        return

    gini_ieee = gini_coefficient(matrix["ieee_total"])
    gini_elsevier = gini_coefficient(matrix["elsevier_total"])
    gini_total = gini_coefficient(matrix["total"])
    metric_row(
        [
            ("📐 Gini — IEEE", f"{gini_ieee:.2f}", _gini_interpretation(gini_ieee)),
            ("📐 Gini — Elsevier", f"{gini_elsevier:.2f}", _gini_interpretation(gini_elsevier)),
            ("📐 Gini — Total", f"{gini_total:.2f}", _gini_interpretation(gini_total)),
        ]
    )

    fig = lorenz_chart(
        {
            "ieee": lorenz_curve(matrix["ieee_total"]),
            "elsevier": lorenz_curve(matrix["elsevier_total"]),
            "total": lorenz_curve(matrix["total"]),
        }
    )
    render_chart(
        fig,
        caption="Índice de Gini calculado sobre o total histórico por autor, separado por base (0 = "
        "todos publicam o mesmo tanto, 1 = um único autor concentra toda a produção). Quanto mais uma "
        "curva observada se afasta da diagonal de equidade perfeita, mais concentrada é a produção "
        "naquela base.",
    )


def _productivity_trend(matrix: pd.DataFrame) -> None:
    st.subheader("📈 Tendência de produtividade — top autores")
    if matrix.empty:
        st.info("Sem dados suficientes para esta análise.")
        return

    trend_df = author_productivity_trend(matrix, top_n=TOP_AUTHORS)
    st.caption(
        f"Reta de tendência (mínimos quadrados, `numpy.polyfit` grau 1) de artigos por ano para cada "
        f"um dos {TOP_AUTHORS} autores mais prolíficos, usando somente os anos em que o autor "
        "publicou. Classificado como 'crescendo'/'caindo' quando a inclinação ultrapassa ±0,15 "
        "artigo/ano; dentro dessa faixa é 'estável'. Séries deste tamanho (poucos anos ativos) não "
        "sustentam um teste de significância estatística confiável — é um indicador direcional, não "
        "uma previsão."
    )
    display = trend_df.rename(
        columns={
            "author": "Autor",
            "total": "Total histórico",
            "first_year": "Primeiro ano",
            "last_year": "Último ano",
            "active_years": "Anos ativos",
            "slope": "Inclinação (artigos/ano)",
            "trend": "Tendência",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _coauthorship_network(author_rows: pd.DataFrame) -> None:
    st.subheader("🕸️ Rede de coautoria (top autores)")
    if "doi" not in author_rows.columns:
        st.info("Coluna 'doi' não disponível para reconstruir a rede.")
        return

    counts = author_rows.groupby("author_display")["doi"].nunique()
    # A force-directed layout was tried here and, even after several rounds of
    # tuning, still put too many authors too close together to read at 60
    # nodes. A much smaller, fixed circular layout trades "shows everyone" for
    # "every connection is actually legible" -- no physics, no randomness, no
    # possible overlap (evenly spaced points on a circle can't collide).
    top_authors = (
        counts[counts >= MIN_PAPERS_FOR_NETWORK]
        .sort_values(ascending=False)
        .head(TOP_NETWORK_AUTHORS)
        .index
    )
    if len(top_authors) < 3:
        st.info(
            f"Poucos autores com ≥{MIN_PAPERS_FOR_NETWORK} artigos para montar uma rede legível."
        )
        return

    scoped = author_rows[author_rows["author_display"].isin(top_authors)]
    by_doi = scoped.groupby("doi")["author_display"].apply(list)

    graph = nx.Graph()
    graph.add_nodes_from(top_authors)
    for authors in by_doi:
        unique_authors = sorted(set(authors))
        for i in range(len(unique_authors)):
            for j in range(i + 1, len(unique_authors)):
                a, b = unique_authors[i], unique_authors[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)

    # Authors in the top-N by volume with no coauthor also in the top-N add
    # nothing to a *network* view -- drop them rather than scatter meaningless
    # isolated dots around the circle.
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_edges() == 0:
        st.info("Nenhuma coautoria encontrada entre os autores mais produtivos.")
        return

    n = graph.number_of_nodes()
    # Order nodes by a depth-first walk of the graph (starting from the most
    # connected author) rather than alphabetically or by rank -- neighbors in
    # the walk tend to be actual collaborators, so placing them next to each
    # other around the circle keeps most edges short instead of criss-crossing
    # the whole diagram. Any node a DFS from one root can't reach (a separate
    # component) is appended afterwards.
    root = max(graph.degree, key=lambda kv: kv[1])[0]
    order = list(nx.dfs_preorder_nodes(graph, source=root))
    order += [node for node in graph.nodes() if node not in order]

    radius = 9.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {
        node: np.array([radius * np.cos(a), radius * np.sin(a)])
        for node, a in zip(order, angles, strict=True)
    }

    degree = dict(graph.degree())
    nodes = list(graph.nodes())

    communities = coauthorship_community_detection(graph)
    net_metrics = graph_advanced_metrics(graph)
    n_comms = len(set(communities.values())) if communities else 1

    partners_df = analyze_coauthorship_partners(graph, author_rows, communities)
    partners_by_author = (
        partners_df.set_index("author").to_dict(orient="index") if not partners_df.empty else {}
    )

    recurrent_edges = [
        (u, v, int(d.get("weight", 1)))
        for u, v, d in graph.edges(data=True)
        if d.get("weight", 1) >= 2
    ]
    n_recurrent = len(recurrent_edges)
    top_pair = max(recurrent_edges, key=lambda x: x[2]) if recurrent_edges else None
    top_pair_note = (
        f"Mais forte: {top_pair[0]} & {top_pair[1]} ({top_pair[2]} arts)"
        if top_pair
        else "Nenhuma com ≥2 artigos"
    )

    sw_value = (
        f"{net_metrics['small_world_sigma']:.2f}" if net_metrics.get("small_world_sigma") else "—"
    )
    sw_note = (
        f"L={net_metrics['avg_path_length']:.2f} (Topologia Small-World)"
        if net_metrics.get("small_world_sigma") and net_metrics["small_world_sigma"] > 1.0
        else f"Densidade: {net_metrics['density']:.3f}"
    )

    metric_row(
        [
            ("👥 Pesquisadores Conectados", f"{n}", f"{n_comms} comunidades Louvain"),
            (
                "🔗 Conexões Únicas (Pares)",
                f"{graph.number_of_edges()}",
                f"{graph.number_of_edges() - n_recurrent} ocasionais (1 art.)",
            ),
            ("🔁 Parcerias Recorrentes (≥2 arts)", f"{n_recurrent}", top_pair_note),
            ("🌐 Coef. Pequeno Mundo (σ)", sw_value, sw_note),
        ]
    )

    # Edges as separate line segments:
    # Single-article edges are rendered subtly; recurrent partnerships (>= 2 papers)
    # are rendered with higher opacity and thickness to stand out.
    edge_traces = []
    max_w = max((d["weight"] for _, _, d in graph.edges(data=True)), default=1)

    # 1. Single-article connections (w == 1)
    for a, b, d in graph.edges(data=True):
        if d.get("weight", 1) == 1:
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        color="rgba(94,169,255,0.20)",
                        width=1.2,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # 2. Recurrent connections (w >= 2)
    for a, b, d in graph.edges(data=True):
        w = d.get("weight", 1)
        if w >= 2:
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        color="rgba(235,104,52,0.80)",
                        width=2.5 + 4 * (w / max_w),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # 3. Interactive midpoints on recurrent edges to show partner details on hover
    if recurrent_edges:
        mid_x = [(pos[a][0] + pos[b][0]) / 2 for a, b, _ in recurrent_edges]
        mid_y = [(pos[a][1] + pos[b][1]) / 2 for a, b, _ in recurrent_edges]
        recurrent_custom = [[a, b, w] for a, b, w in recurrent_edges]
        edge_traces.append(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker=dict(
                    size=7,
                    color="#eb6834",
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                customdata=recurrent_custom,
                hovertemplate=(
                    "🔁 <b>Parceria Recorrente</b><br>"
                    "👥 %{customdata[0]} ↔ %{customdata[1]}<br>"
                    "📚 <b>%{customdata[2]} artigos</b> em coautoria na rede"
                    "<extra></extra>"
                ),
                name="Parcerias Recorrentes",
                showlegend=False,
            )
        )

    node_colors = [
        CATEGORICAL_PALETTE[communities.get(a, 0) % len(CATEGORICAL_PALETTE)] for a in nodes
    ]

    customdata = []
    for a in nodes:
        p = partners_by_author.get(a, {})
        customdata.append(
            [
                p.get("community", f"#{communities.get(a, 0) + 1}"),
                p.get("articles", 0),
                p.get("network_unique_count", degree[a]),
                p.get("network_recurrent_count", 0),
                p.get("network_recurrent_names", "—"),
                p.get("global_unique_count", 0),
                p.get("global_recurrent_count", 0),
                p.get("global_top_partners", "—"),
            ]
        )

    fig = go.Figure(data=edge_traces)
    fig.add_trace(
        go.Scatter(
            x=[pos[a][0] for a in nodes],
            y=[pos[a][1] for a in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=10, color=theme_tokens()["chart_text"]),
            marker=dict(
                size=[10 + 4 * degree[a] for a in nodes],
                color=node_colors,
                line=dict(width=1.5, color="rgba(255,255,255,0.4)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "🏘️ Comunidade Louvain: <b>%{customdata[0]}</b><br>"
                "📄 Total de artigos no corpus: <b>%{customdata[1]}</b><br>"
                "<br>"
                "🕸️ <b>Na Rede de Top Autores:</b><br>"
                "• Coautores únicos: <b>%{customdata[2]}</b><br>"
                "• Parcerias repetidas (≥2 arts): <b>%{customdata[3]}</b><br>"
                "• Parceiros na rede: %{customdata[4]}<br>"
                "<br>"
                "🌐 <b>No Corpus Global:</b><br>"
                "• Coautores únicos: <b>%{customdata[5]}</b><br>"
                "• Parcerias repetidas: <b>%{customdata[6]}</b><br>"
                "• Top parceiros no corpus: %{customdata[7]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # O layout é um círculo fixo: a posição de um nó não codifica nada, então
    # ticks e grade somem (e `scaleanchor` mantém o círculo redondo). Os eixos
    # ficam nomeados dizendo exatamente isso, em vez de aparecerem anônimos.
    fig.update_layout(
        xaxis=dict(
            title="Posição no layout circular (sem unidade)",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            ticks="",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Posição no layout circular (sem unidade)",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        showlegend=False,
        height=650,
        hovermode="closest",
    )
    render_chart(
        fig,
        caption=f"Layout circular fixo (sem física): os {n} autores com pelo menos uma coautoria entre os "
        f"{TOP_NETWORK_AUTHORS} mais produtivos (≥{MIN_PAPERS_FOR_NETWORK} artigos) ficam igualmente "
        "espaçados ao redor do círculo — a posição não indica proximidade, e a ordem segue uma caminhada "
        "pelo grafo a partir do autor mais conectado, para manter a maioria das conexões como linhas curtas "
        "em vez de cruzarem o desenho inteiro. Linhas laranjas com losango central indicam **parcerias recorrentes** "
        "(≥2 artigos conjuntos), enquanto linhas azuis tênues representam coautorias pontuais (1 artigo). "
        "O tamanho do nó reflete o número de coautores distintos na rede.",
    )

    if not partners_df.empty:
        st.divider()
        st.markdown("##### 👥 Conexões com Autores Únicos vs. Recorrentes (Top Autores)")
        st.caption(
            "Detalhamento quantitativo e nominal das colaborações científicas. "
            "A seção **Na Rede** restringe a análise aos top autores representados no grafo acima; "
            "a seção **No Corpus Global** cobre a totalidade de artigos e colaboradores registrados no banco de dados."
        )
        pr_map = net_metrics.get("pagerank", {})
        close_map = net_metrics.get("closeness", {})
        partners_df["pagerank"] = (
            partners_df["author"].map(pr_map).fillna(0.0).apply(lambda x: f"{x:.4f}")
        )
        partners_df["closeness"] = (
            partners_df["author"].map(close_map).fillna(0.0).apply(lambda x: f"{x:.3f}")
        )

        display_df = partners_df[
            [
                "author",
                "articles",
                "community",
                "network_unique_count",
                "network_recurrent_count",
                "network_recurrent_names",
                "pagerank",
                "closeness",
                "global_unique_count",
                "global_recurrent_count",
                "global_top_partners",
            ]
        ].rename(
            columns={
                "author": "Pesquisador",
                "articles": "Total Artigos",
                "community": "Comunidade",
                "network_unique_count": "Coautores Únicos (Rede)",
                "network_recurrent_count": "Parcerias Repetidas (Rede ≥2)",
                "network_recurrent_names": "Quem são os Parceiros (Rede)",
                "pagerank": "PageRank",
                "closeness": "Proximidade (Closeness)",
                "global_unique_count": "Coautores Únicos (Global)",
                "global_recurrent_count": "Parcerias Repetidas (Global ≥2)",
                "global_top_partners": "Top Parceiros no Corpus",
            }
        )
        st.dataframe(display_df, hide_index=True, width="stretch")


def _cognitive_distance_analysis(articles_df: pd.DataFrame, author_rows: pd.DataFrame) -> None:
    st.subheader("🧠 Distância Cognitiva nas Coautorias vs. Impacto em Citações")
    st.caption(
        "A distância cognitiva mede a dispersão conceitual entre os coautores de um artigo "
        "no espaço vetorial do corpus. Permite testar empiricamente se parcerias interdisciplinares "
        "alcançam maior repercussão científica."
    )
    signals = loaders.semantics()
    if signals.empty or "map_x" not in signals.columns or "map_y" not in signals.columns:
        st.info(
            "Sinais semânticos não disponíveis para cálculo da distância cognitiva. Execute `--stage semantic`."
        )
        return

    scoped = loaders.with_semantics(articles_df)
    valid_articles = scoped.dropna(subset=["map_x", "map_y", "citation_count"]).copy()
    if len(valid_articles) < 10:
        st.info("Artigos insuficientes com dados conjuntos de semântica e citações.")
        return

    merged_author_art = author_rows.merge(valid_articles[["doi", "map_x", "map_y"]], on="doi")
    author_pos = merged_author_art.groupby("author_key")[["map_x", "map_y"]].mean()

    records = []
    for doi, grp in merged_author_art.groupby("doi"):
        authors = grp["author_key"].unique()
        if len(authors) >= 2:
            matched = author_pos.index.intersection(authors)
            if len(matched) >= 2:
                coords = author_pos.loc[matched].to_numpy()
                diffs = coords[:, None, :] - coords[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=-1))
                i_upper = np.triu_indices(len(coords), k=1)
                mean_dist = float(np.mean(dists[i_upper]))
                row = valid_articles[valid_articles["doi"] == doi].iloc[0]
                records.append(
                    {
                        "doi": doi,
                        "title": str(row.get("title", "—"))[:80],
                        "year": row.get("year"),
                        "cognitive_distance": round(mean_dist, 3),
                        "citation_count": int(row.get("citation_count", 0)),
                        "team_size": len(authors),
                    }
                )

    if len(records) < 5:
        st.info("Poucos artigos com múltiplos autores posicionados no espaço semântico.")
        return

    dist_df = pd.DataFrame(records)
    r_pearson = float(
        dist_df["cognitive_distance"].corr(dist_df["citation_count"], method="pearson")
    )
    r_spearman = float(
        dist_df["cognitive_distance"].corr(dist_df["citation_count"], method="spearman")
    )

    metric_row(
        [
            ("👥 Artigos Multiautoria Avaliados", f"{len(dist_df):,}", None),
            (
                "📐 Distância Cognitiva Média",
                f"{dist_df['cognitive_distance'].mean():.2f}",
                None,
            ),
            (
                "📈 Correlação de Pearson (r)",
                f"{r_pearson:+.3f}",
                "Relação linear",
            ),
            (
                "📊 Correlação de Spearman (ρ)",
                f"{r_spearman:+.3f}",
                "Relação monotônica",
            ),
        ]
    )

    fig = px.scatter(
        dist_df,
        x="cognitive_distance",
        y="citation_count",
        size="team_size",
        hover_name="title",
        hover_data={
            "cognitive_distance": True,
            "citation_count": True,
            "team_size": True,
            "year": True,
        },
        title="Distância Cognitiva entre Coautores vs. Citações Recebidas",
        labels={
            "cognitive_distance": "Distância Cognitiva da Equipe (Dispersão Semântica)",
            "citation_count": "Citações Recebidas",
            "team_size": "Autores",
        },
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
        opacity=0.7,
    )
    render_chart(
        fig,
        caption="Cada ponto representa um artigo em coautoria. A distância cognitiva quantifica "
        "o grau de complementaridade conceitual entre os históricos de pesquisa de seus autores.",
    )


def _research_line_selector(
    author_rows: pd.DataFrame, articles_df: pd.DataFrame
) -> tuple[str | None, pd.DataFrame, set]:
    """Shared keyword selector for all "Exploração" research-line sub-tabs.

    Returns `(selected_keyword, scoped_authors, dois_with_kw)`; `selected_keyword`
    is None when there's nothing to show (missing data or no valid selection).
    """
    st.subheader("🔎 Quem lidera esta linha de pesquisa")
    kw_exploded = explode_keywords(articles_df)
    if kw_exploded.empty:
        st.info("Coluna 'keywords' não disponível nesta camada.")
        return None, author_rows.iloc[0:0], set()

    top_keywords = kw_exploded["keyword"].value_counts().head(60).index.tolist()
    selected = st.selectbox("Selecione uma palavra-chave:", options=top_keywords)
    if not selected:
        return None, author_rows.iloc[0:0], set()

    dois_with_kw = set(kw_exploded.loc[kw_exploded["keyword"] == selected, "doi"].dropna())
    scoped_authors = (
        author_rows[author_rows["doi"].isin(dois_with_kw)]
        if "doi" in author_rows.columns
        else author_rows.iloc[0:0]
    )
    if scoped_authors.empty:
        st.info("Nenhum autor associado a esse termo nesta camada.")
        return None, scoped_authors, dois_with_kw

    return selected, scoped_authors, dois_with_kw


def _research_line_top_authors(selected: str, scoped_authors: pd.DataFrame) -> None:
    leaders = (
        scoped_authors.groupby("author_display")["doi"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    fig = topn_hbar(
        leaders,
        title=f"Autores mais produtivos em '{selected}'",
        x_title="Quantidade de artigos",
        y_title="Autor",
    )
    render_chart(fig)


def _research_line_trend(selected: str, scoped_authors: pd.DataFrame) -> None:
    trend_df = scoped_authors.copy()
    trend_df["year"] = valid_years(trend_df)
    trend_df = trend_df.dropna(subset=["year"]).astype({"year": int})
    if trend_df.empty:
        st.info("Sem anos válidos para a trajetória.")
        return

    by_year = trend_df.groupby("year")["doi"].nunique().reset_index(name="articles")
    fig = px.line(
        by_year,
        x="year",
        y="articles",
        markers=True,
        title=f"Trajetória anual de '{selected}'",
        labels={"year": "Ano", "articles": "Artigos"},
    )
    fig.update_traces(line_color=CATEGORICAL_PALETTE[2])
    render_chart(fig)


def _research_line_researchers_by_year(
    selected: str, articles_df: pd.DataFrame, dois_with_kw: set
) -> None:
    keyword_articles = articles_df[articles_df["doi"].isin(dois_with_kw)]
    kw_by_year = researchers_by_year(keyword_articles)
    if kw_by_year.empty:
        st.info("Sem anos válidos para este gráfico.")
        return

    fig = source_bars(kw_by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Ano de publicação",
        yaxis_title="Pesquisadores distintos",
    )
    render_chart(
        fig,
        caption=f"Pesquisadores distintos que publicaram em '{selected}' a cada ano, por base.",
    )


def _research_line_researchers_cumulative(
    selected: str, articles_df: pd.DataFrame, dois_with_kw: set
) -> None:
    keyword_articles = articles_df[articles_df["doi"].isin(dois_with_kw)]
    kw_cum = cumulative_researchers(keyword_articles)
    if kw_cum.empty:
        st.info("Sem anos válidos para o acumulado.")
        return

    fig = source_lines(
        kw_cum,
        "year",
        title=f"Pesquisadores acumulados em '{selected}'",
        y_title="Pesquisadores acumulados",
    )
    fig.update_layout(xaxis_title="Ano de publicação")
    render_chart(
        fig,
        caption="Total acumulado de pesquisadores distintos que já publicaram em "
        f"'{selected}' até cada ano ({int(kw_cum['total'].iloc[-1]):,} ao final do período).",
    )


def _research_line_articles(
    articles_df: pd.DataFrame, scoped_authors: pd.DataFrame, selected: str
) -> None:
    top_dois = scoped_authors["doi"].unique()
    subset = articles_df[articles_df["doi"].isin(top_dois)]
    if "citation_count" in subset.columns:
        subset = subset.sort_values("citation_count", ascending=False, na_position="last")
    article_table(
        subset,
        ["title", "year", "venue", "source", "citation_count", "doi"],
        download_key=f"lideres_{selected.replace(' ', '_')}",
    )


def _author_keyword_selector(author_rows: pd.DataFrame) -> str | None:
    """Shared author selector for the "Perfil de Palavras-Chave"/"Mudança de Foco" sub-tabs."""
    st.subheader("🏷️ Perfil de palavras-chave por autor")
    counts = (
        author_rows.groupby("author_display")["doi"].nunique()
        if "doi" in author_rows.columns
        else author_rows.groupby("author_display").size()
    )
    eligible = counts[counts >= 3].sort_values(ascending=False)
    if eligible.empty:
        st.info("Nenhum autor com pelo menos 3 artigos nesta camada.")
        return None

    selected_author = st.selectbox(
        "Selecione um autor (mínimo 3 artigos):", options=eligible.index.tolist()
    )
    return selected_author or None


def _author_keyword_working(
    selected_author: str, author_rows: pd.DataFrame, articles_df: pd.DataFrame
) -> pd.DataFrame | None:
    author_dois = set(
        author_rows.loc[author_rows["author_display"] == selected_author, "doi"].dropna()
    )
    subset = articles_df[articles_df["doi"].isin(author_dois)]
    kw_exploded = explode_keywords(subset)
    if kw_exploded.empty:
        st.info(f"Nenhuma palavra-chave registrada para {selected_author}.")
        return None

    working = kw_exploded.copy()
    working["year"] = valid_years(working)
    return working.dropna(subset=["year"]).astype({"year": int})


def _author_keyword_overview(selected_author: str, working: pd.DataFrame) -> None:
    top_terms = working["keyword"].value_counts().head(10)
    fig = topn_hbar(
        top_terms,
        title=f"Palavras-chave dominantes de {selected_author}",
        x_title="Quantidade de menções",
        y_title="Palavra-chave",
    )
    render_chart(fig)


def _author_keyword_shift(working: pd.DataFrame) -> None:
    if working["year"].nunique() < 2:
        st.info("Anos insuficientes para comparar início vs. fim da carreira no corpus.")
        return

    split_year = int(working["year"].median())
    early = working[working["year"] <= split_year]["keyword"].value_counts()
    late = working[working["year"] > split_year]["keyword"].value_counts()
    all_terms = set(early.index) | set(late.index)
    compare = pd.DataFrame(
        {
            "early": early.reindex(all_terms, fill_value=0),
            "late": late.reindex(all_terms, fill_value=0),
        }
    )
    compare = (
        compare[(compare["early"] + compare["late"]) > 0]
        .sort_values("late", ascending=False)
        .head(10)
    )
    fig = go.Figure()
    fig.add_bar(
        x=compare.index,
        y=compare["early"],
        name=f"até {split_year}",
        marker_color=CATEGORICAL_PALETTE[0],
    )
    fig.add_bar(
        x=compare.index,
        y=compare["late"],
        name=f"após {split_year}",
        marker_color=CATEGORICAL_PALETTE[2],
    )
    fig.update_layout(
        barmode="group",
        title="Mudança de foco: início vs. fim da carreira no corpus",
        xaxis_title="Palavra-chave",
        yaxis_title="Menções no período",
    )
    render_chart(fig)
