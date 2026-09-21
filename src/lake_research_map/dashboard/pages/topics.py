"""🏷️ Tópicos e Periódicos — onde o corpus publica, sobre o que, e como os temas evoluíram."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    conceptual_atypicality_analysis,
    cumulative_by_category,
    detect_structural_breaks,
    dynamic_topic_ctfidf,
    explode_keywords,
    source_counts_by,
    valid_years,
    zipf_law_analysis,
)
from lake_research_map.dashboard.charts import (
    source_bars,
    source_topn_hbar,
    stacked_area,
    topn_hbar,
)
from lake_research_map.dashboard.components import (
    article_table,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from lake_research_map.dashboard.qualis import (
    ESTRATO_ORDER,
    MATCH_THRESHOLD,
    NOT_CLASSIFIED,
    QUALIS_AREA,
)
from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    TREND_DOWN_COLOR,
    TREND_UP_COLOR,
    venue_color_map,
)

TOP_KEYWORDS_TREND = 12
TREND_MIN_YEAR = 2010
MIN_KEYWORD_OCCURRENCES = 15


def render() -> None:
    page_header(
        "🏷️",
        "Tópicos e estrutura científica",
        "Onde o corpus publica, sobre o que, e como os temas evoluíram ao longo do tempo.",
    )

    articles_df = loaders.require_articles()

    kw_lists = None
    all_keywords: list[str] = []
    if "keywords" in articles_df.columns:
        kw_lists = articles_df["keywords"].apply(
            lambda kws: sorted({str(k).strip().lower() for k in kws if str(k).strip()})
        )
        all_keywords = sorted({k for kws in kw_lists for k in kws})

    tab_venues, tab_keywords, tab_explorer, tab_trends = st.tabs(
        ["📚 Periódicos", "🏷️ Palavras-Chave", "🔎 Explorador", "📈 Evolução Temporal"]
    )

    with tab_venues:
        (
            sub_ranking,
            sub_qualis,
            sub_bradford,
            sub_semantic_venues,
        ) = st.tabs(
            [
                "🏆 Ranking & Impacto",
                "📋 Classificação CAPES/Qualis",
                "🎯 Zonas de Bradford",
                "🗺️ Perfil Semântico",
            ]
        )
        with sub_ranking:
            _top_venues(articles_df)

        if require_columns(articles_df, ["venue"]) and articles_df["venue"].notna().any():
            match_df, with_estrato, totals_by_estrato, estrato_order = _qualis_match_data(
                articles_df
            )
            with sub_qualis:
                qualis_view = (
                    st.segmented_control(
                        "Formato de Visualização CAPES/Qualis",
                        options=[
                            "Tabela Detalhada",
                            "Totais por Estrato",
                            "Evolução Acumulada",
                            "Subconjunto A1-A3",
                        ],
                        default="Tabela Detalhada",
                        key="qualis_view_selector",
                    )
                    or "Tabela Detalhada"
                )
                if qualis_view == "Tabela Detalhada":
                    _qualis_table(
                        articles_df, match_df, with_estrato[with_estrato["estrato"] == "A1"]
                    )
                elif qualis_view == "Totais por Estrato":
                    _qualis_totals_chart(totals_by_estrato, estrato_order)
                elif qualis_view == "Evolução Acumulada":
                    _qualis_cumulative_chart(with_estrato, estrato_order)
                else:
                    _qualis_a1_a3_combined(with_estrato)

            with sub_bradford:
                _bradford_analysis(articles_df)
            with sub_semantic_venues:
                _semantic_venues_analysis(articles_df)

    with tab_keywords:
        sub_top, sub_vocab, sub_ctfidf, sub_atypical = st.tabs(
            [
                "🏷️ Top 20 Palavras-Chave",
                "📊 Estrutura do Vocabulário (Zipf & Métricas)",
                "🧬 Vocabulário Dinâmico (c-TF-IDF)",
                "Combinações incomuns",
            ]
        )
        with sub_top:
            _top_keywords(articles_df)
        with sub_vocab:
            if all_keywords:
                _keyword_stats(kw_lists, all_keywords)
                st.divider()
                _zipf_analysis(articles_df)
            else:
                st.info("Nenhuma palavra-chave identificada nesta camada.")
        with sub_ctfidf:
            _dynamic_ctfidf_analysis(articles_df)
        with sub_atypical:
            _conceptual_atypicality_tab(articles_df)

    with tab_explorer:
        if all_keywords:
            _keyword_explorer(articles_df, kw_lists, all_keywords)
        else:
            st.info("Nenhuma palavra-chave identificada nesta camada.")

    with tab_trends:
        if not all_keywords:
            st.info("Nenhuma palavra-chave identificada nesta camada.")
        else:
            kw_year = _prepare_keyword_trend_data(articles_df)
            if kw_year is None:
                st.info("Dados insuficientes para analisar tendências temporais.")
            else:
                sub_share, sub_slope, sub_first, sub_breaks = st.tabs(
                    [
                        "📈 Participação Anual",
                        "🔀 Ascensão vs. Declínio",
                        "🌱 Vocabulário Novo vs. Fundacional",
                        "⚡ Quebras Estruturais (Changepoints)",
                    ]
                )
                with sub_share:
                    _topic_share_area(kw_year)
                with sub_slope:
                    _rising_falling(kw_year)
                with sub_first:
                    _first_appearance(kw_year)
                with sub_breaks:
                    _structural_breaks_tab(articles_df)


def _top_venues(articles_df: pd.DataFrame) -> None:
    st.subheader("Periódicos e Eventos")
    if not require_columns(articles_df, ["venue"]) or not articles_df["venue"].notna().any():
        return

    rank_mode = (
        st.segmented_control(
            "Critério de Destaque",
            options=["Volume de Artigos", "Impacto Médio de Citações"],
            default="Volume de Artigos",
            key="topics_venue_rank_mode",
        )
        or "Volume de Artigos"
    )

    if rank_mode == "Volume de Artigos":
        top_venues = articles_df["venue"].dropna().value_counts().head(15)
        modal_source = None
        if "source" in articles_df.columns:
            modal_source = (
                articles_df.dropna(subset=["venue"])
                .groupby("venue")["source"]
                .agg(lambda s: s.mode().iat[0])
            )

        fig = topn_hbar(
            top_venues,
            color_by=modal_source,
            x_title="Quantidade de artigos",
            y_title="Periódico / Evento",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos publicados<extra></extra>")
        render_chart(
            fig,
            caption="Concentração editorial do corpus — cada periódico é colorido pela base predominante dos "
            "artigos publicados nele.",
        )
    else:
        if not require_columns(articles_df, ["citation_count"]):
            return
        cited_venues = articles_df.dropna(subset=["citation_count", "venue"])
        venue_impact = (
            cited_venues.groupby("venue")["citation_count"]
            .agg(articles="size", mean="mean")
            .query("articles >= 3")
            .sort_values("mean", ascending=False)
            .head(15)
        )
        if venue_impact.empty:
            st.info(
                "Nenhum periódico com artigos suficientes com contagem de citações nesta camada."
            )
            return

        modal_source = (
            cited_venues[cited_venues["venue"].isin(venue_impact.index)]
            .groupby(["venue", "source"], observed=True)
            .size()
            .sort_values(ascending=False)
            .reset_index()
            .drop_duplicates("venue")
            .set_index("venue")["source"]
            if "source" in cited_venues.columns
            else None
        )
        fig = topn_hbar(
            venue_impact["mean"],
            color_by=modal_source,
            x_title="Média de citações por artigo",
            y_title="Periódico / Evento",
        )
        for trace in fig.data:
            trace.customdata = venue_impact["articles"].reindex(trace.y).to_numpy().reshape(-1, 1)
            trace.hovertemplate = "<b>%{y}</b><br>%{x:.1f} citações/artigo (%{customdata[0]:,} artigos analisados)<extra></extra>"
        render_chart(
            fig,
            caption="Média de citações por artigo para veículos com pelo menos 3 publicações no corpus.",
        )


def _top_keywords(articles_df: pd.DataFrame) -> None:
    st.subheader("Palavras-Chave")
    if not require_columns(articles_df, ["keywords"]):
        return

    kw_exploded = explode_keywords(articles_df)
    if kw_exploded.empty:
        st.info("Nenhuma palavra-chave identificada nesta camada.")
        return

    top_20_kw = kw_exploded["keyword"].value_counts().head(20).index.tolist()
    filtered_kw = kw_exploded[kw_exploded["keyword"].isin(top_20_kw)]

    if "source" in filtered_kw.columns:
        grouped = filtered_kw.groupby(["keyword", "source"]).size().unstack(fill_value=0)
        for s in ("ieee", "elsevier"):
            if s not in grouped.columns:
                grouped[s] = 0
        grouped["total"] = grouped.sum(axis=1)
    else:
        grouped = filtered_kw.groupby("keyword").size().to_frame(name="total")
        grouped["ieee"] = 0
        grouped["elsevier"] = 0

    fig = topn_hbar(
        grouped["total"],
        x_title="Quantidade de artigos",
        y_title="Palavra-chave",
        title="Top 20 palavras-chave",
    )
    for trace in fig.data:
        breakdown = grouped.loc[list(trace.y), ["ieee", "elsevier"]].to_numpy()
        trace.customdata = breakdown
        trace.hovertemplate = (
            "<b>%{y}</b><br>"
            "Total de artigos: %{x:,}<br>"
            "• Artigos IEEE: %{customdata[0]:,}<br>"
            "• Artigos Elsevier: %{customdata[1]:,}<extra></extra>"
        )
    render_chart(
        fig,
        caption="Tópicos mais recorrentes no corpus. Passe o mouse sobre as barras para conferir a divisão "
        "exata entre IEEE e Elsevier.",
    )


def _keyword_stats(kw_lists: pd.Series, all_keywords: list[str]) -> None:
    st.subheader("Estatísticas do Vocabulário")
    kw_counts = kw_lists.apply(len)
    metric_row(
        [
            ("🏷️ Palavras-chave únicas", f"{len(all_keywords):,}", None),
            ("📊 Média de termos por artigo", f"{kw_counts.mean():.1f}", None),
            ("🚫 Artigos sem palavras-chave", f"{(kw_counts == 0).mean():.0%}", None),
        ]
    )


def _zipf_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("📖 Lei de Zipf do Vocabulário Técnico")
    st.caption(
        "A Lei de Zipf estabelece que a frequência de uma palavra é inversamente proporcional ao seu posto "
        "($f \\propto 1/r^\\gamma$). Em acervos bibliométricos consolidados, o coeficiente de inclinação "
        "aproxima-se de $\\gamma \\approx 1.0$, indicando um vocabulário linguístico equilibrado entre termos centrais "
        "e cauda longa de especialização."
    )

    zipf_res = zipf_law_analysis(articles_df)
    if not zipf_res["valid"]:
        st.info("Texto insuficiente para avaliar a Lei de Zipf.")
        return

    metric_row(
        [
            ("📐 Coeficiente de Inclinação (γ)", f"{zipf_res['gamma']:.2f}", "Alvo teórico: ~1.0"),
            (
                "🎯 Coef. Determinação (R²)",
                f"{zipf_res['r_squared']:.3f}",
                "Qualidade do ajuste log-log",
            ),
            ("📚 Vocabulário Único", f"{zipf_res['vocab_size']:,} termos", None),
            ("📝 Total de Ocorrências", f"{zipf_res['total_tokens']:,} palavras", None),
        ]
    )

    plot_df = pd.DataFrame(
        {
            "posto": zipf_res["ranks"],
            "frequencia": zipf_res["frequencies"],
            "ajuste": zipf_res["expected_zipf"],
            "termo": zipf_res["words"],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["posto"],
            y=plot_df["frequencia"],
            mode="markers",
            name="Frequência Real",
            text=plot_df["termo"],
            marker=dict(size=6, color="#2a78d6", opacity=0.7),
            hovertemplate="<b>%{text}</b><br>Posto: %{x}<br>Frequência: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["posto"],
            y=plot_df["ajuste"],
            mode="lines",
            name=f"Regressão Zipf (γ = {zipf_res['gamma']:.2f})",
            line=dict(color="#eb6834", width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis=dict(type="log", title="Posto do Termo (log r)"),
        yaxis=dict(type="log", title="Frequência de Ocorrência (log f)"),
        title="Distribuição Posto-Frequência do Vocabulário (Escala Log-Log)",
        height=480,
    )
    render_chart(
        fig,
        caption="A proximidade dos pontos em relação à reta tracejada confirma a aderência do corpus à Lei de Zipf.",
    )

    st.markdown("##### 📋 Comparativo dos Top 30 Termos")
    st.dataframe(zipf_res["top_words_df"], hide_index=True, width="stretch")


def _dynamic_ctfidf_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🧬 Vocabulário Dinâmico por Tema e Época (c-TF-IDF)")
    st.caption(
        "O algoritmo c-TF-IDF (Class-based TF-IDF) extrai os termos que mais diferenciam cada tema temático "
        "dos demais em cada uma das três épocas cronológicas. Revela a evolução dos tópicos tecnológicos "
        "na literatura de planejamento de distribuição."
    )

    signals = loaders.semantics()
    if signals.empty or "theme_label" not in signals.columns:
        st.info("A camada semântica com atribuição de temas é necessária para esta análise.")
        return

    merged = pd.merge(articles_df, signals[["doi", "theme_label"]], on="doi", how="inner")
    ctfidf_res = dynamic_topic_ctfidf(merged)

    if not ctfidf_res["valid"]:
        st.info("Dados temporais e temáticos insuficientes para segmentar o c-TF-IDF dinâmico.")
        return

    st.dataframe(ctfidf_res["summary_df"], hide_index=True, width="stretch")


def _keyword_explorer(
    articles_df: pd.DataFrame, kw_lists: pd.Series, all_keywords: list[str]
) -> None:
    st.subheader("🔎 Explorador: filtrar artigos por palavra-chave")
    selected = st.multiselect(
        "Selecione um ou mais termos (serão exibidos artigos que contenham qualquer um deles):",
        options=all_keywords,
    )
    if not selected:
        st.caption("Selecione palavras-chave acima para filtrar os artigos correspondentes.")
        return

    selected_set = set(selected)
    mask = kw_lists.apply(lambda kws: bool(set(kws) & selected_set))
    filtered = articles_df.loc[mask]
    st.caption(f"{len(filtered):,} artigos encontrados")

    if "citation_count" in filtered.columns:
        filtered = filtered.sort_values("citation_count", ascending=False, na_position="last")
    article_table(
        filtered,
        ["title", "year", "venue", "source", "citation_count", "reference_count", "doi"],
        download_key="artigos_por_palavra_chave",
    )


def _prepare_keyword_trend_data(articles_df: pd.DataFrame) -> pd.DataFrame | None:
    """Shared prep for the three "Evolução Temporal" sub-tabs, or None if there's not enough data."""
    kw_year = explode_keywords(articles_df)
    if kw_year.empty or "year" not in kw_year.columns:
        return None
    kw_year["year"] = valid_years(kw_year, lo=TREND_MIN_YEAR, hi=2026)
    kw_year = kw_year.dropna(subset=["year"]).astype({"year": int})
    if len(kw_year) < 30:
        return None
    return kw_year


def _topic_share_area(kw_year: pd.DataFrame) -> None:
    st.markdown("**Participação anual das principais palavras-chave**")
    top_terms = kw_year["keyword"].value_counts().head(TOP_KEYWORDS_TREND).index.tolist()
    scoped = kw_year[kw_year["keyword"].isin(top_terms)]
    by_year_kw = scoped.groupby(["year", "keyword"]).size().reset_index(name="count")

    color_map = {
        kw: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, kw in enumerate(top_terms)
    }
    fig = stacked_area(
        by_year_kw,
        x="year",
        y="count",
        color="keyword",
        color_map=color_map,
        title=f"Participação (%) das top {TOP_KEYWORDS_TREND} palavras-chave por ano",
        groupnorm="percent",
    )
    fig.update_layout(
        xaxis_title=f"Ano de publicação (desde {TREND_MIN_YEAR})",
        yaxis_title="Participação entre as menções do ano (%)",
        legend_title_text="Palavra-chave",
    )
    render_chart(
        fig,
        caption=f"Área 100% empilhada: mostra como o peso relativo de cada termo mudou ano a ano, entre as "
        f"top {TOP_KEYWORDS_TREND} palavras-chave do corpus.",
    )


def _rising_falling(kw_year: pd.DataFrame) -> None:
    st.markdown("**Termos em ascensão vs. em declínio**")
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= MIN_KEYWORD_OCCURRENCES].index

    by_year_total = kw_year.groupby("year").size()
    slopes = {}
    for kw in eligible:
        yearly = kw_year[kw_year["keyword"] == kw].groupby("year").size()
        share = (yearly / by_year_total.reindex(yearly.index)).fillna(0.0) * 100
        if len(share) < 3:
            continue
        x = share.index.to_numpy(dtype=float)
        y = share.to_numpy(dtype=float)
        slope = np.polyfit(x - x.mean(), y, 1)[0]
        slopes[kw] = slope

    if not slopes:
        st.info(
            f"Nenhum termo com pelo menos {MIN_KEYWORD_OCCURRENCES} ocorrências para calcular tendência."
        )
        return

    slope_series = pd.Series(slopes).sort_values()
    top_slopes = pd.concat([slope_series.head(7), slope_series.tail(7)]).drop_duplicates()
    df = top_slopes.rename_axis("keyword").reset_index(name="slope")
    df["direction"] = df["slope"].apply(lambda s: "Em Alta" if s >= 0 else "Em Queda")
    df = df.sort_values("slope")

    fig = px.bar(
        df,
        x="slope",
        y="keyword",
        orientation="h",
        color="direction",
        color_discrete_map={"Em Alta": TREND_UP_COLOR, "Em Queda": TREND_DOWN_COLOR},
        title="Inclinação da participação anual (regressão linear)",
        labels={
            "slope": "Variação anual na participação (p.p./ano)",
            "keyword": "Palavra-chave",
            "direction": "Tendência",
        },
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:+.2f} p.p./ano<extra></extra>")
    fig.update_layout(legend_title_text="Tendência")
    render_chart(
        fig,
        caption=f"Inclinação da participação percentual anual de cada termo (mínimo {MIN_KEYWORD_OCCURRENCES} "
        "ocorrências no período), estimada por regressão linear simples.",
    )

    from lake_research_map.dashboard.analytics import mann_kendall_trend

    pivoted = kw_year.groupby(["year", "keyword"]).size().unstack(fill_value=0)
    mk_records = []
    for kw in df["keyword"]:
        if kw in pivoted.columns:
            series = pivoted[kw].to_numpy()
            res = mann_kendall_trend(series)
            mk_records.append(
                {
                    "Palavra-chave": kw,
                    "Tendência (Mann-Kendall)": res["trend"].title(),
                    "P-valor": round(res["p_value"], 4),
                    "Significativo (p < 0.05)": "✅ Sim" if res["p_value"] < 0.05 else "Não",
                    "Inclinação de Sen": round(res["slope"], 3),
                }
            )
    if mk_records:
        st.markdown("##### 🔬 Teste Não-Paramétrico de Tendência (Mann-Kendall & Sen)")
        st.caption(
            "O teste de Mann-Kendall verifica se a evolução temporal monótona é estatisticamente "
            "significativa (p < 0,05) e livre de suposições de normalidade residual."
        )
        st.dataframe(pd.DataFrame(mk_records), hide_index=True, width="stretch")


def _first_appearance(kw_year: pd.DataFrame) -> None:
    st.markdown("**Vocabulário novo vs. fundacional**")
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= MIN_KEYWORD_OCCURRENCES]
    if eligible.empty:
        st.info(f"Nenhum termo com pelo menos {MIN_KEYWORD_OCCURRENCES} ocorrências.")
        return

    first_year = kw_year[kw_year["keyword"].isin(eligible.index)].groupby("keyword")["year"].min()
    df = pd.DataFrame({"first_year": first_year, "count": eligible}).reset_index(names="keyword")

    # Many terms share the same first_year (61 eligible terms here, ~39 of them
    # clustered in just 2010-2011) -- always-on text labels for every point pile
    # up and become unreadable. Identification moves to hover; only a handful
    # of standout points (highest volume + most recent debut) get a permanent
    # label, since those are the ones actually worth calling out visually and
    # are few enough not to collide.
    fig = px.scatter(
        df,
        x="first_year",
        y="count",
        hover_name="keyword",
        color="first_year",
        color_continuous_scale=["#4a3aa7", "#eda100", "#1baf7a"],
        labels={
            "first_year": "Ano da primeira menção no corpus",
            "count": "Menções totais no período",
        },
    )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.4)")),
        hovertemplate="<b>%{hovertext}</b><br>1ª menção: %{x}<br>%{y:,} menções totais<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False)

    highlight = pd.concat([df.nlargest(4, "count"), df.nlargest(4, "first_year")]).drop_duplicates(
        subset="keyword"
    )
    for idx, (_, row) in enumerate(highlight.iterrows()):
        fig.add_annotation(
            x=row["first_year"],
            y=row["count"],
            text=row["keyword"],
            showarrow=True,
            arrowhead=0,
            arrowcolor="rgba(148,163,184,0.5)",
            ax=0,
            ay=-22 if idx % 2 == 0 else 22,
            font=dict(size=10),
        )

    render_chart(
        fig,
        caption="Termos no canto superior direito são vocabulário recente que já ganhou volume; termos à "
        "esquerda com contagem alta são vocabulário fundacional da área. Passe o mouse sobre qualquer ponto "
        "para ver o termo; os rótulos fixos destacam apenas os pontos mais extremos (maior volume e estreia "
        "mais recente), para não sobrepor os demais.",
    )


def _qualis_match_data(
    articles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Shared computation for all "Periódicos" CAPES/Qualis sub-tabs.

    Returns `(match_df, with_estrato, totals_by_estrato, estrato_order)`; `with_estrato` is
    `articles_df` with an added `estrato` column (each article's venue's CAPES/Qualis tier).
    """
    # Filter the full-corpus match table down to venues in the current filter
    # scope, rather than calling `venue_qualis_map` with a filtered venues
    # tuple directly -- that would re-trigger the rapidfuzz matching pass
    # (and, transitively, the ~9s CAPES xlsx parse) on every global filter
    # change, since the venues tuple is the cache key.
    venues = set(articles_df["venue"].dropna().unique())
    match_df = loaders.all_venue_qualis_map()
    match_df = match_df[match_df["venue"].isin(venues)].reset_index(drop=True)
    match_df["estrato"] = match_df["estrato"].fillna(NOT_CLASSIFIED)
    # A raw None in a string column renders as literal "undefined" in Streamlit's
    # dataframe grid (pandas' Arrow-backed string dtype stores it as a genuine
    # null, not NaN) -- always give it an explicit placeholder instead.
    match_df["matched_title"] = match_df["matched_title"].fillna("—")
    match_df["score"] = match_df["score"].apply(lambda s: f"{s:.0f}" if pd.notna(s) else "—")

    counts = articles_df["venue"].value_counts()
    match_df["articles"] = match_df["venue"].map(counts).fillna(0).astype(int)
    # Best classification first (A1 ... C, unclassified last); most-published
    # journal leads within a tier.
    tier_rank = {estrato: rank for rank, estrato in enumerate(ESTRATO_ORDER)}
    match_df["_tier_rank"] = match_df["estrato"].map(tier_rank)
    match_df = match_df.sort_values(["_tier_rank", "articles"], ascending=[True, False]).drop(
        columns=["_tier_rank"]
    )

    venue_to_estrato = dict(zip(match_df["venue"], match_df["estrato"], strict=True))
    with_estrato = articles_df.copy()
    with_estrato["estrato"] = with_estrato["venue"].map(venue_to_estrato)
    totals_by_estrato = with_estrato["estrato"].value_counts().reindex(ESTRATO_ORDER, fill_value=0)
    totals_by_estrato = totals_by_estrato[
        totals_by_estrato.index.isin(with_estrato["estrato"].unique())
    ]
    estrato_order = list(totals_by_estrato.index)

    return match_df, with_estrato, totals_by_estrato, estrato_order


def _qualis_table(
    articles_df: pd.DataFrame, match_df: pd.DataFrame, a1_articles_df: pd.DataFrame
) -> None:
    st.subheader("🎓 Classificação CAPES/Qualis")
    n_classified = int((match_df["estrato"] != NOT_CLASSIFIED).sum())
    a1_venues = match_df.loc[match_df["estrato"] == "A1", "venue"]
    pct_a1 = (len(a1_articles_df) / len(articles_df) * 100) if len(articles_df) else 0.0

    metric_row(
        [
            ("📚 Periódicos classificados", f"{n_classified}/{len(match_df)}", None),
            ("🥇 Periódicos A1", f"{len(a1_venues)}", None),
            ("📄 Artigos em periódicos A1", f"{len(a1_articles_df):,}", f"{pct_a1:.1f}% do corpus"),
        ]
    )

    st.caption(
        f"Classificação oficial CAPES/Qualis (quadriênio 2017-2020, área **{QUALIS_AREA}** -- a "
        "última avaliação por periódico; a partir de 2025-2028 a CAPES passa a avaliar por artigo, não "
        "mais por veículo). Cada periódico do corpus é casado com o título de referência por "
        f"similaridade textual (corte de {MATCH_THRESHOLD:.0f}%), já que grafias variam entre bases "
        "(ex.: `&` vs. `and`, sufixos `(Print)`/`(Online)`). Um periódico não encontrado com confiança "
        f'suficiente aparece como "{NOT_CLASSIFIED}" -- nunca como uma nota adivinhada.'
    )
    st.dataframe(
        match_df.rename(
            columns={
                "venue": "Periódico (corpus)",
                "matched_title": "Título casado (CAPES)",
                "estrato": "Estrato",
                "score": "Similaridade (%)",
                "articles": "Artigos",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _qualis_totals_chart(totals_by_estrato: pd.Series, estrato_order: list[str]) -> None:
    st.subheader("Total de publicações por classificação")
    fig = px.bar(
        x=totals_by_estrato.index,
        y=totals_by_estrato.to_numpy(),
        category_orders={"x": estrato_order},
        color=totals_by_estrato.index,
        color_discrete_map=venue_color_map(estrato_order, others_label=NOT_CLASSIFIED),
        labels={"x": "Classificação", "y": "Artigos"},
    )
    fig.update_layout(showlegend=False)
    render_chart(
        fig,
        caption="Total de artigos do corpus por classificação CAPES/Qualis, ordenado da melhor "
        f'("A1") para a pior, com "{NOT_CLASSIFIED}" ao final.',
    )


def _qualis_cumulative_chart(with_estrato: pd.DataFrame, estrato_order: list[str]) -> None:
    st.subheader("Acumulado por classificação")
    cum_by_estrato = cumulative_by_category(with_estrato, "estrato", top_n=10)
    if cum_by_estrato.empty:
        st.info("Sem anos válidos para o acumulado por classificação.")
        return

    fig = stacked_area(
        cum_by_estrato,
        x="year",
        y="cumulative",
        color="estrato",
        color_map=venue_color_map(estrato_order, others_label=NOT_CLASSIFIED),
        category_orders={"estrato": estrato_order},
        title="Artigos acumulados por classificação CAPES/Qualis",
    )
    fig.update_traces(
        hovertemplate="Ano %{x}<br>%{data.name}: %{y:,.0f} artigos acumulados<extra></extra>"
    )
    fig.update_layout(xaxis_title="Ano de publicação", yaxis_title="Artigos acumulados")
    render_chart(
        fig,
        caption="Composição acumulada do corpus por estrato CAPES/Qualis (área "
        f"{QUALIS_AREA}). Periódicos sem classificação confiável ficam em "
        f'"{NOT_CLASSIFIED}", não misturados a nenhum estrato real.',
    )


def _qualis_a1_a3_combined(with_estrato: pd.DataFrame) -> None:
    st.subheader("Periódicos A1-A3 — Acumulado")
    combined_df = with_estrato[with_estrato["estrato"].isin(("A1", "A2", "A3"))]
    if combined_df.empty:
        st.info("Nenhum artigo em periódico classificado A1, A2 ou A3 nesta camada/filtro.")
        return

    tier_palette = venue_color_map(["A1", "A2", "A3"])

    col_volume, col_ranking = st.columns(2)
    with col_volume:
        years_df = combined_df.copy()
        years_df["year"] = valid_years(years_df)
        years_df = years_df.dropna(subset=["year"]).astype({"year": int})
        by_year = source_counts_by(years_df, "year").sort_values("year")
        if by_year.empty:
            st.info("Sem anos válidos para este gráfico.")
        else:
            fig = source_bars(
                by_year,
                "year",
                total_line=True,
                title="Volume Anual de Artigos em Periódicos A1-A3, por Base",
            )
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Ano de publicação",
                yaxis_title="Quantidade de artigos",
            )
            render_chart(
                fig,
                caption="Volume anual de artigos publicados em periódicos A1, A2 ou A3, por base.",
            )
    with col_ranking:
        top_combined = (
            source_counts_by(combined_df, "venue").sort_values("total", ascending=False).head(15)
        )
        fig = source_topn_hbar(
            top_combined,
            "venue",
            title="Top 15 Periódicos A1-A3 por Quantidade de Artigos, por Base",
            x_title="Quantidade de artigos",
            y_title="Periódico",
        )
        render_chart(
            fig,
            caption="Periódicos mais publicados pelo corpus, somando os estratos A1, A2 e A3, "
            "coloridos por base.",
        )

    st.divider()
    col_venue_tier, col_source_tier = st.columns(2)
    with col_venue_tier:
        top_venues = combined_df["venue"].value_counts().head(15)
        venue_to_estrato = combined_df.drop_duplicates("venue").set_index("venue")["estrato"]
        fig = topn_hbar(
            top_venues,
            color_by=venue_to_estrato,
            palette=tier_palette,
            title="Top 15 Periódicos A1-A3 por Quantidade de Artigos, por Classificação CAPES/Qualis",
            x_title="Quantidade de artigos",
            y_title="Periódico",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
        render_chart(
            fig,
            caption="Os mesmos periódicos mais publicados do conjunto A1-A3, agora coloridos pela "
            "própria classificação CAPES/Qualis — mostra qual categoria cada periódico do ranking "
            "pertence.",
        )
    with col_source_tier:
        if "source" not in combined_df.columns:
            st.info("Coluna 'source' não disponível nesta camada.")
        else:
            by_source_tier = (
                combined_df.groupby(["source", "estrato"]).size().reset_index(name="count")
            )
            fig = px.bar(
                by_source_tier,
                x="source",
                y="count",
                color="estrato",
                category_orders={"estrato": ["A1", "A2", "A3"], "source": ["ieee", "elsevier"]},
                color_discrete_map=tier_palette,
                title="Distribuição de Artigos A1/A2/A3 por Base (IEEE vs. Elsevier)",
                labels={"source": "Base", "count": "Artigos", "estrato": "Classificação"},
            )
            fig.update_traces(
                hovertemplate="<b>%{data.name}</b><br>%{x}: %{y:,} artigos<extra></extra>"
            )
            fig.update_layout(
                xaxis_title="Base",
                yaxis_title="Quantidade de artigos",
                legend_title_text="Classificação",
            )
            render_chart(
                fig,
                caption="Distribuição de artigos A1/A2/A3 por base (IEEE vs. Elsevier).",
            )


def _bradford_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🎯 Zonas de Dispersão de Bradford")
    st.caption(
        "A Lei de Bradford divide os periódicos em 3 zonas concêntricas de produtividade igual (~1/3 dos artigos cada). "
        "A proporção teórica do número de periódicos em cada zona segue aproximadamente 1 : k : k²."
    )
    from lake_research_map.dashboard.analytics import bradford_zones

    res = bradford_zones(articles_df)
    zone_sum = res.get("zone_summary")
    if zone_sum is None or zone_sum.empty:
        st.info("Dados insuficientes para análise de Bradford.")
        return

    metric_row(
        [
            (
                "🎯 Periódicos no Núcleo (Zona 1)",
                f"{int(zone_sum.iloc[0]['venues'])}",
                f"{int(zone_sum.iloc[0]['articles']):,} artigos",
            ),
            (
                "📚 Periódicos na Zona 2",
                f"{int(zone_sum.iloc[1]['venues'])}",
                f"{int(zone_sum.iloc[1]['articles']):,} artigos",
            ),
            (
                "🌐 Periódicos na Zona 3",
                f"{int(zone_sum.iloc[2]['venues'])}",
                f"{int(zone_sum.iloc[2]['articles']):,} artigos",
            ),
            (
                "📐 Multiplicador de Bradford (k)",
                f"{res.get('multiplier_mean', 1.0):.2f}",
                "Taxa geométrica de dispersão",
            ),
        ]
    )

    fig = px.bar(
        zone_sum,
        x="zone",
        y="venues",
        text="venues",
        title="Quantidade de periódicos necessários para produzir 1/3 do corpus em cada zona",
        labels={
            "zone": "Zona de Bradford (1 = Núcleo, 3 = Periferia)",
            "venues": "Quantidade de periódicos",
        },
        color="zone",
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    render_chart(
        fig,
        caption="A Zona 1 (núcleo) concentra os periódicos de referência obrigatória para planejamento de "
        "sistemas de distribuição de energia. As zonas 2 e 3 revelam a dispersão ampla em veículos generalistas.",
    )


def _semantic_venues_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🗺️ Perfil Semântico dos Principais Periódicos")
    st.caption(
        "Posicionamento dos periódicos no espaço vetorial a partir do centróide dos artigos "
        "que publicam. Permite visualizar a sobreposição temática de veículos independentemente "
        "de sua editora comercial (IEEE vs. Elsevier)."
    )
    signals = loaders.semantics()
    if signals.empty or "map_x" not in signals.columns or "map_y" not in signals.columns:
        st.info("Sinais semânticos não disponíveis. Execute `--stage semantic`.")
        return

    scoped = loaders.with_semantics(articles_df)
    valid = scoped.dropna(subset=["map_x", "map_y", "venue"]).copy()
    if len(valid) < 10:
        st.info("Artigos insuficientes com dados semânticos e periódicos associados.")
        return

    venue_counts = valid["venue"].value_counts()
    eligible_venues = venue_counts[venue_counts >= 3].index

    venue_stats = (
        valid[valid["venue"].isin(eligible_venues)]
        .groupby("venue")
        .agg(
            map_x=("map_x", "mean"),
            map_y=("map_y", "mean"),
            articles=("venue", "count"),
            margin=("relevance_margin", "mean"),
            source=("source", lambda s: s.mode().iat[0] if len(s) else "ieee"),
        )
        .reset_index()
    )

    if venue_stats.empty:
        st.info("Nenhum periódico com volume suficiente para cálculo de centróide estável.")
        return

    fig = px.scatter(
        venue_stats,
        x="map_x",
        y="map_y",
        size="articles",
        color="source",
        hover_name="venue",
        hover_data={
            "articles": True,
            "margin": ":.3f",
            "map_x": False,
            "map_y": False,
        },
        title="Centróides Semânticos dos Periódicos (mínimo 3 artigos no corpus)",
        labels={
            "source": "Base Predominante",
            "articles": "Artigos",
            "margin": "Margem Média",
        },
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(
        xaxis=dict(title="Dimensão 1 (t-SNE)", showticklabels=False),
        yaxis=dict(title="Dimensão 2 (t-SNE)", showticklabels=False),
    )
    render_chart(
        fig,
        caption="Periódicos próximos entre si publicam artigos com vocabulário e temáticas altamente convergentes.",
    )


def _conceptual_atypicality_tab(articles_df: pd.DataFrame) -> None:
    st.subheader("Combinações incomuns de palavras-chave")
    st.caption(
        "Diagnóstico exploratório de coocorrências em relação à independência marginal das palavras-chave. "
        "Ele descreve combinações raras neste corpus e sua associação com citações; não reproduz o modelo "
        "nulo por pares de periódicos de Uzzi et al."
    )
    res = conceptual_atypicality_analysis(articles_df)
    if not res.get("valid"):
        st.info("Palavras-chave insuficientes para modelagem nula de coocorrência.")
        return

    metric_row(
        [
            (
                "Taxa top 5% no grupo incomum",
                f"{res['hit_rate_high_atypical']:.1f}%",
                f"Baseline: {res['hit_rate_baseline']:.1f}%",
            ),
            (
                "⭐ Limiar Top 5% Citações",
                f"≥ {res['cite_p95_threshold']} citações",
                "Artigos no percentil 95",
            ),
            (
                "Critério descritivo",
                "Coocorrência",
                "Esperado por independência",
            ),
            (
                "📊 Pares Raros Mapeados",
                f"{len(res['atypical_pairs'])} combinações",
                "Z-score < 0 vs modelo nulo",
            ),
        ]
    )

    art_df = res["articles_df"]
    fig = px.scatter(
        art_df,
        x="median_z",
        y="min_z",
        color="is_hit",
        hover_data=["title", "citations", "year"],
        color_discrete_map={True: "#e34948", False: "#2a78d6"},
        labels={
            "median_z": "Convencionalidade (Mediana de Z)",
            "min_z": "Atipicidade (Z Mínimo)",
            "is_hit": "Top 5% Citações?",
        },
        title="Dispersão: Convencionalidade vs. Atipicidade Extrema por Artigo",
    )
    fig.update_layout(
        xaxis_title="Convencionalidade (Mediana Z)", yaxis_title="Atipicidade (Z Mínimo)"
    )
    render_chart(
        fig,
        caption="Artigos no quadrante inferior direito (alta convencionalidade e Z mínimo negativo) "
        "incorporam combinações inovadoras sobre terreno teórico maduro.",
    )

    st.markdown("##### 🔍 Principais Pares Conceituais Atípicos Identificados no Corpus")
    st.dataframe(res["atypical_pairs"], hide_index=True, width="stretch")


def _structural_breaks_tab(articles_df: pd.DataFrame) -> None:
    st.subheader("⚡ Detecção de Quebras Estruturais e Pontos de Inflexão (Changepoints)")
    st.caption(
        "Aplica o teste de Chow e minimização da soma dos quadrados dos resíduos para detectar o momento histórico "
        "em que a taxa média de publicação sofreu uma transição de regime estatisticamente significante (p < 0.05)."
    )
    if "year" not in articles_df.columns:
        st.info("Ano de publicação indisponível.")
        return

    counts = articles_df["year"].dropna().value_counts().sort_index()
    valid_counts = counts[(counts.index >= 2000) & (counts.index <= 2025)]
    if len(valid_counts) < 6:
        st.info("Série temporal insuficiente para detecção de quebras.")
        return

    res = detect_structural_breaks(valid_counts)
    if not res.get("has_break"):
        st.info("Nenhuma quebra estrutural abrupta identificada na série analisada.")
        return

    metric_row(
        [
            ("📅 Ano de Quebra / Inflexão", f"{res['break_year']}", "Transição de regime"),
            (
                "📈 Salto Relativo de Produção",
                f"+{res['relative_jump_pct']:.1f}%",
                f"{res['pre_mean']} → {res['post_mean']} artigos/ano",
            ),
            ("🧪 Estatística F de Chow", f"{res['f_stat']:.2f}", f"p-valor: {res['p_value']:.4f}"),
            ("⚖️ Significância", "Estatisticamente Significante", "p < 0.05"),
        ]
    )

    break_yr = res["break_year"]
    years = valid_counts.index.to_numpy()
    vals = valid_counts.to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=vals, name="Publicações / Ano", marker_color="#2a78d6"))
    # Pre-break mean
    fig.add_trace(
        go.Scatter(
            x=[years[0], break_yr],
            y=[res["pre_mean"], res["pre_mean"]],
            mode="lines",
            name=f"Regime 1 ({res['pre_mean']:.1f}/ano)",
            line=dict(color="#f39c12", width=3, dash="dash"),
        )
    )
    # Post-break mean
    fig.add_trace(
        go.Scatter(
            x=[break_yr, years[-1]],
            y=[res["post_mean"], res["post_mean"]],
            mode="lines",
            name=f"Regime 2 ({res['post_mean']:.1f}/ano)",
            line=dict(color="#27ae60", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Inflexão Estrutural na Série Temporal de Publicações (Ano de Quebra: {break_yr})",
        xaxis_title="Ano",
        yaxis_title="Artigos publicados",
    )
    render_chart(
        fig,
        caption=f"A transição de regime no ano de {break_yr} reflete a aceleração da produção científica na área.",
    )
