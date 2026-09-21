"""🏆 Destaques e Impacto — citações, referências, colaboração e periódicos mais influentes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    source_counts_by,
    source_means,
    valid_years,
)
from lake_research_map.dashboard.charts import source_lines
from lake_research_map.dashboard.components import (
    article_table,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from lake_research_map.dashboard.theme import CATEGORICAL_PALETTE, SOURCE_COLORS

MIN_CITED_ARTICLES = 3


def render() -> None:
    page_header(
        "🏆",
        "Impacto e citações",
        "Análise bibliométrica: embasamento em referências, dinâmica de citações, distribuições de cauda pesada e determinantes GLM.",
    )

    articles_df = loaders.require_articles()

    tab_refs, tab_citations = st.tabs(
        ["📚 Fundamentação Teórica (Referências)", "⭐ Dinâmica de Citações & Econometria"]
    )

    with tab_refs:
        ref_df = _reference_distribution_intro(articles_df)
        sub_dist, sub_vs_cit, sub_top = st.tabs(
            [
                "📊 Distribuição de Referências",
                "🔗 Refs vs. Citações",
                "📖 Mais Referenciados",
            ]
        )
        with sub_dist:
            if ref_df is not None:
                view_mode = (
                    st.segmented_control(
                        "Formato de visualização da distribuição",
                        options=["Histograma", "Curva Cumulativa (ECDF)", "Box Plot"],
                        default="Histograma",
                        key="ref_dist_mode",
                    )
                    or "Histograma"
                )
                if view_mode == "Histograma":
                    _reference_histogram(ref_df)
                elif view_mode == "Curva Cumulativa (ECDF)":
                    _reference_ecdf(ref_df)
                else:
                    _reference_box(ref_df)
        with sub_vs_cit:
            _references_vs_citations(articles_df)
        with sub_top:
            _top_referenced(articles_df)

    with tab_citations:
        sub_cited, sub_by_year, sub_heavytail, sub_agenorm, sub_glm = st.tabs(
            [
                "🏆 Mais Citados",
                "📅 Citados por Ano",
                "📐 Cauda Pesada (Power-Law)",
                "⏳ Normalizado por Idade",
                "🔬 Determinantes GLM",
            ]
        )
        with sub_cited:
            _top_cited(articles_df)
        with sub_by_year:
            _cited_by_year(articles_df)
        with sub_heavytail:
            _heavy_tail_analysis(articles_df)
        with sub_agenorm:
            _age_normalized_rankings(articles_df)
        with sub_glm:
            _citation_determinants_glm_view(articles_df)


def _reference_distribution_intro(articles_df: pd.DataFrame) -> pd.DataFrame | None:
    """Guard + shared metric row for the Histograma/ECDF/Box Plot sub-tabs.

    Returns the cleaned `reference_count` frame, or None if there's nothing to show.
    """
    st.subheader("📚 Quantidade de referências usadas por artigo (IEEE vs. Elsevier)")
    if not require_columns(articles_df, ["reference_count"]) or not (
        articles_df["reference_count"].notna().any()
    ):
        st.info("Nenhum artigo com contagem de referências disponível nesta camada.")
        return None

    ref_df = articles_df.dropna(subset=["reference_count"]).copy()
    ref_df["reference_count"] = ref_df["reference_count"].astype(int)
    means = source_means(ref_df, "reference_count")
    ieee_med = (
        ref_df.loc[ref_df.get("source") == "ieee", "reference_count"].median()
        if "source" in ref_df
        else None
    )
    els_med = (
        ref_df.loc[ref_df.get("source") == "elsevier", "reference_count"].median()
        if "source" in ref_df
        else None
    )
    tot_med = float(ref_df["reference_count"].median())
    max_refs = int(ref_df["reference_count"].max())

    metric_row(
        [
            (
                "📘 Média de Refs (IEEE)",
                f"{means['ieee']:.1f}" if means["ieee"] is not None else "N/D",
                f"Mediana: {ieee_med:.0f}"
                if ieee_med == ieee_med and ieee_med is not None
                else None,
            ),
            (
                "📙 Média de Refs (Elsevier)",
                f"{means['elsevier']:.1f}" if means["elsevier"] is not None else "N/D",
                f"Mediana: {els_med:.0f}" if els_med == els_med and els_med is not None else None,
            ),
            ("📊 Média de Refs (Total)", f"{means['total']:.1f}", f"Mediana: {tot_med:.0f}"),
            ("🔝 Maior Bibliografia", f"{max_refs:,} refs", f"{len(ref_df):,} artigos analisados"),
        ]
    )
    st.caption(
        "Distribuição do número de referências citadas por artigo — como histograma, curva cumulativa "
        "(ECDF) e box plot, nas abas abaixo. Os dados do IEEE têm origem direta no export CSV do IEEE "
        "Xplore (`Reference Count`). Os dados da Elsevier foram enriquecidos via cache offline "
        "(`data/enrichment_cache.json`); ver os cartões acima para as médias por base."
    )
    return ref_df


def _reference_histogram(ref_df: pd.DataFrame) -> None:
    # In overlay mode, the last category painted sits on top -- draw the
    # smaller-volume source last so its bars aren't hidden behind the
    # larger one wherever their bins overlap.
    source_order = (
        ref_df["source"].value_counts().sort_values(ascending=False).index.tolist()
        if "source" in ref_df.columns
        else None
    )
    fig_hist = px.histogram(
        ref_df,
        x="reference_count",
        color="source" if "source" in ref_df.columns else None,
        barmode="overlay",
        opacity=0.75,
        nbins=40,
        color_discrete_map=SOURCE_COLORS,
        category_orders={"source": source_order} if source_order else None,
        labels={"reference_count": "Referências citadas por artigo", "source": "Base"},
    )
    fig_hist.update_traces(
        hovertemplate="Intervalo: %{x} refs<br>Quantidade: %{y:,} artigos (%{data.name})<extra></extra>"
    )
    fig_hist.update_layout(
        xaxis_title="Referências citadas por artigo (tamanho da bibliografia)",
        yaxis_title="Quantidade de artigos",
        hovermode="x unified",
    )
    render_chart(fig_hist)


def _reference_ecdf(ref_df: pd.DataFrame) -> None:
    fig_cdf = px.ecdf(
        ref_df,
        x="reference_count",
        color="source" if "source" in ref_df.columns else None,
        color_discrete_map=SOURCE_COLORS,
        ecdfnorm="percent",
        labels={"reference_count": "Referências citadas por artigo", "source": "Base"},
    )
    fig_cdf.update_traces(
        hovertemplate="Até %{x} refs: %{y:.1f}% dos artigos (%{data.name})<extra></extra>"
    )
    fig_cdf.update_layout(
        xaxis_title="Referências citadas por artigo (tamanho da bibliografia)",
        yaxis_title="% acumulado de artigos",
        hovermode="x unified",
    )
    render_chart(
        fig_cdf,
        caption="Cada ponto (x, y) lê-se: 'y% dos artigos têm até x referências citadas'.",
    )


def _reference_box(ref_df: pd.DataFrame) -> None:
    fig_box = px.box(
        ref_df,
        x="source" if "source" in ref_df.columns else None,
        y="reference_count",
        color="source" if "source" in ref_df.columns else None,
        color_discrete_map=SOURCE_COLORS,
        points="outliers",
        labels={"reference_count": "Referências citadas por artigo", "source": "Base"},
    )
    fig_box.update_layout(
        yaxis_title="Referências citadas por artigo",
        xaxis_title="Base",
        showlegend=False,
    )
    render_chart(fig_box)


def _references_vs_citations(articles_df: pd.DataFrame) -> None:
    st.subheader("🔗 Referências citadas vs. Citações recebidas")
    if (
        not require_columns(articles_df, ["reference_count", "citation_count"])
        or not articles_df["reference_count"].notna().any()
        or not articles_df["citation_count"].notna().any()
    ):
        return

    scatter_df = articles_df.dropna(subset=["reference_count", "citation_count"]).copy()
    scatter_df = scatter_df[scatter_df["reference_count"] > 0]
    if len(scatter_df) < 5:
        st.info("Poucos artigos com dados de referências e citações disponíveis para correlação.")
        return

    corr = scatter_df[["reference_count", "citation_count"]].corr().iloc[0, 1]
    ref_mean = float(scatter_df["reference_count"].mean())
    cit_mean = float(scatter_df["citation_count"].mean())

    fig = px.scatter(
        scatter_df,
        x="reference_count",
        y="citation_count",
        color="source" if "source" in scatter_df.columns else None,
        color_discrete_map=SOURCE_COLORS,
        hover_data={
            "title": True,
            "year": True,
            "venue": True,
            "reference_count": True,
            "citation_count": True,
        },
        title=f"Relação entre tamanho da bibliografia e impacto em citações (Correlação de Pearson r = {corr:.2f})",
        labels={
            "reference_count": "Referências citadas (bibliografia)",
            "citation_count": "Citações recebidas",
            "source": "Base",
        },
    )
    fig.update_layout(hovermode="closest")
    metric_row(
        [
            ("📚 Média de referências", f"{ref_mean:.1f}", None),
            ("⭐ Média de citações", f"{cit_mean:.1f}", None),
            ("📈 Correlação de Pearson (r)", f"{corr:.2f}", None),
        ]
    )
    render_chart(
        fig,
        caption="Examina se artigos que constroem uma fundamentação teórica com maior quantidade de "
        "referências tendem a receber mais citações no decorrer dos anos.",
    )


def _top_referenced(articles_df: pd.DataFrame) -> None:
    st.subheader("📖 Artigos com maior bibliografia (Revisões sistemáticas e surveys)")
    if (
        not require_columns(articles_df, ["reference_count"])
        or not (articles_df["reference_count"] > 0).any()
    ):
        st.info("Nenhum artigo com contagem de referências disponível nesta camada.")
        return

    top_ref = articles_df[articles_df["reference_count"] > 0].nlargest(20, "reference_count")
    article_table(
        top_ref,
        ["title", "year", "venue", "source", "reference_count", "citation_count", "doi"],
        download_key="artigos_mais_referenciados",
    )
    st.caption(
        "Artigos ordenados pelo tamanho da bibliografia — contagens altas são típicas de "
        "surveys, revisões de literatura abrangentes e estudos do estado da arte. Clique no DOI para abrir o artigo."
    )


def _top_cited(articles_df: pd.DataFrame) -> None:
    st.subheader("🏆 Artigos mais citados")
    if (
        not require_columns(articles_df, ["citation_count"])
        or not (articles_df["citation_count"] > 0).any()
    ):
        st.info("Nenhum artigo com contagem de citações disponível nesta camada.")
        return

    top_cited = articles_df[articles_df["citation_count"] > 0].nlargest(20, "citation_count")
    article_table(
        top_cited,
        ["title", "year", "venue", "source", "citation_count", "reference_count", "doi"],
        download_key="artigos_mais_citados",
    )
    st.caption(
        "Artigos ordenados pelo volume de citações acumuladas no corpus. Clique no link do DOI para abrir a publicação."
    )


def _cited_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("📅 Artigos citados por ano de publicação")
    has_citations = (
        "citation_count" in articles_df.columns and articles_df["citation_count"].notna().any()
    )
    has_year = "year" in articles_df.columns and articles_df["year"].notna().any()
    if not (has_citations and has_year):
        st.info("Colunas 'citation_count' e 'year' não disponíveis nesta camada.")
        return

    cited = articles_df.copy()
    cited["year"] = valid_years(cited)
    cited = cited.dropna(subset=["citation_count", "year"])
    cited = cited[cited["citation_count"] > 0].astype({"year": int})
    if cited.empty:
        st.info("Nenhum artigo com citações registradas nesta camada.")
        return

    counts = source_counts_by(cited, "year").sort_values("year")
    fig = source_lines(
        counts,
        "year",
        y_title="Artigos com citações registradas",
        spline=True,
        fill=True,
    )
    fig.update_layout(
        xaxis_title="Ano de publicação", hovermode="x unified", legend_title_text="Base"
    )
    render_chart(
        fig,
        caption="Quantidade de artigos publicados em cada ano que acumularam ao menos uma citação na "
        "literatura, por base e no total.",
    )


def _heavy_tail_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("📐 Modelagem de Cauda Pesada em Citações (Power-Law vs. Log-Normal)")
    st.caption(
        "Citações acadêmicas exibem assimetria extrema. Este painel ajusta distribuições de cauda pesada "
        "por Máxima Verossimilhança (MLE) e avalia a aderência pelo teste de Kolmogorov-Smirnov (KS)."
    )
    if "citation_count" not in articles_df.columns:
        st.info("Contagem de citações não disponível.")
        return

    from lake_research_map.dashboard.analytics import fit_heavy_tail_distributions

    cites = articles_df["citation_count"].dropna().to_numpy()
    fit_res = fit_heavy_tail_distributions(cites)
    if not fit_res.get("valid"):
        st.info("Dados insuficientes para ajuste estatístico de cauda pesada.")
        return

    models = fit_res["models"]
    best = fit_res["best_fit"]
    best_name = {
        "power_law": "Lei de Potência (Pareto)",
        "log_normal": "Log-Normal",
        "exponential": "Exponencial",
    }.get(best, best)

    metric_row(
        [
            ("🏆 Melhor Ajuste (KS)", best_name, f"Distância KS: {models[best]['ks_stat']:.4f}"),
            (
                "⚡ Expoente Power-Law (α)",
                f"{models['power_law']['alpha']:.2f}",
                f"x_min = {models['power_law']['x_min']:.0f}",
            ),
            (
                "📊 Média Log-Normal (μ)",
                f"{models['log_normal']['mu']:.2f}",
                f"σ = {models['log_normal']['sigma']:.2f}",
            ),
            (
                "📉 P-valor KS (Melhor)",
                f"{models[best]['p_value']:.4f}",
                "H0: aderência aos dados",
            ),
        ]
    )

    arr = cites[cites > 0]
    sorted_c = np.sort(arr)
    ccdf = 1.0 - np.arange(len(sorted_c)) / len(sorted_c)
    ccdf_df = pd.DataFrame({"citation_count": sorted_c, "ccdf": ccdf})

    fig = px.line(
        ccdf_df,
        x="citation_count",
        y="ccdf",
        log_x=True,
        log_y=True,
        title="Distribuição Acumulada Complementar Empírica (CCDF Log-Log)",
        labels={
            "citation_count": "Citações (escala log)",
            "ccdf": "P(Citações ≥ x) (escala log)",
        },
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    render_chart(
        fig,
        caption="Em escala log-log, uma Lei de Potência pura (Pareto) forma uma linha reta decrescente. "
        "A curvatura suave nos valores intermediários confirma que a distribuição Log-Normal "
        "frequentemente modela a literatura com maior fidelidade antes do regime assintótico.",
    )


def _age_normalized_rankings(articles_df: pd.DataFrame) -> None:
    st.subheader("⏳ Impacto Normalizado pela Idade do Artigo")
    st.caption(
        "Artigos antigos acumulam mais citações brutas por mera exposição temporal. "
        "A taxa anualizada de citações e o z-score por coorte anual de publicação revelam trabalhos "
        "recentes que estão alcançando velocidade de impacto excepcional."
    )
    from lake_research_map.dashboard.analytics import age_normalized_citations

    norm_df = age_normalized_citations(articles_df)
    if "citation_rate_annual" not in norm_df.columns:
        st.info("Dados insuficientes para normalização por idade.")
        return

    top_rate = norm_df.nlargest(20, "citation_rate_annual").copy()
    for col in (
        "citation_rate_annual",
        "cohort_citation_zscore",
        "cohort_citation_percentile",
    ):
        if col in top_rate.columns:
            top_rate[col] = top_rate[col].round(2)

    article_table(
        top_rate,
        [
            "title",
            "year",
            "venue",
            "source",
            "citation_count",
            "citation_rate_annual",
            "cohort_citation_percentile",
            "doi",
        ],
        download_key="impacto_normalizado_idade",
    )


def _citation_determinants_glm_view(articles_df: pd.DataFrame) -> None:
    st.subheader("Determinantes associados à taxa de citações")
    st.caption(
        "GLM de contagem com exposição pela idade do artigo, diagnóstico de sobredispersão e "
        "incerteza robusta. O IRR dos campos numéricos representa uma variação de um desvio padrão."
    )
    from lake_research_map.dashboard.analytics import citation_determinants_glm

    glm_res = citation_determinants_glm(articles_df)
    if not glm_res.get("valid"):
        st.info(glm_res.get("warning") or "Amostra insuficiente para o modelo de contagem.")
        return

    features = glm_res["features"]
    coefs = glm_res["coefficients"]
    irrs = glm_res["irr"]

    feat_labels = {
        "ano_publicacao": "Ano de Publicação (efeito do tempo)",
        "qtd_referencias": "Quantidade de Referências (embasamento)",
        "tamanho_equipe": "Tamanho da Equipe (autores)",
        "origem_ieee": "Publicado na IEEE (vs. Elsevier)",
    }

    glm_df = pd.DataFrame(
        {
            "Variável Explicativa": [feat_labels.get(f, f) for f in features],
            "Coeficiente (β)": [round(c, 4) for c in coefs],
            "IRR (Multiplicador de Citações)": [round(i, 4) for i in irrs],
            "IC 95% inferior": [round(i, 4) for i in glm_res["irr_lower"]],
            "IC 95% superior": [round(i, 4) for i in glm_res["irr_upper"]],
            "p-valor robusto": [round(i, 4) for i in glm_res["p_values"]],
        }
    )
    st.dataframe(glm_df, hide_index=True, width="stretch")
    st.caption(
        f"Família: {glm_res['family'].replace('_', ' ')} · dispersão Poisson: "
        f"{glm_res['dispersion']:.2f} · cobertura: {glm_res['n_used']}/{glm_res['n_total']} "
        f"({glm_res['coverage']:.1%}) · pseudo R²: {glm_res.get('score', 0):.3f}."
    )
    if glm_res.get("warning"):
        st.warning(glm_res["warning"])
