"""🏆 Highlights and Impact — citations, references, collaboration and most influential journals."""

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
        "Impact and citations",
        "Bibliometric analysis: reference basis, citation dynamics, heavy tail distributions and GLM determinants.",
    )

    articles_df = loaders.require_articles()

    tab_refs, tab_citations = st.tabs(
        ["📚 Theoretical Background (Reference)", "⭐ Citation Dynamics & Econometry"]
    )

    with tab_refs:
        ref_df = _reference_distribution_intro(articles_df)
        sub_dist, sub_vs_cit, sub_top = st.tabs(
            [
                "📊 Reference Distribution",
                "🔗 Refs vs. Quotations",
                "📖 Mais Referenciados",
            ]
        )
        with sub_dist:
            if ref_df is not None:
                view_mode = (
                    st.segmented_control(
                        "Distribution visualization format",
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
                "📅 Cited by Year",
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
    st.subheader("📚 Number of references used per article (IEEE vs. Elsevier)")
    if not require_columns(articles_df, ["reference_count"]) or not (
        articles_df["reference_count"].notna().any()
    ):
        st.info("No articles with reference counting available in this layer.")
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
                "📘 Mean Refs (IEEE)",
                f"{means['ieee']:.1f}" if means["ieee"] is not None else "N/D",
                f"Mediana: {ieee_med:.0f}"
                if ieee_med == ieee_med and ieee_med is not None
                else None,
            ),
            (
                "📙 Average Refs (Elsevier)",
                f"{means['elsevier']:.1f}" if means["elsevier"] is not None else "N/D",
                f"Mediana: {els_med:.0f}" if els_med == els_med and els_med is not None else None,
            ),
            ("📊 Mean Refs (Total)", f"{means['total']:.1f}", f"Mediana: {tot_med:.0f}"),
            ("🔝 Largest bibliography", f"{max_refs:,} refs", f"{len(ref_df):,} articles analyzed"),
        ]
    )
    st.caption(
        "Distribution of the number of references cited per article — such as histogram, cumulative curve"
        "(ECDF) and box plot, in the tabs below."
        "Xplore (`Reference Count`)."
        "(`data/enrichment_cache.json`); see the cards above for averages by source."
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
        labels={"reference_count": "References cited by article", "source": "Source"},
    )
    fig_hist.update_traces(
        hovertemplate="Interval: %{x} refs<br>Quantity: %{y:,} articles (%{data.name})<extra></extra>"
    )
    fig_hist.update_layout(
        xaxis_title="References cited by article (bibliography size)",
        yaxis_title="Number of articles",
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
        labels={"reference_count": "References cited by article", "source": "Source"},
    )
    fig_cdf.update_traces(
        hovertemplate="Up to %{x} refs: %{y:.1f}% of the articles (%{data.name})<extra></extra>"
    )
    fig_cdf.update_layout(
        xaxis_title="References cited by article (bibliography size)",
        yaxis_title="Accumulated % of articles",
        hovermode="x unified",
    )
    render_chart(
        fig_cdf,
        caption="Each point (x, y) reads: 'y% of the articles have up to x references cited'.",
    )


def _reference_box(ref_df: pd.DataFrame) -> None:
    fig_box = px.box(
        ref_df,
        x="source" if "source" in ref_df.columns else None,
        y="reference_count",
        color="source" if "source" in ref_df.columns else None,
        color_discrete_map=SOURCE_COLORS,
        points="outliers",
        labels={"reference_count": "References cited by article", "source": "Source"},
    )
    fig_box.update_layout(
        yaxis_title="References cited by article",
        xaxis_title="Source",
        showlegend=False,
    )
    render_chart(fig_box)


def _references_vs_citations(articles_df: pd.DataFrame) -> None:
    st.subheader("References cited vs. citations received")
    if (
        not require_columns(articles_df, ["reference_count", "citation_count"])
        or not articles_df["reference_count"].notna().any()
        or not articles_df["citation_count"].notna().any()
    ):
        return

    scatter_df = articles_df.dropna(subset=["reference_count", "citation_count"]).copy()
    scatter_df = scatter_df[scatter_df["reference_count"] > 0]
    if len(scatter_df) < 5:
        st.info("Few articles with reference data and citations available for correlation.")
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
        title=f"Relationship between bibliography size and citation impact (Pearson correlation r = {corr:.2f})",
        labels={
            "reference_count": "References cited (bibliography)",
            "citation_count": "Citations received",
            "source": "Source",
        },
    )
    fig.update_layout(hovermode="closest")
    metric_row(
        [
            ("📚 Average of references", f"{ref_mean:.1f}", None),
            ("", f"{cit_mean:.1f}", None),
            ("📈 Pearson's correlation (r)", f"{corr:.2f}", None),
        ]
    )
    render_chart(
        fig,
        caption="It examines articles that build a theoretical foundation with a greater number of"
        "references tend to receive more citations over the years.",
    )


def _top_referenced(articles_df: pd.DataFrame) -> None:
    st.subheader("📖 Articles with the highest bibliography (Systematic reviews and surveys)")
    if (
        not require_columns(articles_df, ["reference_count"])
        or not (articles_df["reference_count"] > 0).any()
    ):
        st.info("No articles with reference counting available in this layer.")
        return

    top_ref = articles_df[articles_df["reference_count"] > 0].nlargest(20, "reference_count")
    article_table(
        top_ref,
        ["title", "year", "venue", "source", "reference_count", "citation_count", "doi"],
        download_key="artigos_mais_referenciados",
    )
    st.caption(
        "Articles ordered by the size of the bibliography — high counts are typical of"
        "Researches, comprehensive literature reviews and state of the art studies."
    )


def _top_cited(articles_df: pd.DataFrame) -> None:
    st.subheader("🏆 Most cited articles")
    if (
        not require_columns(articles_df, ["citation_count"])
        or not (articles_df["citation_count"] > 0).any()
    ):
        st.info("No articles with citation counting available in this layer.")
        return

    top_cited = articles_df[articles_df["citation_count"] > 0].nlargest(20, "citation_count")
    article_table(
        top_cited,
        ["title", "year", "venue", "source", "citation_count", "reference_count", "doi"],
        download_key="artigos_mais_citados",
    )
    st.caption("Articles ordered by the volume of citations accumulated in the corpus.")


def _cited_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("Articles cited by year of publication")
    has_citations = (
        "citation_count" in articles_df.columns and articles_df["citation_count"].notna().any()
    )
    has_year = "year" in articles_df.columns and articles_df["year"].notna().any()
    if not (has_citations and has_year):
        st.info("'citation_count' and 'year' columns not available in this layer.")
        return

    cited = articles_df.copy()
    cited["year"] = valid_years(cited)
    cited = cited.dropna(subset=["citation_count", "year"])
    cited = cited[cited["citation_count"] > 0].astype({"year": int})
    if cited.empty:
        st.info("No article with citations recorded in this layer.")
        return

    counts = source_counts_by(cited, "year").sort_values("year")
    fig = source_lines(
        counts,
        "year",
        y_title="Articles with citations recorded",
        spline=True,
        fill=True,
    )
    fig.update_layout(
        xaxis_title="Year of publication", hovermode="x unified", legend_title_text="Source"
    )
    render_chart(
        fig,
        caption="Number of articles published in each year that accumulated at least one citation"
        "literature, by source and overall.",
    )


def _heavy_tail_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("📐 Heavy Caude Modeling in Quotations (Power-Law vs. Log-Normal)")
    st.caption(
        "Academic citations exhibit extreme asymmetry."
        "by Maximum Likelihood (MLE) and evaluates adherence by the Kolmogorov-Smirnov test (KS)."
    )
    if "citation_count" not in articles_df.columns:
        st.info("Count of citations not available.")
        return

    from lake_research_map.dashboard.analytics import fit_heavy_tail_distributions

    cites = articles_df["citation_count"].dropna().to_numpy()
    fit_res = fit_heavy_tail_distributions(cites)
    if not fit_res.get("valid"):
        st.info("Insufficient data for statistical adjustment of heavy tail.")
        return

    models = fit_res["models"]
    best = fit_res["best_fit"]
    best_name = {
        "power_law": "Power Law (Pareto)",
        "log_normal": "Log-Normal",
        "exponential": "Exponencial",
    }.get(best, best)

    metric_row(
        [
            ("🏆 Best fit (KS)", best_name, f"KS distance: {models[best]['ks_stat']:.4f}"),
            (
                "⚡ Expoente Power-Law (α)",
                f"{models['power_law']['alpha']:.2f}",
                f"x_min = {models['power_law']['x_min']:.0f}",
            ),
            (
                "📊 Log-Normal Mean (μ)",
                f"{models['log_normal']['mu']:.2f}",
                f"σ = {models['log_normal']['sigma']:.2f}",
            ),
            (
                "📉 P-value KS (Best)",
                f"{models[best]['p_value']:.4f}",
                "H0: adherence to data",
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
        title="Empirical Complementary Accumulated Distribution (CCDF Log-Log)",
        labels={
            "citation_count": "Quotations (log scale)",
            "ccdf": "P(Citations ≥ x) (log scale)",
        },
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    render_chart(
        fig,
        caption="On log-log scale, a pure Power Law (Pareto) forms a decreasing straight line."
        "The smooth curvature in the intermediate values confirms that the Log-Normal distribution"
        "It often models the literature with greater fidelity before the asymptotic regime.",
    )


def _age_normalized_rankings(articles_df: pd.DataFrame) -> None:
    st.subheader("⏳ Impact Normalized by the Article Age")
    st.caption(
        "Old articles accumulate more gross citations due to mere temporal exposure."
        "Annualized citation rate and z-score by annual publication cohort reveal studies"
        "recent that are reaching exceptional impact velocity."
    )
    from lake_research_map.dashboard.analytics import age_normalized_citations

    norm_df = age_normalized_citations(articles_df)
    if "citation_rate_annual" not in norm_df.columns:
        st.info("Insufficient data for normalization by age.")
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
    st.subheader("Determinants associated with citation rate")
    st.caption(
        "GLM of counting with exposure by the age of the article, diagnosis of overdispersity and"
        "The IRR of the numerical fields represents a variation of a standard deviation."
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
        "ano_publicacao": "Year of Publication (time effect)",
        "qtd_referencias": "Number of references (basement)",
        "tamanho_equipe": "Team Size (authors)",
        "origem_ieee": "Publicado na IEEE (vs. Elsevier)",
    }

    glm_df = pd.DataFrame(
        {
            "Explanatory Variable": [feat_labels.get(f, f) for f in features],
            "Coeficiente (β)": [round(c, 4) for c in coefs],
            "IRR (Citation Multiplier)": [round(i, 4) for i in irrs],
            "IC 95% inferior": [round(i, 4) for i in glm_res["irr_lower"]],
            "IC 95% superior": [round(i, 4) for i in glm_res["irr_upper"]],
            "robust p-value": [round(i, 4) for i in glm_res["p_values"]],
        }
    )
    st.dataframe(glm_df, hide_index=True, width="stretch")
    st.caption(
        f"Family: {glm_res['family'].replace('_', ' ')} · Poisson dispersion: "
        f"{glm_res['dispersion']:.2f} · cobertura: {glm_res['n_used']}/{glm_res['n_total']} "
        f"({glm_res['coverage']:.1%}) · pseudo R²: {glm_res.get('score', 0):.3f}."
    )
    if glm_res.get("warning"):
        st.warning(glm_res["warning"])
