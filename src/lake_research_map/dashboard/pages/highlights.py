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
        ["Theoretical background", "Citation dynamics and econometrics"],
        on_change="rerun",
        key="highlights_primary_tab",
    )

    if tab_refs.open:
        with tab_refs:
            ref_df = _reference_distribution_intro(articles_df)
            reference_view = st.segmented_control(
                "Reference analysis",
                options=["Distribution", "References vs. citations", "Most referenced"],
                default="Distribution",
                key="reference_analysis_view",
            )
            if reference_view == "Distribution" and ref_df is not None:
                view_mode = (
                    st.segmented_control(
                        "Distribution visualization format",
                        options=["Histogram", "Cumulative curve (ECDF)", "Box plot"],
                        default="Histogram",
                        key="ref_dist_mode",
                    )
                    or "Histogram"
                )
                if view_mode == "Histogram":
                    _reference_histogram(ref_df)
                elif view_mode == "Cumulative curve (ECDF)":
                    _reference_ecdf(ref_df)
                else:
                    _reference_box(ref_df)
            elif reference_view == "References vs. citations":
                _references_vs_citations(articles_df)
            elif reference_view == "Most referenced":
                _top_referenced(articles_df)
    elif tab_citations.open:
        with tab_citations:
            citation_view = st.segmented_control(
                "Citation analysis",
                options=[
                    "Most cited",
                    "Cited by year",
                    "Heavy-tail diagnostics",
                    "Age-normalized impact",
                    "GLM determinants",
                ],
                default="Most cited",
                key="citation_analysis_view",
            )
            if citation_view == "Most cited":
                _top_cited(articles_df)
            elif citation_view == "Cited by year":
                _cited_by_year(articles_df)
            elif citation_view == "Heavy-tail diagnostics":
                _heavy_tail_analysis(articles_df)
            elif citation_view == "Age-normalized impact":
                _age_normalized_rankings(articles_df)
            else:
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
                f"{means['ieee']:.1f}" if means["ieee"] is not None else "n/a",
                f"Median: {ieee_med:.0f}"
                if ieee_med == ieee_med and ieee_med is not None
                else None,
            ),
            (
                "📙 Average Refs (Elsevier)",
                f"{means['elsevier']:.1f}" if means["elsevier"] is not None else "n/a",
                f"Median: {els_med:.0f}" if els_med == els_med and els_med is not None else None,
            ),
            ("📊 Mean Refs (Total)", f"{means['total']:.1f}", f"Median: {tot_med:.0f}"),
            ("🔝 Largest bibliography", f"{max_refs:,} refs", f"{len(ref_df):,} articles analyzed"),
        ]
    )
    st.caption(
        "Distribution of the number of references cited per article — such as histogram, cumulative curve "
        "(ECDF) and box plot, in the tabs below. "
        "Xplore (`Reference Count`). "
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
        yaxis_title="Cumulative % of articles",
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
            ("📑 Average citations", f"{cit_mean:.1f}", None),
            ("📈 Pearson's correlation (r)", f"{corr:.2f}", None),
        ]
    )
    render_chart(
        fig,
        caption="It examines articles that build a theoretical foundation with a greater number of "
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
        download_key="most_referenced_articles",
    )
    st.caption(
        "Articles ordered by the size of their bibliography — high counts are typical of "
        "surveys, comprehensive literature reviews and state-of-the-art studies."
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
        download_key="most_cited_articles",
    )
    st.caption("Articles ordered by the volume of citations cumulative in the corpus.")


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
        caption="Number of articles published in each year that cumulative at least one citation "
        "literature, by source and overall.",
    )


_HEAVY_TAIL_NAMES = {
    "power_law": "Power law (Pareto)",
    "log_normal": "Log-normal",
    "exponential": "Exponential",
}


def _heavy_tail_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("📐 Heavy-tail modelling of citations (power law vs. log-normal)")
    st.caption(
        "Citation counts are extremely skewed. Each family is fitted by maximum likelihood "
        "above a data-driven x_min, and the families are compared by AIC rather than by "
        "goodness-of-fit alone — a lower KS distance does not by itself favour a model "
        "with more free parameters."
    )
    if "citation_count" not in articles_df.columns:
        st.info("Citation counts are not available for this population.")
        return

    from lake_research_map.dashboard.analytics import fit_heavy_tail_distributions

    cites = articles_df["citation_count"].dropna().to_numpy()
    fit_res = fit_heavy_tail_distributions(cites)
    if not fit_res.get("valid"):
        st.info("Not enough citation mass to fit a heavy tail.")
        return

    models = fit_res["models"]
    best = fit_res["best_fit"]
    best_name = _HEAVY_TAIL_NAMES.get(best, best)

    metric_row(
        [
            (
                "🏆 Best fit (AIC)",
                best_name,
                f"AIC {models[best]['aic']:.1f} · KS {models[best]['ks_stat']:.4f}",
            ),
            (
                "⚡ Power-law exponent (α)",
                f"{models['power_law']['alpha']:.2f}",
                f"x_min = {models['power_law']['x_min']:.0f}",
            ),
            (
                "📊 Log-normal mean (μ)",
                f"{models['log_normal']['mu']:.2f}",
                f"σ = {models['log_normal']['sigma']:.2f}",
            ),
            (
                "🧮 Population used",
                f"{fit_res['tail_n']:,}",
                f"of {fit_res['n']:,} articles · {fit_res['zero_count']:,} with zero citations",
            ),
        ]
    )

    # The bootstrap p-value resamples from the fitted Pareto and refits alpha per
    # sample, so it is the only one here that does not reuse its own parameters.
    # The log-normal and exponential p-values do, which makes them optimistic --
    # labelling them together as one "KS p-value" hid exactly that difference.
    comparison = pd.DataFrame(
        [
            {
                "Family": _HEAVY_TAIL_NAMES.get(name, name),
                "AIC": round(model["aic"], 1),
                "ΔAIC": round(model["aic"] - models[best]["aic"], 1),
                "Log-likelihood": round(model["log_likelihood"], 1),
                "KS distance": round(model["ks_stat"], 4),
            }
            for name, model in models.items()
        ]
    ).sort_values("AIC")
    st.dataframe(comparison, hide_index=True, width="stretch")

    ratios = fit_res["log_likelihood_ratios"]
    st.caption(
        f"Bootstrap goodness-of-fit for the power law: p = "
        f"{models['power_law']['p_value']:.4f} (H0: the data are Pareto above x_min; "
        "refitted per simulated sample). Log-likelihood ratios — power law vs. log-normal "
        f"{ratios['power_law_vs_log_normal']:+.1f}, vs. exponential "
        f"{ratios['power_law_vs_exponential']:+.1f}; a positive value favours the power law. "
        "The ratios are reported without a significance test, so treat them as descriptive."
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
        title="Empirical complementary cumulative distribution (CCDF, log-log)",
        labels={
            "citation_count": "Citations (log scale)",
            "ccdf": "P(Citations ≥ x) (log scale)",
        },
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    render_chart(
        fig,
        caption="On log-log axes a pure power law is a straight descending line, while a log-normal "
        "curves through the middle of the range before the tail straightens out. Read the shape "
        "against the AIC comparison above rather than instead of it: the eye is a poor judge of "
        "which family fits a heavy tail, which is why the selection is made by likelihood.",
    )


def _age_normalized_rankings(articles_df: pd.DataFrame) -> None:
    observation_year, observation_basis = loaders.citation_observation_context()
    st.subheader("⏳ Impact Normalized by the Article Age")
    st.caption(
        "Old articles accumulate more gross citations due to mere temporal exposure. "
        "Annualized citation rate and z-score by annual publication cohort reveal studies "
        "recent that are reaching exceptional impact velocity. "
        f"Rates use observation year {observation_year} ({observation_basis})."
    )
    from lake_research_map.dashboard.analytics import age_normalized_citations

    norm_df = age_normalized_citations(articles_df, observation_year=observation_year)
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


# Internal design-matrix column names kept for schema stability; the dashboard
# renders these English labels instead (PRD NFR-07).
_GLM_FEATURE_LABELS = {
    "ano_publicacao": "Publication year (time effect)",
    "qtd_referencias": "Reference count",
    "tamanho_equipe": "Team size (authors)",
    "origem_ieee": "Published in IEEE (vs. Elsevier)",
}


def _citation_determinants_glm_view(articles_df: pd.DataFrame) -> None:
    observation_year, observation_basis = loaders.citation_observation_context()
    st.subheader("Determinants associated with citation rate")
    st.caption(
        "Count GLM with an exposure offset for article age. The IRR of a numeric field is the "
        "multiplier for a one-standard-deviation change. These are conditional associations "
        "within this corpus, not causal effects — the source indicator in particular also "
        "encodes source-specific missingness. "
        f"Article age is measured at {observation_year} ({observation_basis})."
    )
    from lake_research_map.dashboard.analytics import citation_determinants_glm

    glm_res = citation_determinants_glm(articles_df, observation_year=observation_year)
    if not glm_res.get("valid"):
        st.info(glm_res.get("warning") or "Sample too small for the count model.")
        return

    features = glm_res["features"]
    coefs = glm_res["coefficients"]
    irrs = glm_res["irr"]

    feat_labels = _GLM_FEATURE_LABELS

    glm_df = pd.DataFrame(
        {
            "Explanatory variable": [feat_labels.get(f, f) for f in features],
            "Coefficient (β)": [round(c, 4) for c in coefs],
            "IRR (citation multiplier)": [round(i, 4) for i in irrs],
            "95% CI lower": [round(i, 4) for i in glm_res["irr_lower"]],
            "95% CI upper": [round(i, 4) for i in glm_res["irr_upper"]],
            "Robust p-value": [round(i, 4) for i in glm_res["p_values"]],
        }
    )
    st.dataframe(glm_df, hide_index=True, width="stretch")
    st.caption(
        f"Family: {glm_res['family'].replace('_', ' ')}, selected by AIC over "
        f"{len(glm_res.get('candidate_aic') or {})} candidates · Poisson dispersion "
        f"{glm_res['dispersion']:.2f} · coverage {glm_res['n_used']}/{glm_res['n_total']} "
        f"({glm_res['coverage']:.1%}) · pseudo R² {glm_res.get('score', 0):.3f}."
    )
    if glm_res.get("warning"):
        st.warning(glm_res["warning"])

    _glm_specification_diagnostics(glm_res)


def _family_aic_metric(glm_res: dict) -> tuple[str, str, str]:
    """Report the AIC of every candidate family beside the one AIC selected.

    `candidate_aic` maps each family that converged to its AIC, and the model
    now selects the minimum rather than applying a dispersion threshold. Listing
    the losers is what lets a reader see how decisive that choice was: two
    families a point apart is a different claim from two hundred apart.
    """
    candidates = glm_res.get("candidate_aic") or {}
    selected = glm_res["family"]
    readable = selected.replace("_", " ")
    if not candidates:
        return ("\U0001f9ee Family AIC", "n/a", f"selected: {readable}")

    detail = " \u00b7 ".join(
        f"{name.replace('_', ' ')} {value:.1f}" for name, value in sorted(candidates.items())
    )
    if glm_res.get("zero_inflated_status") == "did_not_converge":
        detail += " \u2014 no zero-inflated fit converged"
    value = f"{candidates[selected]:.1f}" if selected in candidates else "n/a"
    return (f"\U0001f9ee AIC ({readable})", value, detail)


def _age_specification_table(glm_res: dict, feat_labels: dict[str, str]) -> None:
    """Show whether each association survives a different exposure assumption.

    `log(age + 1)` as a fixed-coefficient offset forces citations to accumulate
    exactly proportionally to log age. WP-15 requires the alternative
    specifications to be visible, because a predictor whose sign flips between
    them is an artifact of that assumption rather than a finding.
    """
    specifications = glm_res.get("age_specifications") or []
    if len(specifications) < 2:
        return

    readable = {
        "offset_log_age": "log(age+1) as offset (default)",
        "covariate_log_age": "log(age+1) as free covariate",
        "covariate_linear_age": "age as free covariate",
    }
    rows = []
    for spec in specifications:
        row = {
            "Age specification": readable.get(spec["specification"], spec["specification"]),
            "AIC": round(float(spec["aic"]), 1),
        }
        for feature, coefficient in spec["coefficients"].items():
            row[f"\u03b2 {feat_labels.get(feature, feature)}"] = round(float(coefficient), 4)
        rows.append(row)

    st.markdown("**Sensitivity to the age specification**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if glm_res.get("age_specification_signs_agree"):
        st.caption(
            "Every coefficient keeps its sign across the three exposure choices, so the "
            "reported associations are not an artifact of treating log age as a fixed offset."
        )
    else:
        st.warning(
            "At least one coefficient changes sign when the age term is estimated instead of "
            "fixed. Treat that association as an artifact of the exposure choice, not a finding."
        )


def _influence_delta(glm_res: dict, max_cooks: float | None) -> str:
    """Caption the influence count, naming the fit it was measured on.

    Cook's distance needs a hat matrix, which a zero-inflated fit has not got,
    so the model falls back to the Poisson GLM. Saying which fit produced the
    number keeps it from reading as a diagnostic of the selected family.
    """
    if max_cooks is None or not pd.notna(max_cooks):
        return "Cook's distance unavailable for this fit"
    basis = glm_res.get("influence_basis")
    suffix = ""
    if basis and basis not in {glm_res.get("family"), "unavailable"}:
        suffix = f", measured on the {basis.replace('_', ' ')} fit"
    return f"max Cook's distance {max_cooks:.3f}{suffix}"


def _glm_specification_diagnostics(glm_res: dict) -> None:
    """Show the misspecification evidence the model already computes.

    A coefficient table on its own invites a confirmatory reading. PRD section
    8.1 requires multicollinearity, influence and zero-inflation to be visible
    beside it, and every number below is already in the model result.
    """
    with st.expander("Specification diagnostics", expanded=False):
        condition_number = glm_res.get("condition_number")
        max_cooks = glm_res.get("max_cooks_distance")
        observed_zero = glm_res.get("observed_zero_fraction")
        predicted_zero = glm_res.get("predicted_zero_fraction")
        zero_gap = glm_res.get("zero_inflation_gap")

        metric_row(
            [
                (
                    "📐 Condition number",
                    "n/a" if condition_number is None else f"{condition_number:.1f}",
                    "Above ~30 indicates collinear predictors",
                ),
                (
                    "🎯 Influential observations",
                    f"{glm_res.get('influential_count', 0):,}",
                    _influence_delta(glm_res, max_cooks),
                ),
                (
                    "⚠️ Zero-inflation gap",
                    "n/a" if zero_gap is None else f"{zero_gap:+.1%}",
                    "n/a"
                    if observed_zero is None or predicted_zero is None
                    else f"observed {observed_zero:.1%} vs. predicted {predicted_zero:.1%}",
                ),
                _family_aic_metric(glm_res),
            ]
        )

        vif = glm_res.get("vif") or {}
        if vif:
            st.dataframe(
                pd.DataFrame(
                    {"Predictor": list(vif), "VIF": [round(float(v), 2) for v in vif.values()]}
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption("Variance inflation factor; above 5 a coefficient is hard to read.")

        missingness = glm_res.get("missingness") or {}
        if missingness:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Field": list(missingness),
                        "Missing": [
                            f"{float(value):.1%}" if isinstance(value, int | float) else value
                            for value in missingness.values()
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Rows with any missing predictor are dropped, so a field that is missing "
                "unevenly across sources also shifts which articles the model sees."
            )

        _age_specification_table(glm_res, _GLM_FEATURE_LABELS)

        if glm_res.get("zero_inflated_status") == "fitted":
            st.caption(
                "A positive zero-inflation gap means the model under-predicts articles with "
                "zero citations. Zero-inflated Poisson and negative-binomial models were "
                "fitted as candidates and compared by AIC, so a gap that survives selection "
                "is a property of the corpus rather than an uncorrected misspecification."
            )
        else:
            st.caption(
                "A positive zero-inflation gap means the model under-predicts articles with "
                "zero citations. No zero-inflated candidate converged under the current filters, "
                "gap marks these estimates as exploratory rather than being corrected for."
            )
