"""Topics and venues: where the corpus publishes and how its themes evolved."""

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
        "Topics and scientific structure",
        "Where the corpus publishes, about what, and how the themes evolved over time.",
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
        ["Journals", "Keywords", "Explorer", "Temporal evolution"],
        on_change="rerun",
        key="topics_primary_tab",
    )

    if tab_venues.open:
        with tab_venues:
            venue_view = st.segmented_control(
                "Venue analysis",
                options=["Ranking and impact", "CAPES/Qualis", "Bradford zones", "Semantics"],
                default="Ranking and impact",
                key="topics_venue_view",
            )
            if venue_view == "Ranking and impact":
                _top_venues(articles_df)
            elif require_columns(articles_df, ["venue"]) and articles_df["venue"].notna().any():
                match_df, with_estrato, totals_by_estrato, estrato_order = _qualis_match_data(
                    articles_df
                )
                if venue_view == "CAPES/Qualis":
                    qualis_view = (
                        st.segmented_control(
                            "CAPES/Qualis visualization format",
                            options=[
                                "Detailed table",
                                "Totals by stratum",
                                "Cumulative evolution",
                                "A1–A3 subset",
                            ],
                            default="Detailed table",
                            key="qualis_view_selector",
                        )
                        or "Detailed table"
                    )
                    if qualis_view == "Detailed table":
                        _qualis_table(
                            articles_df,
                            match_df,
                            with_estrato[with_estrato["estrato"] == "A1"],
                        )
                    elif qualis_view == "Totals by stratum":
                        _qualis_totals_chart(totals_by_estrato, estrato_order)
                    elif qualis_view == "Cumulative evolution":
                        _qualis_cumulative_chart(with_estrato, estrato_order)
                    else:
                        _qualis_a1_a3_combined(with_estrato)
                elif venue_view == "Bradford zones":
                    _bradford_analysis(articles_df)
                else:
                    _semantic_venues_analysis(articles_df)
    elif tab_keywords.open:
        with tab_keywords:
            keyword_view = st.segmented_control(
                "Keyword analysis",
                options=[
                    "Top keywords",
                    "Vocabulary structure",
                    "Dynamic vocabulary",
                    "Atypicality",
                ],
                default="Top keywords",
                key="topics_keyword_view",
            )
            if keyword_view == "Top keywords":
                _top_keywords(articles_df)
            elif keyword_view == "Vocabulary structure":
                if all_keywords:
                    _keyword_stats(kw_lists, all_keywords)
                    st.divider()
                    _zipf_analysis(articles_df)
                else:
                    st.info("No keywords identified in this layer.")
            elif keyword_view == "Dynamic vocabulary":
                _dynamic_ctfidf_analysis(articles_df)
            else:
                _conceptual_atypicality_tab(articles_df)
    elif tab_explorer.open:
        with tab_explorer:
            if all_keywords:
                _keyword_explorer(articles_df, kw_lists, all_keywords)
            else:
                st.info("No keywords identified in this layer.")
    elif tab_trends.open:
        with tab_trends:
            if not all_keywords:
                st.info("No keywords identified in this layer.")
            else:
                kw_year = _prepare_keyword_trend_data(articles_df)
                if kw_year is None:
                    st.info("Insufficient data to analyze temporal trends.")
                else:
                    trend_view = st.segmented_control(
                        "Trend analysis",
                        options=[
                            "Annual share",
                            "Rise vs. decline",
                            "First appearance",
                            "Breakpoints",
                        ],
                        default="Annual share",
                        key="topics_trend_view",
                    )
                    if trend_view == "Annual share":
                        _topic_share_area(kw_year)
                    elif trend_view == "Rise vs. decline":
                        _rising_falling(kw_year)
                    elif trend_view == "First appearance":
                        _first_appearance(kw_year)
                    else:
                        _structural_breaks_tab(articles_df)


def _top_venues(articles_df: pd.DataFrame) -> None:
    st.subheader("Journals and conferences")
    if not require_columns(articles_df, ["venue"]) or not articles_df["venue"].notna().any():
        return

    rank_mode = (
        st.segmented_control(
            "Highlight Criteria",
            options=["Volume of Articles", "Average citation impact"],
            default="Volume of Articles",
            key="topics_venue_rank_mode",
        )
        or "Volume of Articles"
    )

    if rank_mode == "Volume of Articles":
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
            x_title="Number of articles",
            y_title="Journal / conference",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} published articles<extra></extra>")
        render_chart(
            fig,
            caption="Editorial concentration of the corpus — each journal is colored by the predominant basis of the "
            "articles published in it.",
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
            st.info("No journal with enough articles with citation counting in this layer.")
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
            x_title="Average citations per article",
            y_title="Journal / conference",
        )
        for trace in fig.data:
            trace.customdata = venue_impact["articles"].reindex(trace.y).to_numpy().reshape(-1, 1)
            trace.hovertemplate = "<b>%{y}</b><br>%{x:.1f} citations/article (%{customdata[0]:,} analyzed articles)<extra></extra>"
        render_chart(
            fig,
            caption="Average citations per article for vehicles with at least 3 publications in the corpus.",
        )


def _top_keywords(articles_df: pd.DataFrame) -> None:
    st.subheader("Keywords")
    if not require_columns(articles_df, ["keywords"]):
        return

    kw_exploded = explode_keywords(articles_df)
    if kw_exploded.empty:
        st.info("No keywords identified in this layer.")
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
        x_title="Number of articles",
        y_title="Keyword",
        title="Top 20 keywords",
    )
    for trace in fig.data:
        breakdown = grouped.loc[list(trace.y), ["ieee", "elsevier"]].to_numpy()
        trace.customdata = breakdown
        trace.hovertemplate = (
            "<b>%{y}</b><br> "
            "Total articles: %{x:,}<br> "
            "• Articles IEEE: %{customdata[0]:,}<br> "
            "• Elsevier Articles: %{customdata[1]:,}<extra></extra>"
        )
    render_chart(
        fig,
        caption="Hover over the bars to inspect the exact IEEE and Elsevier breakdown.",
    )


def _keyword_stats(kw_lists: pd.Series, all_keywords: list[str]) -> None:
    st.subheader("Vocabulary Statistics")
    kw_counts = kw_lists.apply(len)
    metric_row(
        [
            ("🔤 Keyword occurrences", f"{len(all_keywords):,}", None),
            ("📊 Average terms per article", f"{kw_counts.mean():.1f}", None),
            ("🚫 Articles without keywords", f"{(kw_counts == 0).mean():.0%}", None),
        ]
    )


def _zipf_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("📖 Zipf Law of the Technical Vocabulary")
    st.caption(
        "Zipf's law states that word frequency is inversely proportional to rank "
        "($f \\propto 1/r^\\gamma$). In established bibliometric corpora, a coefficient near "
        "$\\gamma \\approx 1.0$ indicates a vocabulary with both recurring core terms and a long "
        "tail of specialized language."
    )

    zipf_res = zipf_law_analysis(articles_df)
    if not zipf_res["valid"]:
        st.info("Not enough text to evaluate Zipf's law.")
        return

    metric_row(
        [
            ("📐 Slope coefficient (γ)", f"{zipf_res['gamma']:.2f}", "Theoretical target: ~1.0"),
            (
                "Determination (R2)",
                f"{zipf_res['r_squared']:.3f}",
                "Quality of log-log adjustment",
            ),
            ("📚 Unique vocabulary", f"{zipf_res['vocab_size']:,} terms", None),
            ("📝 Total Occurrence", f"{zipf_res['total_tokens']:,} palavras", None),
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
            name="Real Frequency",
            text=plot_df["termo"],
            marker=dict(size=6, color="#2a78d6", opacity=0.7),
            hovertemplate="<b>%{text}</b><br>Post: %{x}<br>Frequence: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["posto"],
            y=plot_df["ajuste"],
            mode="lines",
            name=f"Zipf regression (γ = {zipf_res['gamma']:.2f})",
            line=dict(color="#eb6834", width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis=dict(type="log", title="Term rank (log r)"),
        yaxis=dict(type="log", title="Frequency of Occurrence (log f)"),
        title="Vocabulary Post-Frequence Distribution (Log-Log Scale)",
        height=480,
    )
    render_chart(
        fig,
        caption="The proximity of the points in relation to the traced line confirms the adherence of the corpus to the Zipf Law.",
    )

    st.markdown("##### 📋 Top 30 term comparison")
    st.dataframe(zipf_res["top_words_df"], hide_index=True, width="stretch")


def _dynamic_ctfidf_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🧬 Dynamic Vocabulary by Theme and Time (c-TF-IDF)")
    st.caption(
        "The c-TF-IDF (Class-based TF-IDF) algorithm extracts the terms that most differentiate each thematic theme "
        "reveals the evolution of technological topics "
        "in the literature of distribution planning."
    )

    signals = loaders.semantics()
    if signals.empty or "theme_label" not in signals.columns:
        st.info("The semantic layer with theme assignment is necessary for this analysis.")
        return

    merged = pd.merge(articles_df, signals[["doi", "theme_label"]], on="doi", how="inner")
    ctfidf_res = dynamic_topic_ctfidf(merged)

    if not ctfidf_res["valid"]:
        st.info("Insufficient temporal and thematic data to segment the dynamic c-TF-IDF.")
        return

    st.dataframe(ctfidf_res["summary_df"], hide_index=True, width="stretch")


def _keyword_explorer(
    articles_df: pd.DataFrame, kw_lists: pd.Series, all_keywords: list[str]
) -> None:
    st.subheader("🔎 Explorer: filter articles by keyword")
    selected = st.multiselect(
        "Select one or more terms (articles containing any of them will be displayed):",
        options=all_keywords,
    )
    if not selected:
        st.caption("Select keywords above to filter the corresponding articles.")
        return

    selected_set = set(selected)
    mask = kw_lists.apply(lambda kws: bool(set(kws) & selected_set))
    filtered = articles_df.loc[mask]
    st.caption(f"{len(filtered):,} articles found")

    if "citation_count" in filtered.columns:
        filtered = filtered.sort_values("citation_count", ascending=False, na_position="last")
    article_table(
        filtered,
        ["title", "year", "venue", "source", "citation_count", "reference_count", "doi"],
        download_key="articles_by_keyword",
    )


def _prepare_keyword_trend_data(articles_df: pd.DataFrame) -> pd.DataFrame | None:
    """Shared prep for the three "Time Evolution" sub-tabs, or None if there's not enough data."""
    kw_year = explode_keywords(articles_df)
    if kw_year.empty or "year" not in kw_year.columns:
        return None
    kw_year["year"] = valid_years(kw_year, lo=TREND_MIN_YEAR)
    kw_year = kw_year.dropna(subset=["year"]).astype({"year": int})
    if len(kw_year) < 30:
        return None
    return kw_year


def _topic_share_area(kw_year: pd.DataFrame) -> None:
    st.markdown("**Annual share of the main keywords**")
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
        title=f"Share (%) of the top {TOP_KEYWORDS_TREND} keywords by year",
        groupnorm="percent",
    )
    fig.update_layout(
        xaxis_title=f"Publication year (since {TREND_MIN_YEAR})",
        yaxis_title="Participation among the mentions of the year (%)",
        legend_title_text="Keyword",
    )
    render_chart(
        fig,
        caption=f"100% stacked area: shows how each term's relative weight changed year by year among the "
        f"top {TOP_KEYWORDS_TREND} keywords in the corpus.",
    )


def _rising_falling(kw_year: pd.DataFrame) -> None:
    st.markdown("**Rising vs. declining terms**")
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= MIN_KEYWORD_OCCURRENCES].index

    from lake_research_map.dashboard.analytics import linear_slope_with_ci

    by_year_total = kw_year.groupby("year").size()
    fits: dict[str, dict[str, float]] = {}
    for kw in eligible:
        yearly = kw_year[kw_year["keyword"] == kw].groupby("year").size()
        share = (yearly / by_year_total.reindex(yearly.index)).fillna(0.0) * 100
        if len(share) < 3:
            continue
        fit = linear_slope_with_ci(share.index.to_numpy(dtype=float), share.to_numpy(dtype=float))
        if np.isfinite(fit["slope"]):
            fits[kw] = fit

    if not fits:
        st.info(
            f"No term has at least {MIN_KEYWORD_OCCURRENCES} occurrences for trend calculation."
        )
        return

    slope_series = pd.Series({kw: fit["slope"] for kw, fit in fits.items()}).sort_values()
    top_slopes = pd.concat([slope_series.head(7), slope_series.tail(7)]).drop_duplicates()
    df = top_slopes.rename_axis("keyword").reset_index(name="slope")
    df["direction"] = df["slope"].apply(lambda s: "Rising" if s >= 0 else "Falling")
    # Error bars carry the sampling spread the ranking hides: on series this
    # short the interval very often straddles zero even at the extremes.
    df["ci_low"] = df["keyword"].map(lambda kw: fits[kw]["ci_low"])
    df["ci_high"] = df["keyword"].map(lambda kw: fits[kw]["ci_high"])
    df["err_plus"] = df["ci_high"] - df["slope"]
    df["err_minus"] = df["slope"] - df["ci_low"]
    df["excludes_zero"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    df = df.sort_values("slope")

    fig = px.bar(
        df,
        x="slope",
        y="keyword",
        orientation="h",
        color="direction",
        color_discrete_map={"Rising": TREND_UP_COLOR, "Falling": TREND_DOWN_COLOR},
        error_x="err_plus",
        error_x_minus="err_minus",
        title="Annual participation slope (linear regression, 95% CI)",
        labels={
            "slope": "Annual variation in participation (p.p./year)",
            "keyword": "Keyword",
            "direction": "Trend",
        },
        custom_data=["ci_low", "ci_high"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>%{x:+.2f} pp/year"
            "<br>95% CI %{customdata[0]:+.2f} to %{customdata[1]:+.2f}<extra></extra>"
        )
    )
    fig.update_layout(legend_title_text="Trend")
    decisive = int(df["excludes_zero"].sum())
    render_chart(
        fig,
        caption=f"Slope of each term's annual percentage share (minimum {MIN_KEYWORD_OCCURRENCES} "
        "occurrences in the period), by ordinary least squares. Bars show the 95% confidence "
        f"interval: {decisive} of the {len(df)} terms shown have an interval that excludes zero, "
        "so the rest are ranked by a slope the data cannot separate from no trend.",
    )

    from lake_research_map.dashboard.analytics import benjamini_hochberg, mann_kendall_trend

    # The test family is every eligible keyword, not the extremes plotted above.
    # Adjusting only the top/bottom slopes would correct a set already chosen for
    # being extreme, which inflates significance instead of controlling it -- so
    # Mann-Kendall runs over the full family, Benjamini-Hochberg is applied there,
    # and only then is the table narrowed to the terms on the chart.
    pivoted = kw_year.groupby(["year", "keyword"]).size().unstack(fill_value=0)
    mk_records = []
    for kw in eligible:
        if kw in pivoted.columns:
            series = pivoted[kw].to_numpy()
            res = mann_kendall_trend(series)
            mk_records.append(
                {
                    "Keyword": kw,
                    "Trend": res["trend"].title(),
                    "Raw p-value": float(res["p_value"]),
                    # Mann-Kendall assumes independent years, which annual
                    # counts are not. The Hamed-Rao variance inflation is
                    # reported beside the raw value as a sensitivity rather
                    # than swapped in: it changes which terms look significant.
                    "Serial-corrected p-value": float(res["p_value_serial_corrected"]),
                    "Sen slope": round(res["slope"], 3),
                }
            )
    if mk_records:
        family = pd.DataFrame(mk_records)
        family["Adjusted p-value"] = benjamini_hochberg(family["Raw p-value"])
        family["Adjusted p-value (serial)"] = benjamini_hochberg(family["Serial-corrected p-value"])
        family["FDR significant"] = family["Adjusted p-value"] < 0.05
        family["FDR significant (serial)"] = family["Adjusted p-value (serial)"] < 0.05
        survives = int((family["FDR significant"] & family["FDR significant (serial)"]).sum())
        flagged = int(family["FDR significant"].sum())
        displayed = set(df["keyword"])
        trends = family[family["Keyword"].isin(displayed)].copy()
        p_columns = [
            "Raw p-value",
            "Adjusted p-value",
            "Serial-corrected p-value",
            "Adjusted p-value (serial)",
        ]
        trends[p_columns] = trends[p_columns].round(4)
        trends = trends.sort_values("Sen slope")
        st.markdown("#### Nonparametric trend test")
        st.caption(
            "Mann-Kendall and Sen slopes use zero-filled annual series. Benjamini-Hochberg "
            f"adjustment controls the false-discovery rate across all {len(family)} keywords "
            f"with at least {MIN_KEYWORD_OCCURRENCES} occurrences; the table shows the "
            f"{len(trends)} terms charted above, with their family-adjusted values. "
            f"Of the {flagged} keywords significant under the independence assumption, "
            f"{survives} stay significant once the Hamed-Rao correction for autocorrelated "
            "years is applied — treat the difference as the cost of that assumption."
        )
        st.dataframe(trends, hide_index=True, width="stretch")


def _first_appearance(kw_year: pd.DataFrame) -> None:
    st.markdown("**New vocabulary vs. foundational**")
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= MIN_KEYWORD_OCCURRENCES]
    if eligible.empty:
        st.info(f"No term has at least {MIN_KEYWORD_OCCURRENCES} occurrences.")
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
            "first_year": "Year of the first mention in the corpus",
            "count": "Total mentions in the period",
        },
    )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.4)")),
        hovertemplate="<b>%{hovertext}</b><br>1a mention: %{x}<br>%{y:,} total mentions<extra></extra>",
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
        caption="Terms in the upper-right are recent vocabulary that has already gained volume; "
        "high-count terms on the left are foundational vocabulary. Hover to inspect each term. "
        "Fixed labels mark only the most extreme combinations of volume and recent debut to avoid "
        "overplotting.",
    )


def _qualis_match_data(
    articles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Returns `(match_df, with_estrato, totals_by_estrato, estrato_order)`; `with_estrato` is `articles_df` with an added `estrato` column (each article's vein's CAPES/Qualis tier)."""
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
    st.subheader("🎓 CAPES/Qualis classification")
    n_classified = int((match_df["estrato"] != NOT_CLASSIFIED).sum())
    a1_venues = match_df.loc[match_df["estrato"] == "A1", "venue"]
    pct_a1 = (len(a1_articles_df) / len(articles_df) * 100) if len(articles_df) else 0.0

    metric_row(
        [
            ("📚 Periodicals classified", f"{n_classified}/{len(match_df)}", None),
            ("🥇 A1 journals", f"{len(a1_venues)}", None),
            (
                "📄 Articles in A1 journals",
                f"{len(a1_articles_df):,}",
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


def _qualis_totals_chart(totals_by_estrato: pd.Series, estrato_order: list[str]) -> None:
    st.subheader("Total publications by classification")
    fig = px.bar(
        x=totals_by_estrato.index,
        y=totals_by_estrato.to_numpy(),
        category_orders={"x": estrato_order},
        color=totals_by_estrato.index,
        color_discrete_map=venue_color_map(estrato_order, others_label=NOT_CLASSIFIED),
        labels={"x": "Classification", "y": "Articles"},
    )
    fig.update_layout(showlegend=False)
    render_chart(
        fig,
        caption="Total corpus articles by CAPES/Qualis classification, ordered from best "
        f'("A1") to worst, with "{NOT_CLASSIFIED}" last.',
    )


def _qualis_cumulative_chart(with_estrato: pd.DataFrame, estrato_order: list[str]) -> None:
    st.subheader("Cumulative by classification")
    cum_by_estrato = cumulative_by_category(with_estrato, "estrato", top_n=10)
    if cum_by_estrato.empty:
        st.info("No valid years for the cumulative by classification.")
        return

    fig = stacked_area(
        cum_by_estrato,
        x="year",
        y="cumulative",
        color="estrato",
        color_map=venue_color_map(estrato_order, others_label=NOT_CLASSIFIED),
        category_orders={"estrato": estrato_order},
        title="Articles cumulative by CAPES/Qualis classification",
    )
    fig.update_traces(
        hovertemplate="Year %{x}<br>%{data.name}: %{y:,.0f} cumulative articles<extra></extra>"
    )
    fig.update_layout(xaxis_title="Year of publication", yaxis_title="Cumulative articles")
    render_chart(
        fig,
        caption="Cumulative corpus composition by CAPES/Qualis stratum (area"
        f"{QUALIS_AREA}). Journals without a reliable match remain in "
        f'"{NOT_CLASSIFIED}" and are not mixed into a real stratum.',
    )


def _qualis_a1_a3_combined(with_estrato: pd.DataFrame) -> None:
    st.subheader("A1–A3 journals — cumulative")
    combined_df = with_estrato[with_estrato["estrato"].isin(("A1", "A2", "A3"))]
    if combined_df.empty:
        st.info("No journal articles classified A1, A2 or A3 in this layer/filter.")
        return

    tier_palette = venue_color_map(["A1", "A2", "A3"])

    col_volume, col_ranking = st.columns(2)
    with col_volume:
        years_df = combined_df.copy()
        years_df["year"] = valid_years(years_df)
        years_df = years_df.dropna(subset=["year"]).astype({"year": int})
        by_year = source_counts_by(years_df, "year").sort_values("year")
        if by_year.empty:
            st.info("No valid years for this graph.")
        else:
            fig = source_bars(
                by_year,
                "year",
                total_line=True,
                title="Annual volume of articles in A1–A3 journals, by source",
            )
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Year of publication",
                yaxis_title="Number of articles",
            )
            render_chart(
                fig,
                caption="Annual volume of articles published in A1, A2, or A3 journals, by source.",
            )
    with col_ranking:
        top_combined = (
            source_counts_by(combined_df, "venue").sort_values("total", ascending=False).head(15)
        )
        fig = source_topn_hbar(
            top_combined,
            "venue",
            title="Top 15 A1–A3 journals by article count and source",
            x_title="Number of articles",
            y_title="Journal",
        )
        render_chart(
            fig,
            caption="Most frequent journals in the corpus across the A1, A2, and A3 tiers, "
            "colored by source.",
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
            title="Top 15 A1–A3 journals by article count and CAPES/Qualis classification",
            x_title="Number of articles",
            y_title="Journal",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
        render_chart(
            fig,
            caption="The same most published journals of the A1-A3 set, now colored by "
            "CAPES/Qualis classification itself — shows which category each journal of the ranking "
            "pertence.",
        )
    with col_source_tier:
        if "source" not in combined_df.columns:
            st.info("Source column not available in this layer.")
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
                title="Distribution of A1/A2/A3 by source (IEEE vs. Elsevier)",
                labels={"source": "Source", "count": "Articles", "estrato": "Classification"},
            )
            fig.update_traces(
                hovertemplate="<b>%{data.name}</b><br>%{x}: %{y:,} articles<extra></extra>"
            )
            fig.update_layout(
                xaxis_title="Source",
                yaxis_title="Number of articles",
                legend_title_text="Classification",
            )
            render_chart(
                fig,
                caption="Distribution of A1/A2/A3 articles by source (IEEE vs. Elsevier).",
            )


def _bradford_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🎯 Bradford Dispersal Zones")
    st.caption(
        "Bradford's Law divides the journals into 3 concentric zones of equal productivity (~1/3 of the articles each). "
        "The theoretical proportion of the number of journals in each zone follows approximately 1 : k : k2."
    )
    from lake_research_map.dashboard.analytics import bradford_zones

    res = bradford_zones(articles_df)
    zone_sum = res.get("zone_summary")
    if zone_sum is None or zone_sum.empty:
        st.info("Insufficient data for Bradford analysis.")
        return

    metric_row(
        [
            (
                "🎯 Core journals (Zone 1)",
                f"{int(zone_sum.iloc[0]['venues'])}",
                f"{int(zone_sum.iloc[0]['articles']):,} articles",
            ),
            (
                "📚 Journals in Zone 2",
                f"{int(zone_sum.iloc[1]['venues'])}",
                f"{int(zone_sum.iloc[1]['articles']):,} articles",
            ),
            (
                "🌐 Journals in Zone 3",
                f"{int(zone_sum.iloc[2]['venues'])}",
                f"{int(zone_sum.iloc[2]['articles']):,} articles",
            ),
            (
                "📐 Multiplicador de Bradford (k)",
                f"{res.get('multiplier_mean', 1.0):.2f}",
                "Geometric dispersion rate",
            ),
        ]
    )

    fig = px.bar(
        zone_sum,
        x="zone",
        y="venues",
        text="venues",
        title="Number of journals needed to produce 1/3 of the corpus in each zone",
        labels={
            "zone": "Bradford Zone (1 = Nucleus, 3 = Periphery)",
            "venues": "Number of periodicals",
        },
        color="zone",
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    render_chart(
        fig,
        caption="Zone 1 (nucleus) concentrates the mandatory reference journals for planning of data. "
        "The zones 2 and 3 reveal the wide dispersion in general vehicles.",
    )


def _semantic_venues_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🗺️ Semantic positioning of venues")
    st.caption(
        "Positioning of journals in the vectorial space from the centroid of articles "
        "allows viewing the thematic overlap of vehicles independently "
        "of their commercial publisher (IEEE vs. Elsevier)."
    )
    signals = loaders.semantics()
    if signals.empty or "map_x" not in signals.columns or "map_y" not in signals.columns:
        st.info("Semantic signals are unavailable. Run `--stage semantic`.")
        return

    scoped = loaders.with_semantics(articles_df)
    valid = scoped.dropna(subset=["map_x", "map_y", "venue"]).copy()
    if len(valid) < 10:
        st.info("Insufficient articles with semantic data and associated periodicals.")
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
        st.info("No periodicals with sufficient volume to calculate stable centroids.")
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
        title="Semantic centroids of venues (minimum 3 corpus articles)",
        labels={
            "source": "Predominant source",
            "articles": "Articles",
            "margin": "Mean Margin",
        },
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(
        xaxis=dict(title="Dimension 1 (t-SNE)", showticklabels=False),
        yaxis=dict(title="Dimension 2 (t-SNE)", showticklabels=False),
    )
    render_chart(
        fig,
        caption="Close journals publish articles with highly convergent vocabulary and themes.",
    )


def _conceptual_atypicality_tab(articles_df: pd.DataFrame) -> None:
    st.subheader("Unusual combinations of keywords")
    st.caption(
        "Exploratory diagnosis of co-occurrences in relation to the marginal independence of keywords. "
        "It describes rare combinations in this corpus and its association with citations; it does not reproduce the model "
        "null by pairs of periodicals by Uzzi et al."
    )
    res = conceptual_atypicality_analysis(articles_df)
    if not res.get("valid"):
        st.info("Insufficient keywords for null co-occurrence modeling.")
        return

    metric_row(
        [
            (
                "Taxa top 5% no grupo incomum",
                f"{res['hit_rate_high_atypical']:.1f}%",
                f"Baseline: {res['hit_rate_baseline']:.1f}%",
            ),
            (
                "⭐ Top 5% citation threshold",
                f"≥ {res['cite_p95_threshold']} citations",
                "Articles in the 95th percentile",
            ),
            (
                "Descriptive criterion",
                "Co-occurrence",
                "Expected for Independence",
            ),
            (
                "📊 Rare pairs mapped",
                f"{len(res['atypical_pairs'])} combinations",
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
            "median_z": "Conventionality (median Z)",
            "min_z": "Atypicality (Minimum Z)",
            "is_hit": "Top 5% by citations?",
        },
        title="Dispersion: Conventionality vs. Extreme Atypicality by Article",
    )
    fig.update_layout(
        xaxis_title="Conventionality (median Z)", yaxis_title="Atypicality (Minimum Z)"
    )
    render_chart(
        fig,
        caption="Articles in the right lower quadrant (high conventionality and minimum negative Z) "
        "incorporate innovative combinations on mature theoretical ground.",
    )

    st.markdown("##### Main Atypical Conceptual Pairs Identified in Corpus")
    st.dataframe(res["atypical_pairs"], hide_index=True, width="stretch")


def _structural_breaks_tab(articles_df: pd.DataFrame) -> None:
    st.subheader("Structural Crack Detection and Inflection Points (Changepoints)")
    st.caption(
        "It applies the Chow test and minimization of the sum of the residue squares to detect the historical moment "
        "in which the average rate of publication suffered a statistically significant regime transition (p < 0.05)."
    )
    if "year" not in articles_df.columns:
        st.info("Year of publication unavailable.")
        return

    counts = articles_df["year"].dropna().value_counts().sort_index()
    valid_counts = counts[(counts.index >= 2000) & (counts.index <= 2025)]
    if len(valid_counts) < 6:
        st.info("Insufficient time series for breaking detection.")
        return

    res = detect_structural_breaks(valid_counts)
    if not res.get("has_break"):
        st.info("No abrupt structural breakdown identified in the analyzed series.")
        return

    metric_row(
        [
            ("📅 Year of Breaking / Inflection", f"{res['break_year']}", "Regime transition"),
            (
                "📈 Relative Production Jump",
                f"+{res['relative_jump_pct']:.1f}%",
                f"{res['pre_mean']} → {res['post_mean']} articles/year",
            ),
            ("🧪 Chow F statistic", f"{res['f_stat']:.2f}", f"p-value: {res['p_value']:.4f}"),
            ("📏 Significance", "Statistically significant", "p < 0.05"),
        ]
    )

    break_yr = res["break_year"]
    years = valid_counts.index.to_numpy()
    vals = valid_counts.to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=vals, name="Publications / Year", marker_color="#2a78d6"))
    # Pre-break mean
    fig.add_trace(
        go.Scatter(
            x=[years[0], break_yr],
            y=[res["pre_mean"], res["pre_mean"]],
            mode="lines",
            name=f"Regime 1 ({res['pre_mean']:.1f}/year)",
            line=dict(color="#f39c12", width=3, dash="dash"),
        )
    )
    # Post-break mean
    fig.add_trace(
        go.Scatter(
            x=[break_yr, years[-1]],
            y=[res["post_mean"], res["post_mean"]],
            mode="lines",
            name=f"Regime 2 ({res['post_mean']:.1f}/year)",
            line=dict(color="#27ae60", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Structural break in the publication time series (break year: {break_yr})",
        xaxis_title="Year",
        yaxis_title="Articles published",
    )
    render_chart(
        fig,
        caption=f"The regime transition in {break_yr} reflects accelerated scientific production in the field.",
    )
