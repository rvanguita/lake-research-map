"""Topics and venues: where the corpus publishes and how its themes evolved."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard import loaders
from apps.dashboard.analytics.counts import explode_keywords
from apps.dashboard.analytics.vocabulary import (
    conceptual_atypicality_analysis,
    dynamic_topic_ctfidf,
    keyword_trend_frame,
    zipf_law_analysis,
)
from apps.dashboard.charts import stacked_area, topn_hbar
from apps.dashboard.components import (
    article_table,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from apps.dashboard.forecasting import detect_structural_breaks
from apps.dashboard.theme import (
    CATEGORICAL_PALETTE,
    FLAG_COLORS,
    TREND_DOWN_COLOR,
    TREND_UP_COLOR,
    theme_tokens,
)

TOP_KEYWORDS_TREND = 12
TREND_MIN_YEAR = 2010
MIN_KEYWORD_OCCURRENCES = 15


def render() -> None:
    page_header(
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
            venue_view = (
                st.segmented_control(
                    "Venue analysis",
                    options=["Ranking and impact", "Bradford zones", "Semantics"],
                    default="Ranking and impact",
                    key="topics_venue_view",
                )
                or "Ranking and impact"
            )  # clicking the selected option off returns None
            # The CAPES/Qualis views live on the Production and venues page
            # (page ownership): this tab ranks venues, it does not grade them.
            if venue_view == "Ranking and impact":
                _top_venues(articles_df)
            elif require_columns(articles_df, ["venue"]) and articles_df["venue"].notna().any():
                if venue_view == "Bradford zones":
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
                kw_year = keyword_trend_frame(articles_df, min_year=TREND_MIN_YEAR)
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
                        _rising_falling()
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
            ("📝 Total Occurrence", f"{zipf_res['total_tokens']:,} words", None),
        ]
    )

    plot_df = pd.DataFrame(
        {
            "rank": zipf_res["ranks"],
            "frequency": zipf_res["frequencies"],
            "fit": zipf_res["expected_zipf"],
            "term": zipf_res["words"],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["rank"],
            y=plot_df["frequency"],
            mode="markers",
            name="Observed frequency",
            text=plot_df["term"],
            marker=dict(size=6, color=CATEGORICAL_PALETTE[0], opacity=0.7),
            hovertemplate="<b>%{text}</b><br>Rank: %{x}<br>Frequency: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["rank"],
            y=plot_df["fit"],
            mode="lines",
            name=f"Zipf regression (γ = {zipf_res['gamma']:.2f})",
            line=dict(color=CATEGORICAL_PALETTE[1], width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis=dict(type="log", title="Term rank (log r)"),
        yaxis=dict(type="log", title="Frequency of occurrence (log f)"),
        title="Vocabulary rank-frequency distribution (log-log scale)",
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


def _rising_falling() -> None:
    st.markdown("**Rising vs. declining terms**")
    result = loaders.keyword_rising_falling(
        loaders.filter_signature(), TREND_MIN_YEAR, MIN_KEYWORD_OCCURRENCES
    )
    df = result["slopes"] if result is not None else None
    if df is None or df.empty:
        st.info(
            f"No term has at least {MIN_KEYWORD_OCCURRENCES} occurrences for trend calculation."
        )
        return

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

    if result["family_size"]:
        trends = result["tests"]
        st.markdown("#### Nonparametric trend test")
        st.caption(
            "Mann-Kendall and Sen slopes use zero-filled annual series. Benjamini-Hochberg "
            f"adjustment controls the false-discovery rate across all {result['family_size']} "
            f"keywords with at least {MIN_KEYWORD_OCCURRENCES} occurrences; the table shows the "
            f"{len(trends)} terms charted above, with their family-adjusted values. "
            f"Of the {result['flagged']} keywords significant under the independence assumption, "
            f"{result['survives']} stay significant once the Hamed-Rao correction for "
            "autocorrelated years is applied — treat the difference as the cost of that assumption."
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
        marker=dict(size=10, line=dict(width=1, color=theme_tokens()["marker_outline"])),
        hovertemplate="<b>%{hovertext}</b><br>First mention: %{x}<br>%{y:,} total mentions<extra></extra>",
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
            arrowcolor=theme_tokens()["annotation_arrow"],
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


def _bradford_analysis(articles_df: pd.DataFrame) -> None:
    st.subheader("🎯 Bradford Dispersal Zones")
    st.caption(
        "Bradford's Law divides the journals into 3 concentric zones of equal productivity (~1/3 of the articles each). "
        "The theoretical proportion of the number of journals in each zone follows approximately 1 : k : k2."
    )
    from apps.dashboard.analytics.bibliometrics import bradford_zones

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
                "📐 Bradford multiplier (k)",
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
        color_discrete_map=FLAG_COLORS,
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

    # The two regime means used to be drawn as dashed horizontal traces over
    # the bars; the chart contract forbids guide lines, so each bar takes its
    # regime's color instead, and the means move to the legend, the hover and
    # the caption. `break_year` is the first year of the second regime.
    pre = years < break_yr
    regimes = (
        (pre, "Regime 1", f"before {break_yr}", res["pre_mean"]),
        (~pre, "Regime 2", f"{break_yr} on", res["post_mean"]),
    )
    fig = go.Figure()
    for slot, (in_regime, regime, span, mean) in enumerate(regimes):
        fig.add_trace(
            go.Bar(
                x=years[in_regime],
                y=vals[in_regime],
                name=f"{regime} ({span}; mean {mean:.1f}/year)",
                marker_color=CATEGORICAL_PALETTE[slot],
                hovertemplate=(
                    f"Year %{{x}}<br>%{{y:,}} articles<br>{regime} mean: {mean:.1f}/year"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        # The two regime traces cover disjoint years, so the default "group" mode
        # would halve every bar and shift it off its year tick.
        barmode="overlay",
        title=f"Structural break in the publication time series (break year: {break_yr})",
        xaxis_title="Year",
        yaxis_title="Articles published",
    )
    render_chart(
        fig,
        caption=f"Bars are colored by regime: before {break_yr} the mean was "
        f"{res['pre_mean']:.1f} articles/year, from {break_yr} on {res['post_mean']:.1f}. "
        f"The regime transition in {break_yr} reflects accelerated scientific production "
        "in the field.",
    )
