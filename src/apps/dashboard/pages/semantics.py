"""🧭 Semantics & Relevance — what the corpus really contains, according to the embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard import loaders
from apps.dashboard.analytics.themes import (
    DRIFT_MIN_YEAR,
    SMOOTHING_WINDOWS,
    THEME_METRICS,
    horizon_options,
    theme_evolution,
    theme_sizes,
    theme_years,
)
from apps.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from apps.dashboard.theme import (
    CATEGORICAL_PALETTE,
    TREND_DOWN_COLOR,
    theme_tokens,
)
from transform.screening_calibration import (
    calibrate_screening_threshold,
    generate_stratified_screening_sample,
    resolve_review_consensus,
    reviewer_agreement,
    validate_review_labels,
)

LOW_RELEVANCE_PERCENTILE = 10
TOP_REVIEW_ROWS = 40

# How the semantic map can be colored. The theme is the default, but a reviewer
# screening a corpus wants to ask the same picture different questions -- where
# the off-topic mass sits, whether a region is recent or old, which publisher
# indexed it.
MAP_COLOR_OPTIONS = {
    "Theme": "theme_label",
    "Relevance": "relevance_margin",
    "Year": "year",
    "Source": "source_display",
}


def render() -> None:
    page_header(
        "Screening and discovery",
        "Relevance screening, automatically discovered themes, and near-duplicates — all "
        "derived from abstract embeddings.",
    )

    signals = loaders.semantics()
    if signals.empty:
        st.info(
            "The `semantic` stage has not run yet — run "
            "`uv run lake-research-map --stage semantic`. It depends on `gold` and `embed`."
        )
        return

    articles_df = loaders.require_articles()
    scoped = loaders.with_semantics(articles_df)
    if not require_columns(scoped, ["relevance_score"]):
        return
    scored = scoped.dropna(subset=["relevance_score"])

    hero_banner(
        "Why this page exists",
        "The search that generated this corpus (<i>distribution system planning</i>) is ambiguous: it can refer "
        "to both <b>electric power distribution</b> and <b>logistics distribution</b>. Each article receives a "
        "scope-proximity score calculated from its abstract embedding. Relevance screening is a methodological "
        "stage of a systematic review, not an implementation detail.",
    )

    tab_triagem, tab_space, tab_isolation, tab_dupes = st.tabs(
        [
            "Relevance screening",
            "Semantic space and themes",
            "Semantic isolation",
            "Near-duplicates",
        ],
        on_change="rerun",
        key="semantics_primary_tab",
    )

    if tab_triagem.open:
        with tab_triagem:
            _relevance_screening(scored)
    elif tab_space.open:
        with tab_space:
            _semantic_map(scored)
            _themes(scored)
    elif tab_isolation.open:
        with tab_isolation:
            _semantic_novelty_panel(scored)
    elif tab_dupes.open:
        with tab_dupes:
            _duplicates()


def _relevance_screening(scored: pd.DataFrame) -> None:
    # The margin needs `offtopic_score`, written by the contrastive anchor; a
    # database whose last `semantic` run predates it still gets the old view.
    has_margin = "relevance_margin" in scored.columns and scored["relevance_margin"].notna().any()
    if not has_margin:
        _legacy_relevance_screening(scored)
        return

    st.subheader("Relevant margin distribution")
    margin = scored["relevance_margin"]
    low = scored[margin < 0]

    # Articles without an abstract have a title-only vector -- their score
    # comes from much weaker signal and should not influence percentile stats.
    has_abstract_col = "has_abstract" in scored.columns
    title_only = (
        scored[~scored["has_abstract"].astype(bool)] if has_abstract_col else scored.iloc[0:0]
    )
    stats_base = scored[scored["has_abstract"].astype(bool)] if has_abstract_col else scored
    stats_margin = stats_base["relevance_margin"] if not stats_base.empty else margin

    metrics = [
        ("📄 Articles with score", f"{len(scored):,}", None),
        ("📊 Median margin", f"{stats_margin.median():+.3f}", None),
        (
            "🚩 Out of scope (margin < 0)",
            f"{len(low):,}",
            f"{100 * len(low) / len(scored):.1f}% of the corpus",
        ),
    ]
    if len(title_only) > 0:
        metrics.append(
            (
                "📝 Title-only articles",
                f"{len(title_only):,}",
                "excluded from margin statistics",
            )
        )
    metric_row(metrics)

    fig = px.histogram(
        scored,
        x="relevance_margin",
        nbins=60,
        title="How much each article depends on the theme of the review, and not on logistics",
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    fig.update_layout(
        xaxis_title="Margin (relevance − proximity to logistics)",
        yaxis_title="Number of articles",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="Each abstract is compared with **two** anchors: the review topic and the "
        "logistics interpretation of the same search phrase. The margin is their difference, "
        "and **zero is the cutoff**: articles to its left are closer to supply-chain logistics "
        "than to electric-power distribution networks. With one anchor, the two distributions "
        "overlapped, and any percentile cutoff also discarded in-scope work. Use the sidebar "
        "filter to apply this cutoff to **all** charts.",
    )

    st.divider()
    st.subheader(f"Out of scope — {len(low):,} articles for manual review")
    st.caption(
        "Ordered from the most negative margin upward. The score supports screening; it is not "
        "a verdict, so review each article before excluding it."
    )
    review = low.sort_values("relevance_margin").head(TOP_REVIEW_ROWS).copy()
    for column in ("relevance_margin", "relevance_score"):
        review[column] = review[column].round(3)
    display_cols = [
        "relevance_margin",
        "relevance_score",
        "theme_label",
        "title",
        "year",
        "venue",
        "source",
        "doi",
    ]
    article_table(
        review,
        display_cols,
        download_key="out_of_scope",
    )

    st.divider()
    st.subheader("🤖 Active-learning-assisted screening (uncertainty sampling)")
    st.caption(
        "Articles whose contrastive margin sits closest to zero (|Δ| ≈ 0) are the cases of "
        "maximum ambiguity. Prioritizing manual inspection of those accelerates systematic "
        "screening refinement at the lowest human reading effort."
    )
    uncertain = scored[scored["relevance_margin"].notna()].copy()
    uncertain["abs_margin"] = uncertain["relevance_margin"].abs()
    uncertain = uncertain.sort_values("abs_margin").head(25)
    for col in ("relevance_margin", "relevance_score"):
        uncertain[col] = uncertain[col].round(3)
    article_table(
        uncertain,
        [
            "relevance_margin",
            "relevance_score",
            "theme_label",
            "title",
            "year",
            "venue",
            "source",
            "doi",
        ],
        download_key="active_learning_uncertainty",
    )

    _screening_calibration_panel(scored)


def _screening_calibration_panel(scored: pd.DataFrame) -> None:
    """Collect reviewed CSVs in memory and report held-out threshold evidence."""
    st.divider()
    st.subheader("Calibration with human review")
    st.caption(
        "The uncertainty queue prioritizes reading; the stratified sample below serves a "
        "different question: to estimate the threshold's performance across the whole margin "
        "range. Nothing on this page records labels or applies exclusions automatically."
    )

    sample = generate_stratified_screening_sample(scored, n_samples=100, seed=42)
    st.download_button(
        "Transfer stratified sample to review",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="screening_review_sample.csv",
        mime="text/csv",
        key="screening_review_sample",
        width="stretch",
    )
    uploaded = st.file_uploader(
        "Send revised decisions (CSV long-form)",
        type=["csv"],
        accept_multiple_files=True,
        key="screening_review_files",
        help=(
            "Mandatory fields: doi and manual_label. reviewer and protocol_version are recommended."
        ),
    )
    if not uploaded:
        st.info(
            "Send independent decisions of reviewers to calculate agreement and validate "
            "a candidate threshold."
        )
        return

    frames: list[pd.DataFrame] = []
    for file in uploaded:
        try:
            frames.append(pd.read_csv(file))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read `{file.name}`: {exc}")
    if not frames:
        return

    labels, issues = validate_review_labels(
        pd.concat(frames, ignore_index=True),
        known_dois=set(scored["doi"].dropna().astype(str)),
    )
    if not issues.empty:
        n_errors = int(issues["severity"].eq("error").sum())
        n_warnings = int(issues["severity"].eq("warning").sum())
        message = f"File audit: {n_errors} error(s) and {n_warnings} warning(s)."
        st.error(message) if n_errors else st.warning(message)
        st.dataframe(issues, hide_index=True, width="stretch")
    if labels.empty:
        return

    agreement = reviewer_agreement(labels)
    if not agreement.empty:
        st.markdown("**Concordance between reviewers**")
        st.dataframe(agreement.round(3), hide_index=True, width="stretch")
        if agreement["status"].ne("ok").any():
            st.caption(
                "κ is only reported with at least 20 binary shared decisions and presence "
                "of both classes; the remaining pairs stay flagged as insufficient support."
            )

    resolved = resolve_review_consensus(labels)
    n_resolved = int(resolved["resolved"].sum())
    n_disagreement = int(resolved["resolution"].eq("disagreement").sum())
    n_single = int(resolved["resolution"].eq("single_reviewer").sum())
    metric_row(
        [
            ("Decisions resolved", f"{n_resolved:,}", None),
            ("Pending divergences", f"{n_disagreement:,}", None),
            ("Labels of a reviewer", f"{n_single:,}", "provisional evidence"),
        ]
    )

    calibration = calibrate_screening_threshold(resolved, scored, seed=42)
    if not calibration["valid"]:
        st.warning(
            "There is still not enough support for holdout validation: it needs at least 40 "
            "resolved subjects, with at least 10 inclusions and 10 exclusions."
        )
        return

    threshold = float(calibration["threshold"])
    metrics = calibration["metrics"]
    intervals = calibration["confidence_intervals"]

    def _metric_with_ci(name: str) -> str:
        low_ci, high_ci = intervals[name]
        return f"{metrics[name]:.1%} (IC95% {low_ci:.1%}–{high_ci:.1%})"

    st.success(
        f"Candidate threshold: margin ≥ {threshold:+.3f}. Fitted on "
        f"{calibration['n_calibration']} decisions and evaluated once on "
        f"{calibration['n_holdout']} holdout decisions."
    )
    metric_row(
        [
            ("Sensitivity", _metric_with_ci("recall"), f"FN: {metrics['fn']}"),
            ("Specificity", _metric_with_ci("specificity"), None),
            ("Precision", _metric_with_ci("precision"), None),
            ("F2", _metric_with_ci("f2"), "favours sensitivity"),
            ("Load reduction", _metric_with_ci("workload_reduction"), None),
        ]
    )

    curve = calibration["curve"]
    fig = px.line(
        curve.sort_values("recall"),
        x="recall",
        y="precision",
        markers=True,
        hover_data={"threshold": ":+.3f", "specificity": ":.1%"},
        labels={
            "recall": "Sensitivity",
            "precision": "Precision",
            "threshold": "Threshold",
            "specificity": "Specificity",
        },
        title="Precision and sensitivity in the holdout set",
    )
    fig.add_scatter(
        x=[metrics["recall"]],
        y=[metrics["precision"]],
        mode="markers",
        marker={"size": 13, "symbol": "diamond", "color": TREND_DOWN_COLOR},
        name="Candidate threshold",
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    render_chart(
        fig,
        caption="The highlighted point is validation evidence, not authorization to exclude. "
        "Broad intervals indicate the need to expand the human review.",
    )


def _legacy_relevance_screening(scored: pd.DataFrame) -> None:
    """The percentile view, for a database that predates the contrastive anchor."""
    st.subheader("Relevancy score distribution")
    st.info(
        "This layer was generated before the contrastive anchor — `--stage semantic` to use the "
        "margin, whose zero cutoff replaces the percentile below."
    )
    threshold = float(scored["relevance_score"].quantile(LOW_RELEVANCE_PERCENTILE / 100))
    low = scored[scored["relevance_score"] < threshold]

    metric_row(
        [
            ("📄 Articles with score", f"{len(scored):,}", None),
            ("📉 Median score", f"{scored['relevance_score'].median():.3f}", None),
            (
                f"🚩 Below percentile {LOW_RELEVANCE_PERCENTILE}",
                f"{len(low):,}",
                f"cut at {threshold:.3f}",
            ),
        ]
    )

    # The percentile cut used to be a dashed vertical line; the chart contract
    # forbids guide lines, so the bars on either side of it take their own
    # color and the cut value moves to the legend and the caption. px shares
    # one set of bins across the color groups, so the split stays aligned.
    below_label = f"Below percentile {LOW_RELEVANCE_PERCENTILE} (< {threshold:.3f})"
    above_label = f"At or above percentile {LOW_RELEVANCE_PERCENTILE}"
    histogram_df = scored.assign(
        screening=np.where(scored["relevance_score"] < threshold, below_label, above_label)
    )
    fig = px.histogram(
        histogram_df,
        x="relevance_score",
        color="screening",
        nbins=60,
        title="How close to the theme of the review is each article",
        labels={
            "relevance_score": "Relevance score (cosine)",
            "count": "Articles",
            "screening": "Percentile cut",
        },
        category_orders={"screening": [below_label, above_label]},
        color_discrete_map={below_label: TREND_DOWN_COLOR, above_label: CATEGORICAL_PALETTE[0]},
    )
    fig.update_layout(
        xaxis_title="Relevance score (1.0 = identical to the theme anchor)",
        yaxis_title="Number of articles",
    )
    render_chart(
        fig,
        caption="The score is the cosine between the abstract and an anchor text describing "
        "the scope of the review. The left tail concentrates the search's false positives; "
        f"bars below the percentile-{LOW_RELEVANCE_PERCENTILE} cut ({threshold:.3f}) are "
        "colored separately.",
    )

    st.divider()
    st.subheader(f"Low-relevance tail — {len(low):,} articles for manual review")
    st.caption(
        "Ordered from the least relevant to the most relevant. A low score is a hint, "
        "not a verdict: review before discarding."
    )
    review = low.sort_values("relevance_score").head(TOP_REVIEW_ROWS).copy()
    review["relevance_score"] = review["relevance_score"].round(3)
    article_table(
        review,
        ["relevance_score", "theme_label", "title", "year", "venue", "source", "doi"],
        download_key="low_relevance",
    )


def _semantic_map(scored: pd.DataFrame) -> None:
    st.subheader("Semantic map of the corpus")
    if not require_columns(scored, ["map_x", "map_y", "theme_label"]):
        return

    plot_df = scored.dropna(subset=["map_x", "map_y"]).copy()

    # Prepare descriptive, readable attributes for tooltips and legends.
    plot_df["title_display"] = plot_df["title"].fillna("Untitled")
    plot_df["title_hover"] = plot_df["title_display"].apply(
        lambda t: (
            "<br>".join([t[i : i + 65] for i in range(0, min(len(t), 195), 65)])
            + ("..." if len(t) > 195 else "")
        )
    )
    plot_df["venue_display"] = plot_df["venue"].fillna("Venue not reported")
    plot_df["year_display"] = plot_df["year"].fillna("—").astype(str)
    source_map = {"ieee": "IEEE Xplore", "elsevier": "ScienceDirect (Elsevier)"}
    plot_df["source_display"] = plot_df["source"].map(source_map).fillna(plot_df["source"])
    plot_df["theme_display"] = plot_df["theme_label"].fillna("No theme assigned")

    if "relevance_margin" in plot_df.columns:
        plot_df["status_display"] = np.where(
            plot_df["relevance_margin"] >= 0,
            "🟢 In-Scope (Energia / Relevante)",
            "🔴 Off-Top (Logistic / General)",
        )
    else:
        plot_df["status_display"] = "—"

    # Fall back to relevance_score if relevance_margin is unavailable.
    available_map = dict(MAP_COLOR_OPTIONS)
    if "relevance_margin" not in plot_df.columns and "relevance_score" in plot_df.columns:
        available_map["Relevance"] = "relevance_score"

    available = {
        label: column
        for label, column in available_map.items()
        if column in plot_df.columns and plot_df[column].notna().any()
    }

    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.3, 2.5, 2.5, 2.7])
    with ctrl_col0:
        alternative_projections = loaders.alternative_projections()
        projection_options = ["t-SNE", *alternative_projections]
        proj_choice = (
            st.segmented_control(
                "Projection",
                options=projection_options,
                default="t-SNE",
                key="semantic_proj_choice",
                help="Alters the dimensional reduction technique of 384D vectors.",
            )
            or "t-SNE"
        )
    with ctrl_col1:
        choice = (
            st.segmented_control(
                "Color by",
                options=list(available),
                default="Theme",
                key="semantic_map_color",
                help="It changes the chromatic dimension of the points in the vector space.",
            )
            or "Theme"
        )
    with ctrl_col2:
        legend_pos = (
            st.segmented_control(
                "Subtitle position",
                options=["Right side", "Bottom", "Hide"],
                default="Right side",
                key="semantic_map_legend_pos",
                help="Position the legend for better readability or hide to expand the graph.",
            )
            or "Right side"
        )
    with ctrl_col3:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            show_density = st.checkbox(
                "🌊 Density (KDE)",
                value=False,
                key="semantic_show_density",
                help="It overlaps two-dimensional probability density level curves.",
            )
        with sub_c2:
            show_theme_labels = st.checkbox(
                "🏷️ Theme labels",
                value=True,
                key="semantic_show_theme_labels",
                help="Shows the names of the themes on the median centroids of the clusters in the foreground.",
            )

    if proj_choice in ("PCA 2D", "UMAP"):
        if (
            proj_choice in alternative_projections
            and not alternative_projections[proj_choice].empty
        ):
            alt_df = alternative_projections[proj_choice]
            plot_df = (
                plot_df.drop(columns=["map_x", "map_y"], errors="ignore")
                .merge(alt_df[["doi", "map_x", "map_y"]], on="doi", how="left")
                .dropna(subset=["map_x", "map_y"])
            )

    color_column = available[choice]
    continuous = choice in ("Relevance", "Year")

    fig = px.scatter(
        plot_df,
        x="map_x",
        y="map_y",
        color=color_column,
        custom_data=[
            "title_hover",
            "venue_display",
            "year_display",
            "source_display",
            "theme_display",
            "relevance_score",
            "relevance_margin" if "relevance_margin" in plot_df.columns else "relevance_score",
            "status_display",
        ],
        color_continuous_scale=None if not continuous else ["#e34948", "#eda100", "#1baf7a"],
        color_discrete_sequence=CATEGORICAL_PALETTE,
        opacity=0.75,
    )
    fig.update_traces(
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br> "
            "🏛️ <b>Venue:</b> %{customdata[1]}<br> "
            "📅 <b>Year:</b> %{customdata[2]}  •  🏷️ <b>Source:</b> %{customdata[3]}<br> "
            "🎯 <b>Theme:</b> %{customdata[4]}<br> "
            "📊 <b>Relevance:</b> %{customdata[5]:.3f} (Margin Δ:%{customdata[6]:.3f})<br> "
            "🚦 <b>Screening:</b> %{customdata[7]} "
            "<extra></extra>"
        ),
    )

    if show_density:
        fig.add_trace(
            go.Histogram2dContour(
                x=plot_df["map_x"],
                y=plot_df["map_y"],
                colorscale="Blues",
                reversescale=True,
                showscale=False,
                opacity=0.35,
                contours=dict(coloring="fill", showlabels=False),
                hoverinfo="skip",
            )
        )
        fig.data = (fig.data[-1],) + fig.data[:-1]

    if show_theme_labels:
        _add_theme_labels(fig, plot_df)

    # Structured main title and conceptual axes.
    fig.update_layout(
        title=dict(
            text="Semantic Map of Corpus",
            subtitle=dict(
                text="2D projection of the vector summaries (embeddings) — spatial proximity indicates thematic-conceptual convergence"
            ),
        ),
        xaxis=dict(title="Semantic Axis 1 (Latent Space)", showticklabels=False),
        yaxis=dict(title="Semantic Axis 2 (Latent Space)", showticklabels=False),
    )

    # Dynamic title for the categorical legend.
    if choice == "Theme":
        legend_title_text = (
            "<b>Research theme</b><br><span style='font-size:10px; "
            f"color:{theme_tokens()['muted']};'>Topic clustering</span>"
        )
    elif choice == "Source":
        legend_title_text = "<b>Indexing source</b>"
    else:
        legend_title_text = ""

    # Position the legend or color bar.
    if continuous:
        if choice == "Relevance":
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(
                        text="<b>Margin (Δ)</b><br><span style='font-size:10px;'>Off-topic < 0 < In-scope</span>",
                        side="top",
                    ),
                    tickvals=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    ticktext=["-0.20", "-0.10", "0.00 threshold", "+0.10", "+0.20"],
                )
            )
        elif choice == "Year":
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(text="<b>A Publication </b>", side="top"),
                    dtick=2,
                )
            )
        chart_margin = dict(l=40, r=120, t=95, b=40)
    else:
        if legend_pos == "Right side":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01,
                    title=dict(text=legend_title_text),
                    font=dict(size=11),
                    itemsizing="constant",
                    tracegroupgap=6,
                ),
            )
            chart_margin = dict(l=40, r=300, t=95, b=40)
        elif legend_pos == "Bottom":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.16,
                    xanchor="center",
                    x=0.5,
                    title=dict(text=legend_title_text),
                    font=dict(size=11),
                    itemsizing="constant",
                ),
            )
            chart_margin = dict(l=40, r=40, t=95, b=120)
        else:  # "Hide"
            fig.update_layout(showlegend=False)
            chart_margin = dict(l=40, r=40, t=95, b=40)

    render_chart(
        fig,
        height=650,
        margin=chart_margin,
        caption="t-SNE projection of the abstract embeddings, computed in the **same space** in which "
        "the themes are discovered — that is what makes a point's colour agree with where it falls. "
        "**The axes carry no absolute numerical meaning**: only relative distance between points matters. "
        "The corpus forms a dense continuum of distribution-network planning with **one detached island** "
        "(operational logistics research, a contamination from the word *distribution* in the search); "
        "the themes are slices of that continuum revealed by semantic clustering.",
    )

    _projection_stability(plot_df)


def _projection_stability(plot_df: pd.DataFrame) -> None:
    """Say how much of the map is structure and how much is presentation.

    PRD section 8.3 requires a projection to report neighbourhood preservation
    and a cluster solution to report stability; without them a 2D picture reads
    as stronger evidence than the high-dimensional geometry supports.
    """
    if "theme_label" not in plot_df.columns:
        return
    points = plot_df.dropna(subset=["doi", "map_x", "map_y", "theme_label"])
    points = points.drop_duplicates(subset=["doi"])
    if len(points) < 10:
        return

    result = loaders.semantic_stability(
        tuple(
            (str(row.doi), str(row.theme_label), float(row.map_x), float(row.map_y))
            for row in points.itertuples()
        )
    )
    if not result or not result.get("valid"):
        return

    with st.expander("Projection and cluster stability", expanded=False):
        metric_row(
            [
                (
                    "\U0001f504 Bootstrap ARI (mean)",
                    f"{result['bootstrap_ari_mean']:.2f}",
                    f"worst of {result['n_bootstrap']} resamples: "
                    f"{result['bootstrap_ari_min']:.2f}",
                ),
                (
                    "\U0001f9ed Projection trustworthiness",
                    f"{result['projection_trustworthiness']:.2f}",
                    "Share of 2D neighbours that are also neighbours in the clustering space",
                ),
                (
                    "\U0001f3af Themes compared",
                    f"{result['clusters']}",
                    f"over {len(points):,} embedded abstracts",
                ),
            ]
        )
        neighbors = result.get("neighbors")
        overlap = result.get("knn_overlap")
        continuity = result.get("projection_continuity")
        metric_row(
            [
                (
                    "\U0001f9f2 Projection continuity",
                    "n/a" if continuity is None else f"{continuity:.2f}",
                    "Share of clustering-space neighbours the map keeps",
                ),
                (
                    "\U0001f517 kNN overlap",
                    "n/a" if overlap is None or not np.isfinite(overlap) else f"{overlap:.2f}",
                    f"Of each point's {neighbors} nearest neighbours",
                ),
            ]
        )
        st.caption(
            "ARI near 1 means the same articles group together when the sample and the seed "
            "change; a low value means the theme boundaries are a property of this particular "
            "run. Trustworthiness below roughly 0.9 means the map places points next to each "
            "other that are far apart in the space the themes were built in \u2014 read clusters, "
            "not distances. Trustworthiness alone only penalises neighbours the map invents, so "
            "continuity (neighbours it loses) and the plain kNN overlap are shown beside it: a "
            "projection that tears one real cluster in two scores well on trustworthiness and "
            "badly on continuity. All are measured in that shared PCA space rather than the raw "
            "384-dimensional one, because that is where the clustering actually happened. "
            "Theme numbering is not stable across runs and carries no ontological claim."
        )

        sweep = result.get("k_sweep") or []
        if len(sweep) > 1:
            sweep_df = pd.DataFrame(sweep)
            chosen = result.get("clusters")
            best = sweep_df["silhouette"].max()
            spread = best - sweep_df["silhouette"].min()
            sweep_df = sweep_df.rename(
                columns={
                    "k": "Themes (k)",
                    "silhouette": "Silhouette",
                    "smallest_cluster_share": "Smallest cluster",
                    "rejected_small_cluster": "Rejected (<2%)",
                }
            )
            sweep_df["Silhouette"] = sweep_df["Silhouette"].round(4)
            sweep_df["Smallest cluster"] = (sweep_df["Smallest cluster"] * 100).round(1)
            st.markdown("**How decisive was the choice of k?**")
            st.dataframe(sweep_df, hide_index=True, width="stretch")
            st.caption(
                f"Every candidate the search considered. Silhouette spans only {spread:.3f} "
                f"across the range, so k={chosen} is chosen by the near-tie rule that prefers "
                "the smaller solution rather than by a clear maximum \u2014 which is why the "
                "theme boundaries deserve the stability caveat above, and why a neighbouring "
                "k would be nearly as defensible."
            )


def _add_theme_labels(fig, plot_df: pd.DataFrame) -> None:
    """Write each theme's name on the map as an annotation badge in front of points."""
    if "theme_label" not in plot_df.columns:
        return
    centroids = plot_df.groupby("theme_label")[["map_x", "map_y"]].median().reset_index()
    t = theme_tokens()
    labels = centroids["theme_label"].apply(
        lambda s: "<br>".join(s.split(" · ")) if " · " in s else s
    )
    for (_, row), label in zip(centroids.iterrows(), labels, strict=True):
        fig.add_annotation(
            x=row["map_x"],
            y=row["map_y"],
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=t["chart_annotation"]),
            bgcolor=t["legend_bg"],
            bordercolor=t["legend_border"],
            borderwidth=1,
            borderpad=4,
            opacity=0.92,
        )


def _semantic_novelty_panel(scored: pd.DataFrame) -> None:
    st.subheader("Semantic isolation in embedding space")
    st.caption(
        "Measures mean distance to the nearest $k$ neighbors in the 384-dimensional embedding "
        "space. High values identify documents isolated from the rest of the corpus; isolation "
        "alone does not establish novelty, innovation, or interdisciplinarity."
    )

    nov_df = loaders.semantic_novelty_scores()
    if nov_df.empty:
        st.info("Embedding matrix not available to calculate semantic novelty.")
        return

    merged = pd.merge(scored, nov_df, on="doi", how="inner")
    if merged.empty:
        st.info("No article with corresponding novelty score.")
        return

    nov = merged["novelty_score"]
    p90 = float(nov.quantile(0.90))

    metric_row(
        [
            ("Median isolation", f"{nov.median():.3f}", None),
            ("P90 threshold", f"{p90:.3f}", "10% most isolated"),
            ("Highest isolation", f"{nov.max():.3f}", None),
            ("Total evaluated", f"{len(merged):,}", "384D embeddings"),
        ]
    )

    # The P90 cut used to be a dashed vertical line; the chart contract forbids
    # guide lines, so it lives in the metric card above, the hover and the caption.
    merged = merged.assign(
        isolation_band=np.where(
            merged["novelty_score"] >= p90, "Top 10% most isolated (≥ P90)", "Below P90"
        )
    )
    fig = px.scatter(
        merged,
        x="novelty_score",
        y="relevance_score" if "relevance_score" in merged.columns else "novelty_score",
        color="theme_label" if "theme_label" in merged.columns else None,
        hover_data=["title", "year", "venue", "isolation_band"],
        labels={
            "novelty_score": "Semantic isolation (k-NN distance)",
            "relevance_score": "Thematic Relevance",
            "theme_label": "Theme",
            "isolation_band": "Isolation band",
        },
        color_discrete_sequence=CATEGORICAL_PALETTE,
        title="Semantic isolation and relevance in the corpus",
    )
    fig.update_layout(height=480)
    render_chart(
        fig,
        caption="Points on the right are more distant from their semantic neighbors and deserve "
        f"inspection; those at or beyond the P90 threshold ({p90:.3f}) are the 10% most "
        "isolated (see the hover).",
    )

    st.markdown("#### Semantically more isolated articles")
    top_novel = merged.sort_values("novelty_score", ascending=False).head(20).copy()
    top_novel["novelty_score"] = top_novel["novelty_score"].round(3)
    if "relevance_score" in top_novel.columns:
        top_novel["relevance_score"] = top_novel["relevance_score"].round(3)
    cols = ["novelty_score", "relevance_score", "theme_label", "title", "year", "venue", "doi"]
    display_cols = [c for c in cols if c in top_novel.columns]
    article_table(top_novel, display_cols, download_key="top_semantic_novelty")


def _themes(scored: pd.DataFrame) -> None:
    st.subheader("Automatically discovered themes")
    if not require_columns(scored, ["theme_label"]):
        return

    by_theme = theme_sizes(scored)

    fig = px.bar(
        by_theme,
        x="articles",
        y="theme_label",
        orientation="h",
        color="mean_relevance",
        color_continuous_scale=["#e34948", "#eda100", "#1baf7a"],
        title="Size of each theme and its average proximity to the scope of the review",
        labels={
            "articles": "Number of articles",
            "theme_label": "",
            "mean_relevance": "Relevance",
        },
    )
    fig.update_layout(
        xaxis_title="Number of articles",
        yaxis_title="Theme",
        yaxis=dict(categoryorder="total ascending"),
    )
    render_chart(
        fig,
        caption="Labels come from the terms each group uses **more than the rest of the corpus**. Without that "
        "contrast, every theme would be labeled 'distribution, power, planning'. Color shows mean relevance; "
        "the logistics group has the lowest value, independently confirming the screening score.",
    )

    st.divider()
    st.subheader("Temporal evolution of research themes")
    if "year" not in scored.columns:
        st.info("'year' column not available in this layer.")
        return
    yearly = theme_years(scored)
    if yearly.empty:
        st.info("No valid years for this analysis.")
        return

    min_corpus_year = int(yearly["year"].min())
    max_corpus_year = int(yearly["year"].max())
    horizons = horizon_options(min_corpus_year, max_corpus_year)
    horizon_labels = list(horizons)
    metric_labels = list(THEME_METRICS)
    smoothing_labels = list(SMOOTHING_WINDOWS)

    # Compact interactive controls. Every option list is the key set of the
    # dict that maps it to a value, so a label cannot drift from its meaning.
    c_time, c_metric, c_smooth = st.columns([3, 3, 3])
    with c_time:
        time_choice = (
            st.segmented_control(
                "Time horizon",
                options=horizon_labels,
                default=horizon_labels[0],
                key="theme_evol_horizon",
                help="The modern period avoids artificial oscillations of ancient sparse years.",
            )
            or horizon_labels[0]
        )

    with c_metric:
        metric_choice = (
            st.segmented_control(
                "Metrics",
                options=metric_labels,
                default=metric_labels[0],
                key="theme_evol_metric",
                help="It alters between the relative participation of each theme in the year and the real volume of publications.",
            )
            or metric_labels[0]
        )

    with c_smooth:
        smooth_choice = (
            st.segmented_control(
                "Smoothing",
                options=smoothing_labels,
                default=smoothing_labels[0],
                key="theme_evol_smooth",
                help="It applies centralized moving average to smooth the annual noise and reveal structural trends.",
            )
            or smoothing_labels[0]
        )

    # A stale session value (an option renamed since) falls back to the default.
    start_year = horizons.get(time_choice, horizons[horizon_labels[0]])
    end_year = max_corpus_year
    is_relative = THEME_METRICS.get(metric_choice, True)
    window = SMOOTHING_WINDOWS.get(smooth_choice, SMOOTHING_WINDOWS[smoothing_labels[0]])
    plot_df = theme_evolution(
        yearly,
        start_year=start_year,
        end_year=end_year,
        window=window,
        relative=is_relative,
    )
    if plot_df.empty:
        st.info("No data in the selected period.")
        return
    y_title = "Participation in the year (%)" if is_relative else "Number of published articles"

    fig = px.area(
        plot_df,
        x="year",
        y="value",
        color="theme_label",
        line_shape="spline",
        custom_data=["volume_real", "pct_real"],
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )

    hovertemplate = (
        "<b>%{fullData.name}</b><br> "
        "📅 <b>Year:</b> %{x}<br>"
        + (
            "📊 <b>Participation (softened):</b> %{y:.1f}%<br>"
            if is_relative
            else "📚 <b>Volume (smooth):</b> %{y:.1f} articles<br>"
        )
        + "📚 <b>Real volume of the year:</b> %{customdata[0]:.0f} articles (%{customdata[1]:.1f}%) "
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        title=dict(
            text="Temporal Evolution of Research Themes",
            subtitle=dict(
                text="Scientific attention by theme over the years of publication (softened spline interpolation)"
            ),
        ),
        xaxis=dict(
            title="Year of publication",
            dtick=2 if (end_year - start_year) <= 20 else 5,
        ),
        yaxis=dict(
            title=y_title,
            range=[0, 100] if is_relative else None,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            title=dict(text="<b>Theme</b>"),
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=6,
        ),
        hovermode="x unified",
    )

    chart_caption = (
        "Thematic evolution with a continuous year grid and moving average reduces distortions "
        "from sparse years and shows how scientific attention migrated toward topics such as "
        "electric mobility and distributed storage."
        if is_relative
        else "Absolute volume of articles per theme in each year: reveals the growth of the corpus as a whole "
        "and the accelerated expansion of scientific production in the last two decades."
    )

    render_chart(
        fig,
        height=580,
        margin=dict(l=40, r=300, t=95, b=40),
        caption=chart_caption,
    )

    st.divider()
    st.subheader("🧭 Thematic Drift")
    windows, drift_df = loaders.theme_drift(loaders.filter_signature())
    if drift_df.empty or not windows:
        st.info(
            "Thematic drift needs scored articles spread over at least two publication years "
            "within the current filters."
        )
        return
    window_names = ", ".join(f"{lo}–{hi}" for lo, hi in windows)
    st.caption(
        f"Change of the center of mass of each theme over {len(windows)} chronological windows "
        f"({window_names}). The windows are cut over the filtered articles dated {DRIFT_MIN_YEAR} "
        "or later so that each holds roughly the same number of them: they follow the current "
        "scope rather than fixed eras. The line shows how far each line of research shifted in "
        "the semantic plane."
    )
    if not drift_df.empty:
        fig_drift = px.line(
            drift_df,
            x="map_x",
            y="map_y",
            color="Theme",
            text="Time",
            markers=True,
            title="Trajectory of thematic centroids in two-dimensional space",
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        fig_drift.update_traces(textposition="top center")
        fig_drift.update_layout(
            xaxis=dict(title="Dimension 1", showticklabels=False),
            yaxis=dict(title="Dimension 2", showticklabels=False),
        )
        render_chart(
            fig_drift,
            caption="The lines connect the average centroids at each chronological time.",
        )


def _duplicates() -> None:
    st.subheader("Almost-duplicates that the deduplication by DOI did not catch")
    pairs = loaders.duplicate_pairs()
    overrides = loaders.duplicate_overrides()

    _, articles_df = loaders.articles()
    titles = (
        articles_df.set_index("doi")["title"]
        if "doi" in articles_df.columns and "title" in articles_df.columns
        else pd.Series(dtype="object")
    )
    st.caption(
        "DOI is the corpus's only reliable deduplication key (see `CLAUDE.md`), so the same "
        "work published under different DOIs may survive as separate records. These pairs were "
        "detected through abstract similarity and await human review. The dashboard remains "
        "read-only; record decisions with `lake-research-map duplicates`."
    )

    st.markdown("#### Pending review")
    if pairs.empty:
        st.success("No pair of almost identical abstracts are expected to be reviewed.")
    else:
        table = pairs.copy()
        table["Title A"] = table["doi_a"].map(titles)
        table["Title B"] = table["doi_b"].map(titles)
        table["Similarity"] = table["similarity"].round(4)
        st.dataframe(
            table[["Similarity", "Title A", "Title B", "doi_a", "doi_b"]].sort_values(
                "Similarity", ascending=False
            ),
            hide_index=True,
            width="stretch",
        )
        st.code(
            "uv run lake-research-map duplicates merge --canonical-doi DOI --duplicate-doi DOI "
            '--reason "rationale"\n'
            "uv run lake-research-map duplicates keep --doi-a DOI --doi-b DOI "
            '--reason "rationale"',
            language="bash",
        )

    st.markdown("####")
    if overrides.empty:
        st.info("No quasi-duplicate decision was recorded.")
        return

    history = overrides.copy()
    history["Decision"] = history["decision"].map({"merge": "Merge", "keep": "Keep separate"})
    history["Canonical DOI"] = history["canonical_doi"].fillna("—")
    history["Rationale"] = history["reason"]
    history["Updated at"] = history["updated_at"]
    st.dataframe(
        history[
            [
                "Decision",
                "Canonical DOI",
                "doi_a",
                "doi_b",
                "Rationale",
                "Updated at",
            ]
        ].sort_values("Updated at", ascending=False),
        hide_index=True,
        width="stretch",
    )
