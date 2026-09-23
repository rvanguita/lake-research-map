"""Shared UI pieces used by more than one page (sidebar, headers, tables)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import pandas as pd
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.airflow_client import AirflowError
from lake_research_map.dashboard.pipeline_control import (
    DAG_IDS,
    STAGE_LABELS,
    poll_run,
    trigger_stage,
)
from lake_research_map.dashboard.theme import (
    PUBLICATION_CATEGORY_LABELS,
    SOURCE_LABELS,
    add_redundant_encodings,
    polish_figure_layout,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATES = ("success", "failed", "error")


def _refresh_runs() -> None:
    """Poll every non-terminal run against Airflow. Clears the data cache the
    first time a run is observed to have finished, so the stats reflect it
    without a second manual refresh.
    """
    runs = st.session_state.pipeline_runs
    became_success = False
    for stage, run_ref in list(runs.items()):
        prev_state = run_ref.get("state")
        if prev_state in _TERMINAL_STATES:
            continue
        try:
            updated = poll_run(run_ref)
        except AirflowError as exc:
            logger.warning("_refresh_runs: could not poll stage %r", stage, exc_info=True)
            updated = dict(run_ref, state="error", error=str(exc))
        runs[stage] = updated
        if updated.get("state") == "success" and prev_state != "success":
            became_success = True
    if became_success:
        st.cache_data.clear()


def _trigger(stage: str) -> None:
    try:
        st.session_state.pipeline_runs[stage] = trigger_stage(stage)
    except AirflowError as exc:
        logger.warning("_trigger: could not trigger stage %r", stage, exc_info=True)
        st.session_state.pipeline_runs[stage] = {
            "stage": stage,
            "dag_id": DAG_IDS.get(stage, stage),
            "dag_run_id": None,
            "state": "error",
            "error": str(exc),
        }


def _prepare_filter_state(
    articles_df: pd.DataFrame,
) -> tuple[list[int], list[str], list[str], list[str]]:
    """Keep persisted widget values valid when a new medallion layer appears."""
    years = (
        pd.to_numeric(articles_df.get("year", pd.Series(dtype="float64")), errors="coerce")
        .dropna()
        .astype(int)
    )
    year_options = sorted(years[(years >= 1900) & (years <= 2100)].unique().tolist())
    sources = (
        sorted(articles_df["source"].dropna().astype(str).unique().tolist())
        if "source" in articles_df
        else []
    )
    venues = (
        sorted(articles_df["venue"].dropna().astype(str).unique().tolist())
        if "venue" in articles_df
        else []
    )
    categories = [
        category
        for category in PUBLICATION_CATEGORY_LABELS
        if category in set(articles_df.get("publication_category", pd.Series(dtype=str)).dropna())
    ]

    if year_options:
        bounds = (year_options[0], year_options[-1])
        current = st.session_state.get("global_year_range", bounds)
        if isinstance(current, (int, float)):
            current = (int(current), int(current))
            st.session_state.global_year_range = current
        if "global_year_range" in st.session_state and (
            not isinstance(current, (tuple, list))
            or len(current) != 2
            or current[0] < bounds[0]
            or current[1] > bounds[1]
        ):
            st.session_state.global_year_range = bounds
    else:
        st.session_state.pop("global_year_range", None)

    st.session_state.global_sources = [
        value for value in st.session_state.get("global_sources", []) if value in sources
    ]
    st.session_state.global_venues = [
        value for value in st.session_state.get("global_venues", []) if value in venues
    ]
    st.session_state.global_publication_categories = [
        value
        for value in st.session_state.get("global_publication_categories", [])
        if value in categories
    ]
    return year_options, sources, venues, categories


def _render_relevance_filter() -> None:
    """Opt-in cut on the semantic relevance margin.

    Defaults to off: the score is an aid to screening, not ground truth, so it
    must never silently change the numbers someone sees on first load.

    The cut is the contrastive margin's own zero -- "closer to logistics than
    to the review's topic" -- not a percentile. It used to be a percentile, and
    that was the wrong instrument: the single-anchor score's two distributions
    overlap on this corpus, so *any* percentile also discarded in-scope work.
    The margin separates them (see `transform/semantics.py`), which is what
    makes a fixed, explainable threshold possible.
    """
    signals = loaders.semantics()
    if signals.empty or "relevance_score" not in signals.columns:
        return
    # A database whose last `semantic` run predates the contrastive anchor has
    # no `offtopic_score` -- fall back to the old percentile rather than
    # dropping the filter out of the sidebar entirely.
    has_margin = "offtopic_score" in signals.columns

    st.checkbox(
        "Exclude out-of-scope articles",
        key="exclude_offtopic",
        help=(
            "Uses semantic signals (`--stage semantic`) to exclude articles far from the review "
            "topic, primarily the logistics and supply-chain cluster returned by the query."
        ),
    )
    if not st.session_state.get("exclude_offtopic"):
        st.session_state.pop("global_min_margin", None)
        return

    if has_margin:
        margin = signals["relevance_score"] - signals["offtopic_score"]
        threshold = st.slider(
            "Minimum margin",
            min_value=0.0,
            max_value=max(0.05, round(float(margin.quantile(0.75)), 2)),
            value=0.0,
            step=0.01,
            key="offtopic_margin",
            help=(
                "Zero excludes results closer to logistics than to the review topic. Higher "
                "values make screening stricter."
            ),
        )
        st.session_state.global_min_margin = threshold
        st.caption(
            f"Cutoff: margin ≥ {threshold:.2f} — {int((margin < threshold).sum()):,} excluded articles"
        )
        return

    percentile = st.slider(
        "Exclude below percentile",
        min_value=1,
        max_value=30,
        value=10,
        key="offtopic_percentile",
        help="10 removes the least relevant 10% of the corpus.",
    )
    threshold = float(signals["relevance_score"].quantile(percentile / 100))
    st.session_state.global_min_margin = threshold
    st.caption(f"Cutoff: score ≥ {threshold:.3f} (run `--stage semantic` to use the margin)")


def render_global_filters(articles_df: pd.DataFrame) -> None:
    """Render filters shared by every page and persist them in session state."""
    years, sources, venues, categories = _prepare_filter_state(articles_df)
    st.subheader("Global filters", icon=":material/filter_alt:")
    if years:
        if years[0] < years[-1]:
            slider_kwargs = {}
            if "global_year_range" not in st.session_state:
                slider_kwargs["value"] = (years[0], years[-1])
            st.slider(
                "Publication year",
                min_value=years[0],
                max_value=years[-1],
                key="global_year_range",
                help="Applies to all metrics, charts, tables, and exports.",
                **slider_kwargs,
            )
        else:
            st.session_state.global_year_range = (years[0], years[0])
            st.caption(f"Publication year: {years[0]}")
    else:
        st.caption("Publication year is unavailable in the active layer.")

    if sources:
        st.multiselect(
            "Source",
            options=sources,
            key="global_sources",
            format_func=lambda value: SOURCE_LABELS.get(value, value.title()),
        )
    else:
        st.caption("Source is unavailable in the active layer.")
    if venues:
        st.multiselect(
            "Venue / event",
            options=venues,
            key="global_venues",
            placeholder="All venues",
        )
    else:
        st.caption("Venue or event is unavailable in the active layer.")

    if categories:
        st.multiselect(
            "Publication category",
            options=categories,
            key="global_publication_categories",
            format_func=lambda value: PUBLICATION_CATEGORY_LABELS[value],
            placeholder="All categories",
        )
    else:
        st.caption("Publication category is unavailable in the active layer.")

    _render_relevance_filter()

    if st.button(
        "Clear filters",
        key="clear_global_filters",
        icon=":material/filter_alt_off:",
        width="stretch",
    ):
        st.session_state.global_year_range = (years[0], years[-1]) if years else None
        st.session_state.global_sources = []
        st.session_state.global_venues = []
        st.session_state.global_publication_categories = []
        st.session_state.pop("global_min_margin", None)
        st.session_state.exclude_offtopic = False
        st.rerun()

    _, filtered = loaders.filtered_articles()
    if filtered.empty:
        st.warning("The current filters return no articles.")
    elif len(filtered) != len(articles_df):
        st.caption(f"Showing **{len(filtered):,}** of **{len(articles_df):,} articles")


def render_sidebar() -> None:
    """Render compact global filters; operational controls live on the pipeline page."""
    layer, articles_df = loaders.articles()

    with st.sidebar:
        if not articles_df.empty:
            with st.expander("Global filters", expanded=True, icon=":material/filter_alt:"):
                render_global_filters(articles_df)
        if layer != "none" and not articles_df.empty:
            _, filtered_df = loaders.filtered_articles()
            st.caption(f"{len(filtered_df):,}/{len(articles_df):,} articles · **{layer}** layer")


def render_pipeline_controls() -> None:
    """Render Airflow controls in the operational page instead of every page."""
    st.session_state.setdefault("pipeline_runs", {})
    _refresh_runs()
    st.subheader("Run pipeline through Airflow")

    with st.container(border=True):
        columns = st.columns(3)
        for index, stage in enumerate(("raw", "bronze", "silver", "gold", "embed", "semantic")):
            with columns[index % len(columns)]:
                if st.button(
                    f"Run {STAGE_LABELS[stage]}",
                    key=f"run_{stage}",
                    icon=":material/play_arrow:",
                    width="stretch",
                ):
                    with st.spinner(f"Triggering {STAGE_LABELS[stage]} in Airflow..."):
                        _trigger(stage)

        if st.button(
            "Run all (raw→bronze→silver→gold→embed→semantic)",
            key="run_all",
            icon=":material/fast_forward:",
        ):
            with st.spinner("Triggering the complete pipeline in Airflow..."):
                _trigger("all")

        if st.session_state.pipeline_runs:
            with st.container(border=True):
                st.markdown("**Triggered runs**")
                for stage, run_ref in st.session_state.pipeline_runs.items():
                    label = STAGE_LABELS.get(stage, stage)
                    state = run_ref.get("state")
                    run_id = run_ref.get("dag_run_id")
                    if state == "success":
                        extra = ""
                        tasks = run_ref.get("task_instances") or []
                        if tasks:
                            n_ok = sum(1 for t in tasks if t.get("state") == "success")
                            extra = f" ({n_ok}/{len(tasks)} completed tasks)"
                        st.success(f"{label}: completed{extra}")
                    elif state == "failed":
                        st.error(f"{label}: failed (run `{run_id}`; check the Airflow logs)")
                    elif state == "error":
                        st.error(f"{label}: {run_ref.get('error', 'unknown error')}")
                    else:
                        st.info(f"{label}: {state} (run `{run_id}`)")

    if st.button("Refresh data", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()


def page_header(
    icon_or_title: str,
    title_or_desc: str,
    description: str | None = None,
) -> None:
    """Consistent native title block. Legacy icon arguments are ignored."""
    if description is None:
        st.title(icon_or_title)
        st.caption(title_or_desc)
    else:
        st.title(title_or_desc)
        st.caption(description)


def metric_row(metrics: Sequence[tuple[str, str] | tuple[str, str, str | None]]) -> None:
    """A bordered row of metrics: list of (label, value) or (label, value, delta|None)."""
    with st.container(border=True):
        cols = st.columns(len(metrics))
        for col, item in zip(cols, metrics, strict=True):
            if len(item) == 2:
                label, value = item
                delta = None
            else:
                label, value, delta = item  # type: ignore[misc]
            col.metric(label, value, delta)


def summary_card_row(cards: Sequence[tuple[str, str] | tuple[str, str, str | None]]) -> None:
    """Render wrapping summary cards for categorical values that do not fit in ``st.metric``."""
    columns = st.columns(len(cards))
    for column, item in zip(columns, cards, strict=True):
        label, value = item[:2]
        detail = item[2] if len(item) == 3 else None
        with column:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"### {value}")
                if detail:
                    st.caption(detail)


def hero_banner(title: str, body_html: str) -> None:
    """Native bordered summary block shared by analytical pages."""
    markdown = body_html
    for source, target in (("<b>", "**"), ("</b>", "**"), ("<i>", "*"), ("</i>", "*")):
        markdown = markdown.replace(source, target)
    markdown = markdown.replace("<code>", "`").replace("</code>", "`")
    markdown = re.sub(r"<[^>]+>", "", markdown)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(markdown)


def require_columns(df: pd.DataFrame, cols: list[str], message: str | None = None) -> bool:
    """Guard a chart function against a layer that lacks required columns.

    Returns True (and renders nothing) if all `cols` are present; otherwise
    renders a consistent `st.info` and returns False so the caller can
    `return` early.
    """
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return True
    st.info(
        message or f"Column(s) {', '.join(f'`{c}`' for c in missing)} unavailable in this layer."
    )
    return False


# Trace types that have no cartesian axes at all, so "this axis has no title"
# says nothing about them.
_AXISLESS_TRACES = frozenset(
    {
        "pie",
        "sankey",
        "indicator",
        "treemap",
        "sunburst",
        "funnelarea",
        "table",
        "scatterpolar",
        "scatterpolargl",
        "barpolar",
        "choropleth",
        "scattergeo",
        "scattermapbox",
        "scattermap",
        "scatterternary",
    }
)


def _warn_unnamed_axes(fig) -> None:
    """Log the charts whose visible axes carry no title.

    Every chart in the dashboard goes through `render_chart`, which makes this
    the one place that can audit all of them at once: run the app, click
    through the pages, and the terminal lists whatever is still unlabelled.

    It logs rather than calling `st.warning` on purpose -- a missing axis title
    is a note for whoever is editing the page, not something to show a reviewer
    reading the corpus. An axis explicitly hidden (`visible=False`) or stripped
    of its ticks is exempt: that's how the t-SNE map and the co-authorship
    network declare that their coordinates carry no meaning.
    """
    if all(trace.type in _AXISLESS_TRACES for trace in fig.data):
        return
    for name in ("xaxis", "yaxis"):
        axis = fig.layout[name]
        if axis.visible is False or axis.showticklabels is False:
            continue
        if axis.title and axis.title.text:
            continue
        if any(trace.type == "heatmap" for trace in fig.data):
            fallback = "Dimension X" if name == "xaxis" else "Dimension Y"
            fig.update_layout({f"{name}_title": fallback})
            continue
        chart_title = (fig.layout.title.text if fig.layout.title else None) or ",".join(
            sorted({trace.type for trace in fig.data})
        )
        logger.warning("chart axis without a title: %s on %r", name, chart_title)


def render_chart(
    fig,
    *,
    caption: str | None = None,
    height: int | None = None,
    margin: dict | None = None,
    key: str | None = None,
) -> None:
    """Apply the shared light/dark theme, render the figure, add its caption.

    Replaces the `polish_figure_layout(fig); st.plotly_chart(...); st.caption(...)`
    triplet that used to be repeated in every chart function.

    `theme=None` (not Streamlit's `"streamlit"` default) because the figure is
    already fully styled by `polish_figure_layout`: Streamlit's theme would
    override that template's background with its own, which follows the
    browser/system setting rather than a dashboard theme toggle
    -- so charts rendered near-black against the navy page background.
    """
    polish_figure_layout(fig, height=height, margin=margin)
    add_redundant_encodings(fig)
    _warn_unnamed_axes(fig)
    st.plotly_chart(fig, theme=None, width="stretch", key=key)
    if caption:
        st.caption(caption)


# Shared table formatting for the "pick a reference" tables -- DOI becomes a
# clickable doi.org link.
_ARTICLE_TABLE_CONFIG = {
    "title": st.column_config.TextColumn("Title", width="large"),
    "year": st.column_config.NumberColumn("Year", format="%d"),
    "venue": st.column_config.TextColumn("Venue / event"),
    "source": st.column_config.TextColumn("Source"),
    "publication_category": st.column_config.TextColumn("Publication category"),
    "citation_count": st.column_config.NumberColumn("Citations", format="%d"),
    "reference_count": st.column_config.NumberColumn("References", format="%d"),
    "author_count": st.column_config.NumberColumn("Authors", format="%d"),
    "doi_link": st.column_config.LinkColumn("DOI", display_text=r"10\..*"),
}


def article_table(df: pd.DataFrame, columns: list[str], download_key: str = "") -> None:
    """Article table with clickable DOIs, formatted numbers and a CSV export."""
    table = df.copy()
    if "doi" in table.columns:
        table["doi_link"] = "https://doi.org/" + table["doi"].astype(str)
        columns = [c for c in columns if c != "doi"] + ["doi_link"]
    present = [c for c in columns if c in table.columns]
    st.dataframe(
        table[present],
        hide_index=True,
        width="stretch",
        column_config={k: v for k, v in _ARTICLE_TABLE_CONFIG.items() if k in present},
    )
    if download_key:
        st.download_button(
            "Download CSV",
            data=table[present].to_csv(index=False).encode("utf-8"),
            file_name=f"{download_key}.csv",
            mime="text/csv",
            key=f"dl_{download_key}",
            icon=":material/download:",
        )


def _partition_review_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate human evidence from explicitly assisted draft reviews.

    Reviewer identifiers ending in ``-assisted`` are an operational convention:
    their labels remain auditable, but must not silently upgrade an exploratory
    taxonomy into a human-validated one. Older exports without ``reviewer_id`` are
    treated as human evidence for backward compatibility.
    """
    if labels.empty or "reviewer_id" not in labels.columns:
        return labels, labels.iloc[0:0].copy()
    assisted = labels["reviewer_id"].fillna("").astype(str).str.casefold().str.endswith("-assisted")
    return labels.loc[~assisted].copy(), labels.loc[assisted].copy()


def taxonomy_disclosure(df, taxonomy_name: str, *, key: str = "") -> None:
    """State what share of the corpus a regex taxonomy actually classifies.

    `WP-12` requires every displayed taxonomy to report its evaluated coverage
    and either meet a declared minimum precision or be labelled exploratory.
    Human-reviewed labels can upgrade a taxonomy from exploratory to measured;
    explicitly assisted draft reviews remain visible in the audit trail but do
    not cross that evidence boundary.

    The distinction matters because a frequency chart drawn over matched
    articles only answers "among the ones I recognised, which is most common?"
    while appearing to answer "what does this corpus do?". On five of the seven
    taxonomies here the recognised share is under a third.

    Precision upgrades itself once human labels exist for this taxonomy. Draft
    labels from a reviewer whose identifier ends in ``-assisted`` are excluded.
    """
    from lake_research_map.dashboard.analytics import (
        TAXONOMY_REGISTRY,
        taxonomy_coverage,
        taxonomy_precision_from_labels,
    )

    patterns = TAXONOMY_REGISTRY.get(taxonomy_name)
    if patterns is None or df is None or getattr(df, "empty", True):
        return
    stats = taxonomy_coverage(df, patterns)
    if not stats["population"]:
        return

    scored = pd.DataFrame()
    assisted_labels = pd.DataFrame()
    try:
        labels = loaders.review_labels("taxonomy")
        if not labels.empty:
            human_labels, assisted_labels = _partition_review_labels(labels)
            scored = taxonomy_precision_from_labels(df, patterns, human_labels)
    except Exception:
        # A missing review table must not take down an analytical page.
        logger.debug("taxonomy_disclosure: review labels unavailable", exc_info=True)

    with st.expander(f"Coverage and validation — {taxonomy_name}", expanded=False):
        metric_row(
            [
                (
                    "\U0001f4d0 Corpus classified",
                    f"{stats['coverage']:.1%}",
                    f"{stats['classified']:,} of {stats['population']:,} articles",
                ),
                (
                    "\U0001f573️ Matched no class",
                    f"{stats['unclassified']:,}",
                    "Invisible to the chart above",
                ),
                (
                    "\U0001f501 In more than one class",
                    f"{stats['multi_label']:,}",
                    "Classes overlap; counts are not a partition",
                ),
            ]
        )
        if scored.empty:
            st.warning(
                "**Exploratory.** No human-reviewed sample exists for this taxonomy, so its "
                "precision and recall are unmeasured — the classes are regex matches over "
                "title and abstract, not human-validated labels. Read the chart as "
                f'"among the {stats["coverage"]:.0%} of articles this rule recognises", '
                "never as a description of the corpus. Generate a sample with "
                "`lake-research-map evidence taxonomy` to replace this notice with "
                "measured per-class precision."
            )
            if not assisted_labels.empty:
                st.info(
                    f"An assisted draft contains {len(assisted_labels):,} labels. It remains "
                    "in the audit trail but is excluded from validation metrics until "
                    "independent human review and adjudication are complete."
                )
        else:
            display = scored.copy()
            for column in ("precision", "recall", "f1"):
                display[column] = display[column].map(
                    lambda value: "n/a" if pd.isna(value) else f"{value:.2f}"
                )
            st.dataframe(display, hide_index=True, width="stretch")
            st.caption(
                "Precision and recall are computed against reviewed labels; `ambiguous` "
                "rows are excluded from both and counted separately, because a reviewer "
                "who could not decide is evidence about the class boundary rather than a "
                "negative. Recall is bounded by what the sample covered, not by the corpus."
            )
