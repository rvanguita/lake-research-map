"""Figure builders shared by every dashboard page.

Each function returns a styled `plotly.graph_objects.Figure` ready to be
passed to `components.render_chart`. Centralizing these collapses the
near-identical charts that used to be copy-pasted across pages (top-N bars,
source-split bars/lines, stacked areas) into one implementation each.

Every builder takes `x_title`/`y_title`: a `go.Figure` has no axis titles at
all unless something sets them, and the pages that remembered to call
`update_layout` afterwards drifted apart from the ones that didn't. Naming the
axis where the figure is built is what keeps a chart from shipping with a bare
axis -- `components.render_chart` logs a warning when one does.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    OTHER_COLOR,
    PUBLICATION_CATEGORY_COLORS,
    PUBLICATION_CATEGORY_LABELS,
    SOURCE_COLORS,
    SOURCE_LABELS,
    TOTAL_COLOR,
    TOTAL_LABEL,
    hex_to_rgba,
    polish_figure_layout,
)


def _axis_titles(fig: go.Figure, x_title: str | None, y_title: str | None) -> None:
    """Name the axes, leaving whatever is already there when a title is `None`.

    Assigning `None` through `update_layout` would *clear* the axis title, which
    on a Plotly Express figure means throwing away the label px derived from the
    column name -- a raw `count`/`year` is ugly, but an empty axis is worse. So
    only the titles the caller actually supplied are written.
    """
    updates = {}
    if x_title is not None:
        updates["xaxis_title"] = x_title
    if y_title is not None:
        updates["yaxis_title"] = y_title
    if updates:
        fig.update_layout(**updates)


def source_bars(
    df: pd.DataFrame,
    x: str,
    *,
    title: str | None = None,
    total_line: bool = False,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Stacked IEEE/Elsevier bars, with an optional real Total line on top.

    `df` must have the shape produced by `analytics.source_counts_by`:
    columns `[x, "ieee", "elsevier", "total"]`.
    """
    fig = go.Figure()
    fig.add_bar(
        x=df[x], y=df["ieee"], name=SOURCE_LABELS["ieee"], marker_color=SOURCE_COLORS["ieee"]
    )
    fig.add_bar(
        x=df[x],
        y=df["elsevier"],
        name=SOURCE_LABELS["elsevier"],
        marker_color=SOURCE_COLORS["elsevier"],
    )
    fig.update_layout(barmode="stack", title=title)
    if total_line:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df["total"],
                name=TOTAL_LABEL,
                mode="lines+markers",
                line=dict(color=TOTAL_COLOR, width=2, dash="solid"),
                marker=dict(size=5),
            )
        )
    _axis_titles(fig, x_title, y_title)
    polish_figure_layout(fig)
    return fig


def source_lines(
    df: pd.DataFrame,
    x: str,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    spline: bool = False,
    fill: bool = False,
) -> go.Figure:
    """Three real line traces -- IEEE, Elsevier, and Total -- over `x`.

    `df` must have columns `[x, "ieee", "elsevier", "total"]`, e.g. from
    `analytics.cumulative_by_source` or `analytics.source_counts_by`.

    `spline=True` smooths the lines. `fill=True` shades the area under a
    source's line down to zero, but only when exactly one of IEEE/Elsevier
    is actually present in `df` (e.g. the sidebar is filtered to one source)
    -- filling both at once would just overlap two translucent regions with
    no added meaning.
    """
    line_shape = "spline" if spline else "linear"
    active_sources = [src for src in ("ieee", "elsevier") if df[src].sum() > 0]
    fill_single_source = fill and len(active_sources) == 1

    fig = go.Figure()
    for src in ("ieee", "elsevier"):
        line_kwargs: dict = {"x": df[x], "y": df[src], "mode": "lines+markers"}
        if fill_single_source and src in active_sources:
            line_kwargs["fill"] = "tozeroy"
            line_kwargs["fillcolor"] = hex_to_rgba(SOURCE_COLORS[src], 0.2)
        fig.add_trace(
            go.Scatter(
                name=SOURCE_LABELS[src],
                line=dict(color=SOURCE_COLORS[src], width=2, shape=line_shape),
                marker=dict(size=5),
                **line_kwargs,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df["total"],
            name=TOTAL_LABEL,
            mode="lines+markers",
            line=dict(color=TOTAL_COLOR, width=2.5, shape=line_shape),
            marker=dict(size=6),
        )
    )
    fig.update_layout(title=title)
    _axis_titles(fig, x_title, y_title)
    polish_figure_layout(fig)
    return fig


def publication_category_bars(
    df: pd.DataFrame,
    x: str = "year",
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    category_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Stacked publication-type bars with a reconciled Total trend line."""
    labels = category_labels or PUBLICATION_CATEGORY_LABELS
    fig = go.Figure()
    for category in PUBLICATION_CATEGORY_LABELS:
        label = labels.get(category, category)
        fig.add_bar(
            x=df[x],
            y=df[category],
            name=label,
            marker_color=PUBLICATION_CATEGORY_COLORS[category],
            hovertemplate=f"Year %{{x}}<br>{label}: %{{y:,}} articles<extra></extra>",
        )
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df["total"],
            name=TOTAL_LABEL,
            mode="lines+markers",
            line=dict(color=TOTAL_COLOR, width=2.5),
            marker=dict(size=6),
            hovertemplate="Year %{x}<br>Total: %{y:,} articles<extra></extra>",
        )
    )
    fig.update_layout(barmode="stack", title=title, hovermode="x unified")
    _axis_titles(fig, x_title, y_title)
    polish_figure_layout(fig)
    return fig


def topn_hbar(
    series: pd.Series,
    *,
    color_by: pd.Series | None = None,
    palette: dict[str, str] | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Top-N horizontal bar chart, optionally colored by a categorical series.

    `series` is indexed by category label, valued by the metric to rank on
    (already sorted/head-limited by the caller). `color_by`, if given, must
    share `series`'s index (e.g. the modal source per venue/author).
    """
    ordered = series.sort_values(ascending=True)
    df = pd.DataFrame({"label": ordered.index, "value": ordered.values})
    if color_by is not None:
        df["color"] = color_by.reindex(ordered.index).values
        color_map = palette or SOURCE_COLORS
        fig = px.bar(
            df,
            x="value",
            y="label",
            color="color",
            orientation="h",
            color_discrete_map=color_map,
            labels={"value": x_title or "", "label": y_title or "", "color": "Source"},
        )
    else:
        fig = px.bar(
            df,
            x="value",
            y="label",
            orientation="h",
            labels={"value": x_title or "", "label": y_title or ""},
            color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
        )
    fig.update_layout(title=title, showlegend=color_by is not None)
    polish_figure_layout(fig)
    return fig


def source_topn_hbar(
    df: pd.DataFrame,
    label_col: str,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Stacked IEEE/Elsevier horizontal bars, ranked by `total` (highest on top).

    `df` must have the shape produced by `analytics.source_counts_by`:
    columns `[label_col, "ieee", "elsevier", "total"]`, already head-limited
    by the caller. The horizontal counterpart to `source_bars`' vertical
    stacked bars -- for a top-N ranking, coloring a single bar by an entity's
    *modal* source (`topn_hbar`'s `color_by`) hides a source with fewer, but
    real, contributions whenever the corpus is source-skewed enough that
    "top overall" and "top in the minority source" barely overlap (every bar
    then renders in the majority source's color, looking like the minority
    source doesn't exist at all).
    """
    ordered = df.sort_values("total", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        y=ordered[label_col],
        x=ordered["ieee"],
        name=SOURCE_LABELS["ieee"],
        orientation="h",
        marker_color=SOURCE_COLORS["ieee"],
    )
    fig.add_bar(
        y=ordered[label_col],
        x=ordered["elsevier"],
        name=SOURCE_LABELS["elsevier"],
        orientation="h",
        marker_color=SOURCE_COLORS["elsevier"],
    )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title or "",
        showlegend=True,
    )
    polish_figure_layout(fig)
    return fig


def lorenz_chart(series: dict[str, pd.DataFrame], *, entity_label: str = "authors") -> go.Figure:
    """Lorenz curve: cumulative share of output vs. cumulative share of `entity_label`.

    `series` maps a source key ("ieee"/"elsevier"/"total") to a DataFrame with
    the shape from `analytics.lorenz_curve` (columns `share_of_authors`,
    `share_of_output`) -- one real line trace per key, colored via
    `SOURCE_COLORS`/`SOURCE_LABELS` ("ieee"/"elsevier") or `TOTAL_COLOR`/
    `TOTAL_LABEL` (anything else, i.e. "total"), matching `source_lines`'s
    3-real-series style. The perfect-equality diagonal is always a second kind
    of real trace, not a reference line -- consistent with the "Total is a
    real series" rule, generalized to this chart's own benchmark.

    `entity_label` only changes the x-axis wording (default "authors", the
    original use case in `researchers.py`) -- `lorenz_curve`'s own column
    names stay `share_of_authors`/`share_of_output` regardless of what's
    actually being ranked (e.g. venues instead of authors).
    """
    fig = go.Figure()
    for key, lorenz_df in series.items():
        color = SOURCE_COLORS.get(key, TOTAL_COLOR)
        name = SOURCE_LABELS.get(key, TOTAL_LABEL)
        fig.add_trace(
            go.Scatter(
                x=lorenz_df["share_of_authors"],
                y=lorenz_df["share_of_output"],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Equidade perfeita",
            mode="lines",
            line=dict(color=OTHER_COLOR, width=2, dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title=f"Cumulative share of {entity_label}",
        yaxis_title="Cumulative share of articles",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        yaxis=dict(tickformat=".0%", range=[0, 1]),
    )
    polish_figure_layout(fig)
    return fig


def stacked_area(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    color_map: dict[str, str] | None = None,
    category_orders: dict[str, list[str]] | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    groupnorm: str | None = None,
) -> go.Figure:
    """Stacked area chart, e.g. cumulative composition by venue/keyword.

    `groupnorm="percent"` turns this into a 100%-stacked area (used by the
    topic-share trend chart). `category_orders` (a px constructor argument,
    not a `fig.update_layout` kwarg) controls stacking/legend order.
    """
    fig = px.area(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_map,
        category_orders=category_orders,
        groupnorm=groupnorm,
    )
    if color_map and OTHER_COLOR not in color_map.values():
        pass  # caller is responsible for including the "Others" bucket color
    fig.update_layout(title=title)
    _axis_titles(fig, x_title, y_title)
    polish_figure_layout(fig)
    return fig
