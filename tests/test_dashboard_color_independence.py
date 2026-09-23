"""NFR-07: series must be distinguishable without relying on color alone."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lake_research_map.dashboard import charts
from lake_research_map.dashboard.theme import add_redundant_encodings


def _channels(trace) -> tuple:
    """Every non-color channel a trace can carry."""
    marker = getattr(trace, "marker", None)
    pattern = getattr(marker, "pattern", None)
    return (
        getattr(marker, "symbol", None),
        getattr(getattr(trace, "line", None), "dash", None),
        getattr(pattern, "shape", None) or None,
        getattr(getattr(trace, "fillpattern", None), "shape", None) or None,
    )


def _legend_series(fig) -> dict[str, tuple]:
    series = {}
    for trace in fig.data:
        key = trace.legendgroup or (trace.name if trace.showlegend is not False else None)
        if key:
            series.setdefault(key, _channels(trace))
    return series


def _assert_distinct_without_color(fig) -> None:
    series = _legend_series(fig)
    if len(series) < 2:
        return
    assert len(set(series.values())) == len(series), series


def test_line_series_get_distinct_symbols_and_dashes():
    fig = go.Figure()
    for name in ("IEEE", "Elsevier", "Total"):
        fig.add_scatter(x=[1, 2], y=[1, 2], name=name, mode="lines+markers")

    add_redundant_encodings(fig)

    assert [t.marker.symbol for t in fig.data] == ["circle", "square", "diamond"]
    assert [t.line.dash for t in fig.data] == ["solid", "dash", "dot"]


def test_bar_series_get_patterns_and_the_first_stays_solid():
    fig = go.Figure()
    fig.add_bar(x=["a"], y=[1], name="IEEE")
    fig.add_bar(x=["a"], y=[2], name="Elsevier")

    add_redundant_encodings(fig)

    assert fig.data[0].marker.pattern.shape == ""
    assert fig.data[1].marker.pattern.shape == "/"
    # Overlay keeps the series color under the pattern.
    assert fig.data[1].marker.pattern.fillmode == "overlay"


def test_single_series_and_annotation_traces_are_untouched():
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[1, 2], name="Observed")
    fig.add_scatter(x=[1, 2], y=[0, 1], showlegend=False, fill="tonexty")

    add_redundant_encodings(fig)

    assert fig.data[0].marker.symbol is None
    assert fig.data[1].line.dash is None


def test_an_encoding_that_already_distinguishes_series_is_kept():
    fig = go.Figure()
    fig.add_scatter(x=[1], y=[1], name="A", mode="markers", marker_symbol="star")
    fig.add_scatter(x=[1], y=[2], name="B", mode="markers", marker_symbol="diamond")

    add_redundant_encodings(fig)

    assert [t.marker.symbol for t in fig.data] == ["star", "diamond"]


def test_an_encoding_shared_by_every_series_is_replaced():
    # Plotly Express writes symbol="circle" on every trace; that is a default,
    # not a decision, and leaves the series told apart by color alone.
    fig = go.Figure()
    fig.add_scatter(x=[1], y=[1], name="A", mode="markers", marker_symbol="circle")
    fig.add_scatter(x=[1], y=[2], name="B", mode="markers", marker_symbol="circle")

    add_redundant_encodings(fig)

    assert [t.marker.symbol for t in fig.data] == ["circle", "square"]


def test_continuous_color_scales_are_not_given_symbols():
    fig = px.scatter(x=[1, 2, 3], y=[1, 2, 3], color=[0.1, 0.5, 0.9])
    fig.add_scatter(x=[1], y=[1], name="Anchor", mode="markers")
    fig.add_scatter(x=[2], y=[2], name="Anchor 2", mode="markers")
    before = fig.data[0].marker.symbol

    add_redundant_encodings(fig)

    assert fig.data[0].marker.symbol == before
    assert [t.marker.symbol for t in fig.data[1:]] == ["circle", "square"]


def test_px_legend_groups_share_one_encoding_across_facets():
    df = pd.DataFrame(
        {"x": [1, 2, 1, 2], "y": [1, 2, 3, 4], "g": ["a", "b", "a", "b"], "f": [0, 0, 1, 1]}
    )
    fig = px.line(df, x="x", y="y", color="g", facet_col="f", markers=True)

    add_redundant_encodings(fig)

    by_group = {}
    for trace in fig.data:
        by_group.setdefault(trace.legendgroup, set()).add(trace.marker.symbol)
    assert all(len(symbols) == 1 for symbols in by_group.values())
    assert by_group["a"] != by_group["b"]


def test_shared_chart_builders_are_readable_without_color():
    counts = pd.DataFrame(
        {"year": [2020, 2021], "ieee": [1, 2], "elsevier": [3, 4], "total": [4, 6]}
    )
    shares = pd.DataFrame(
        {"year": [2020, 2020, 2021, 2021], "n": [1, 2, 3, 4], "venue": ["A", "B", "A", "B"]}
    )
    figures = [
        charts.source_bars(counts, "year", total_line=True),
        charts.source_lines(counts, "year"),
        charts.stacked_area(shares, x="year", y="n", color="venue"),
    ]
    for fig in figures:
        add_redundant_encodings(fig)
        _assert_distinct_without_color(fig)
