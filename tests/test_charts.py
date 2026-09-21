"""Axis-naming contract for the shared figure builders.

A `go.Figure` has no axis titles unless something sets them, so a builder that
drops `x_title`/`y_title` on the floor produces a chart with bare axes and no
error anywhere -- which is exactly how three charts shipped unlabelled. These
assert the titles survive all the way onto the figure.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go

from lake_research_map.dashboard.charts import (
    lorenz_chart,
    source_bars,
    source_lines,
    source_topn_hbar,
    stacked_area,
    topn_hbar,
)
from lake_research_map.dashboard.components import _warn_unnamed_axes, metric_row, summary_card_row


def _by_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2020, "ieee": 2, "elsevier": 1, "total": 3},
            {"year": 2021, "ieee": 4, "elsevier": 2, "total": 6},
        ]
    )


def _axis_titles(fig) -> tuple[str | None, str | None]:
    return fig.layout.xaxis.title.text, fig.layout.yaxis.title.text


def test_source_bars_names_both_axes() -> None:
    fig = source_bars(
        _by_source(), "year", x_title="Ano de publicação", y_title="Quantidade de artigos"
    )
    assert _axis_titles(fig) == ("Ano de publicação", "Quantidade de artigos")


def test_source_lines_names_both_axes() -> None:
    fig = source_lines(
        _by_source(), "year", x_title="Ano de publicação", y_title="Artigos acumulados"
    )
    assert _axis_titles(fig) == ("Ano de publicação", "Artigos acumulados")


def test_topn_hbar_names_value_and_category_axes() -> None:
    series = pd.Series({"J. Liu": 5, "A. Silva": 3})
    fig = topn_hbar(series, x_title="Quantidade de artigos", y_title="Autor")
    assert _axis_titles(fig) == ("Quantidade de artigos", "Autor")


def test_topn_hbar_names_axes_when_colored_by_a_category() -> None:
    # The colored branch builds a different px.bar call, so it needs its own
    # assertion -- the labels dict is duplicated between the two.
    series = pd.Series({"J. Liu": 5, "A. Silva": 3})
    color_by = pd.Series({"J. Liu": "ieee", "A. Silva": "elsevier"})
    fig = topn_hbar(series, color_by=color_by, x_title="Quantidade de artigos", y_title="Autor")
    assert _axis_titles(fig) == ("Quantidade de artigos", "Autor")


def test_source_topn_hbar_names_both_axes() -> None:
    df = pd.DataFrame(
        [
            {"venue": "IEEE Trans. Power Syst.", "ieee": 4, "elsevier": 0, "total": 4},
            {"venue": "Int. J. Electr. Power", "ieee": 0, "elsevier": 3, "total": 3},
        ]
    )
    fig = source_topn_hbar(df, "venue", x_title="Quantidade de artigos", y_title="Periódico")
    assert _axis_titles(fig) == ("Quantidade de artigos", "Periódico")


def _venue_cumulative() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2020, "cumulative": 1, "venue": "A"},
            {"year": 2021, "cumulative": 3, "venue": "A"},
            {"year": 2021, "cumulative": 2, "venue": "B"},
        ]
    )


def test_stacked_area_names_both_axes() -> None:
    fig = stacked_area(
        _venue_cumulative(),
        x="year",
        y="cumulative",
        color="venue",
        x_title="Ano de publicação",
        y_title="Artigos acumulados",
    )
    assert _axis_titles(fig) == ("Ano de publicação", "Artigos acumulados")


def test_stacked_area_keeps_the_express_defaults_when_no_title_is_given() -> None:
    # Passing `None` through `update_layout` would clear the axis title that
    # Plotly Express derived from the column name, turning a merely ugly label
    # into an empty one. An omitted title must leave that default alone.
    fig = stacked_area(_venue_cumulative(), x="year", y="cumulative", color="venue")
    assert _axis_titles(fig) == ("year", "cumulative")


def test_lorenz_chart_names_both_axes_without_being_asked() -> None:
    curve = pd.DataFrame({"share_of_authors": [0.0, 0.5, 1.0], "share_of_output": [0.0, 0.2, 1.0]})
    fig = lorenz_chart({"total": curve}, entity_label="periódicos")
    assert _axis_titles(fig) == (
        "Parcela acumulada de periódicos",
        "Parcela acumulada de artigos",
    )


def test_warn_unnamed_axes_exempts_scatterpolar(caplog) -> None:
    fig = go.Figure(data=go.Scatterpolar(r=[1, 2, 3], theta=["A", "B", "C"]))
    with caplog.at_level("WARNING"):
        _warn_unnamed_axes(fig)
    assert not any("chart axis without a title" in r.message for r in caplog.records)


def test_warn_unnamed_axes_flags_unlabelled_cartesian(caplog) -> None:
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    with caplog.at_level("WARNING"):
        _warn_unnamed_axes(fig)
    assert any("chart axis without a title: xaxis" in r.message for r in caplog.records)


def test_warn_unnamed_axes_heatmaps_fallback(caplog) -> None:
    fig = go.Figure(data=go.Heatmap(z=[[1, 2], [3, 4]]))
    with caplog.at_level("WARNING"):
        _warn_unnamed_axes(fig)
    assert not any("chart axis without a title" in r.message for r in caplog.records)
    assert fig.layout.xaxis.title.text == "Dimensão X"
    assert fig.layout.yaxis.title.text == "Dimensão Y"


def test_metric_row_accepts_both_2_and_3_tuples() -> None:
    mock_container = MagicMock()
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    with (
        patch("streamlit.container", return_value=mock_container),
        patch("streamlit.columns", return_value=[mock_col1, mock_col2]),
    ):
        metric_row(
            [
                ("Taxa", "50%"),
                ("Score", "9.5", "+1.2"),
            ]
        )
        mock_col1.metric.assert_called_once_with("Taxa", "50%", None)
        mock_col2.metric.assert_called_once_with("Score", "9.5", "+1.2")

    long_value = "GAMS / AMPL (Modeladores Algébricos)"
    with (
        patch("streamlit.container", return_value=mock_container),
        patch("streamlit.columns", return_value=[mock_col1]),
        patch("streamlit.caption"),
        patch("streamlit.markdown") as markdown,
    ):
        summary_card_row([("Ferramenta mais citada", long_value, "42 artigos")])
        markdown.assert_called_once_with(f"### {long_value}")
