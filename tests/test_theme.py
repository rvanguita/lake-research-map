"""Contract for the chart chrome that Streamlit would otherwise fill in itself.

Streamlit's frontend runs `layoutWithThemeDefaults` over every Plotly spec --
including with `theme=None`, which is what `components.render_chart` passes --
and backfills `paper_bgcolor`, `plot_bgcolor` and `font` from *its own* theme
whenever the figure's layout doesn't carry them. It reads the figure, never the
template, so styling declared only in `_figure_template` lost to Streamlit's
near-black background and every chart rendered as a black slab on the
dashboard's navy page. These assert the keys reach the serialized figure, where
Streamlit looks for them.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import pytest
import streamlit as st

from lake_research_map.dashboard.theme import (
    _DARK_TOKENS,
    _LIGHT_TOKENS,
    _THEME_STATE_KEY,
    CHART_PAPER_BG,
    polish_figure_layout,
)


@pytest.fixture
def theme_mode():
    """Switch the dashboard's dark/light mode for one test, then restore it.

    `st.session_state` works outside `streamlit run` (it only warns), which is
    what lets the theme be exercised without a running app.
    """
    original = st.session_state.get(_THEME_STATE_KEY)

    def _set(mode: str) -> None:
        st.session_state[_THEME_STATE_KEY] = mode

    yield _set

    if original is None:
        del st.session_state[_THEME_STATE_KEY]
    else:
        st.session_state[_THEME_STATE_KEY] = original


def _polished_layout() -> dict:
    fig = px.scatter(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), x="x", y="y")
    polish_figure_layout(fig)
    return fig.to_dict()["layout"]


def test_background_lands_on_the_figure_not_only_the_template(theme_mode) -> None:
    theme_mode("dark")
    layout = _polished_layout()
    # Straight off the figure: `layout["template"]` having the color is exactly
    # the case Streamlit ignores.
    assert layout["paper_bgcolor"] == CHART_PAPER_BG
    assert layout["plot_bgcolor"] == CHART_PAPER_BG


def test_figure_font_follows_the_dashboard_theme(theme_mode) -> None:
    theme_mode("dark")
    assert _polished_layout()["font"]["color"] == _DARK_TOKENS["chart_text"]
    theme_mode("light")
    assert _polished_layout()["font"]["color"] == _LIGHT_TOKENS["chart_text"]


def test_hover_box_keeps_a_solid_themed_fill(theme_mode) -> None:
    # With a transparent canvas, an unnamed hoverlabel background resolves to a
    # transparent box -- unreadable, most visibly on the `hovermode="x unified"`
    # charts.
    theme_mode("light")
    template = _polished_layout()["template"]
    assert template["layout"]["hoverlabel"]["bgcolor"] == _LIGHT_TOKENS["chart_bg"]


def test_both_themes_define_the_same_tokens() -> None:
    # A token added to one dict only is a color that silently comes out wrong
    # in the other theme.
    assert set(_DARK_TOKENS) == set(_LIGHT_TOKENS)


def test_polish_figure_layout_custom_margin() -> None:
    fig = px.scatter(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), x="x", y="y")
    custom_margin = dict(l=50, r=300, t=95, b=60)
    polish_figure_layout(fig, margin=custom_margin)
    assert fig.layout.margin.r == 300
    assert fig.layout.margin.l == 50
    assert fig.layout.margin.t == 95
    assert fig.layout.margin.b == 60


def test_polish_figure_layout_preserves_heatmap_textfont() -> None:
    fig = px.imshow([[1, 2], [3, 4]], text_auto=True)
    polish_figure_layout(fig)
    # Heatmap traces must not have a forced textfont.color so Plotly's auto-contrast survives
    trace = fig.data[0]
    assert (
        getattr(trace, "textfont", None) is None or getattr(trace.textfont, "color", None) is None
    )


def test_page_header_accepts_two_and_three_args(monkeypatch) -> None:
    from lake_research_map.dashboard.components import page_header

    recorded = []

    monkeypatch.setattr(st, "title", lambda text: recorded.append(("title", text)))
    monkeypatch.setattr(st, "caption", lambda text: recorded.append(("caption", text)))

    # 3-arg call
    page_header("🏷️", "Tópicos", "Descrição longa")
    assert recorded[-2] == ("title", "Tópicos")
    assert recorded[-1] == ("caption", "Descrição longa")

    # 2-arg call
    page_header("🏷️ Tópicos", "Descrição longa")
    assert recorded[-2] == ("title", "🏷️ Tópicos")
    assert recorded[-1] == ("caption", "Descrição longa")


def test_all_dashboard_pages_importable_and_render_callable() -> None:
    from lake_research_map.dashboard.app import PAGES

    assert len(PAGES) == 10
    for render_fn, title, icon, url_path in PAGES:
        assert callable(render_fn), f"Page {title} render function is not callable"
        assert title and isinstance(title, str)
        assert icon and isinstance(icon, str)
        assert url_path and isinstance(url_path, str)


def test_retired_dashboard_controllers_are_not_registered() -> None:
    from lake_research_map.dashboard.app import PAGES

    slugs = {url_path for _, _, _, url_path in PAGES}
    assert slugs.isdisjoint({"frontiers", "strategic", "search-config"})
    pages_dir = Path(__file__).parents[1] / "src/lake_research_map/dashboard/pages"
    for page in pages_dir.glob("*.py"):
        source = page.read_text(encoding="utf-8")
        assert "@st.cache_data" not in source, page.name
        assert "read_sql" not in source, page.name
        assert "get_engine" not in source, page.name


def test_polar_figure_gets_transparent_background() -> None:
    import plotly.graph_objects as go

    from lake_research_map.dashboard.theme import CHART_PAPER_BG, polish_figure_layout

    fig = go.Figure(data=go.Scatterpolar(r=[10, 20, 30], theta=["A", "B", "C"]))
    polish_figure_layout(fig)
    assert fig.layout.polar.bgcolor == CHART_PAPER_BG
