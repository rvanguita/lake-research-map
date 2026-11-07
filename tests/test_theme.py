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

import ast
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard.theme import (
    _DARK_TOKENS,
    CHART_PAPER_BG,
    _active_theme_type,
    polish_figure_layout,
)


def _polished_layout() -> dict:
    fig = px.scatter(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), x="x", y="y")
    polish_figure_layout(fig)
    return fig.to_dict()["layout"]


def test_background_lands_on_the_figure_not_only_the_template() -> None:
    layout = _polished_layout()
    # Straight off the figure: `layout["template"]` having the color is exactly
    # the case Streamlit ignores.
    assert layout["paper_bgcolor"] == CHART_PAPER_BG
    assert layout["plot_bgcolor"] == CHART_PAPER_BG


def test_figure_font_uses_the_fixed_dark_theme() -> None:
    assert _polished_layout()["font"]["color"] == _DARK_TOKENS["chart_text"]
    assert _active_theme_type() == "dark"


def test_hover_box_keeps_a_solid_themed_fill() -> None:
    # With a transparent canvas, an unnamed hoverlabel background resolves to a
    # transparent box -- unreadable, most visibly on the `hovermode="x unified"`
    # charts.
    template = _polished_layout()["template"]
    assert template["layout"]["hoverlabel"]["bgcolor"] == _DARK_TOKENS["chart_bg"]


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
    from apps.dashboard.components import page_header

    recorded = []

    monkeypatch.setattr(st, "title", lambda text: recorded.append(("title", text)))
    monkeypatch.setattr(st, "caption", lambda text: recorded.append(("caption", text)))

    # 3-arg call
    page_header("🏷️", "Topics", "Long description")
    assert recorded[-2] == ("title", "Topics")
    assert recorded[-1] == ("caption", "Long description")

    # 2-arg call
    page_header("🏷️ Topics", "Long description")
    assert recorded[-2] == ("title", "🏷️ Topics")
    assert recorded[-1] == ("caption", "Long description")


def test_all_dashboard_pages_importable_and_render_callable() -> None:
    from apps.dashboard.app import PAGES

    assert len(PAGES) == 10
    for render_fn, title, icon, url_path in PAGES:
        assert callable(render_fn), f"Page {title} render function is not callable"
        assert title and isinstance(title, str)
        assert icon and isinstance(icon, str)
        assert url_path and isinstance(url_path, str)


def test_retired_dashboard_controllers_are_not_registered() -> None:
    from apps.dashboard.app import PAGES

    slugs = {url_path for _, _, _, url_path in PAGES}
    assert slugs.isdisjoint({"frontiers", "strategic", "search-config"})
    pages_dir = Path(__file__).parents[1] / "src/apps/dashboard/pages"
    for page in pages_dir.glob("*.py"):
        source = page.read_text(encoding="utf-8")
        assert "@st.cache_data" not in source, page.name
        assert "read_sql" not in source, page.name
        assert "get_engine" not in source, page.name


def test_polar_figure_gets_transparent_background() -> None:
    import plotly.graph_objects as go

    from apps.dashboard.theme import CHART_PAPER_BG, polish_figure_layout

    fig = go.Figure(data=go.Scatterpolar(r=[10, 20, 30], theta=["A", "B", "C"]))
    polish_figure_layout(fig)
    assert fig.layout.polar.bgcolor == CHART_PAPER_BG


# A color written as a literal outside theme.py is a color one edit to the
# theme cannot move. Continuous data color scales are the documented exception
# (they color marks by value, not chrome), so only those keyword arguments are
# exempt; docstrings are not code and are skipped too.
_COLOR_LITERAL = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b|\brgba?\(")
_DATA_SCALE_KEYWORDS = {"color_continuous_scale", "colorscale"}
_DASHBOARD_ROOT = Path(__file__).parents[1] / "src/apps/dashboard"


def _color_literals(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in _DATA_SCALE_KEYWORDS:
            exempt.update(id(child) for child in ast.walk(node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in _DATA_SCALE_KEYWORDS:
                    exempt.update(id(child) for child in ast.walk(value))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            exempt.add(id(node.value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exempt
        and _COLOR_LITERAL.search(node.value)
    ]


def test_color_literal_detector_flags_chrome_but_not_data_scales() -> None:
    source = (
        'fig.update_traces(marker={"color": "#2a78d6"})\n'
        'px.scatter(df, color_continuous_scale=["#4a3aa7", "#eda100"])\n'
        'go.Scatter(marker={"colorscale": ["#000000", "#ffffff"]})\n'
        'go.Scatter(line=dict(color="rgba(255,255,255,0.4)"))\n'
        "label = f\"<span style='color:#888;'>x</span>\"\n"
        'community = f"#{index + 1}"\n'
    )
    assert sorted(line for line, _ in _color_literals(source)) == [1, 4, 5]


def test_dashboard_colors_come_from_theme_tokens() -> None:
    failures = []
    for path in sorted(_DASHBOARD_ROOT.rglob("*.py")):
        if path.name == "theme.py":
            continue
        for line, value in _color_literals(path.read_text(encoding="utf-8")):
            failures.append(f"{path.relative_to(_DASHBOARD_ROOT)}:{line}: {value[:60]!r}")
    assert not failures, "Use a theme.py token or constant instead:\n" + "\n".join(failures)
