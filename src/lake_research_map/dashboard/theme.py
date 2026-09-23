"""Colors and small chart helpers shared by every dashboard page.

Keeping these in one module is what makes IEEE blue mean IEEE on every page,
and keeps the venue/categorical palettes from drifting apart over time.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

logger = logging.getLogger(__name__)

# Brand colors -- IEEE blue and Elsevier orange, from each publisher's own brand
# guidelines. Used everywhere a chart breaks down by source, so the same two
# colors always mean the same two publishers across the whole app.
SOURCE_COLORS = {"ieee": "#00629B", "elsevier": "#FF6C00"}
SOURCE_LABELS = {"ieee": "IEEE", "elsevier": "Elsevier"}
PUBLICATION_CATEGORY_LABELS = {
    "journal": "Journal",
    "conference": "Conference",
    "review": "Review",
    "other": "Other",
}
PUBLICATION_CATEGORY_COLORS = {
    "journal": "#2a78d6",
    "conference": "#eb6834",
    "review": "#1baf7a",
    "other": "#9a9a94",
}

# Validated categorical palette (dataviz skill's default 8-slot theme) for charts
# that distinguish many sub-categories within one source (e.g. venues) -- a
# monochromatic brand-color ramp is the wrong tool there (low contrast between
# shades of the same hue); this fixed, CVD-checked order is deliberately NOT
# derived from SOURCE_COLORS. "Others" (the catch-all bucket) always gets a
# neutral gray instead of a palette slot -- it isn't a real category.
CATEGORICAL_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
OTHER_COLOR = "#9a9a94"

# Diverging pair for growth/decline charts -- deliberately not the brand colors,
# which carry publisher meaning everywhere else.
TREND_UP_COLOR = "#1baf7a"
TREND_DOWN_COLOR = "#4a3aa7"

# "Total" needs to visually pop against both IEEE blue and Elsevier orange on
# every chart that shows all three series together -- a neutral gray used to
# sit here and nearly disappeared next to the two saturated brand colors.
TOTAL_COLOR = "#ff2e77"
TOTAL_LABEL = "Total"

CHART_HEIGHT = 420  # consistent height for side-by-side chart pairs

# The chart canvas is transparent so the page's own gradient shows through it,
# instead of a slab of one flat color sitting on top of a gradient that has
# already moved on by the bottom of a long page. This is the one chart-chrome
# color that is deliberately NOT a light/dark token: transparent is correct in
# both themes by construction, because it *is* whatever the page is.
# `chart_bg` stays a solid token, for the things that need a real color to
# stand on: the gauge track in `quality.py` and the "partial year" marker halo
# in `forecasting.py`.
CHART_PAPER_BG = "rgba(0,0,0,0)"

_CHART_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

# ---------------------------------------------------------------------------
# Fixed dark-theme tokens. Streamlit and Plotly both use this single palette,
# so browser or system preferences cannot produce mismatched chart chrome.
# ---------------------------------------------------------------------------

_DARK_TOKENS = {
    "bg_top": "#07131f",
    "bg_mid": "#0b1725",
    "bg_bottom": "#0f1d2b",
    "sidebar_bg": "rgba(10, 18, 28, 0.95)",
    "border": "rgba(148, 163, 184, 0.18)",
    "text": "#e5eefb",
    "muted": "#a5b7d5",
    "accent": "#5ea9ff",
    "input_bg": "rgba(17, 24, 39, 0.9)",
    "input_border": "rgba(148, 163, 184, 0.2)",
    "metric_bg": "linear-gradient(135deg, rgba(12, 46, 75, 0.96), rgba(18, 78, 140, 0.72))",
    "metric_border": "rgba(94, 169, 255, 0.24)",
    "metric_label": "#d7e8ff",
    "metric_value": "#f8fbff",
    "table_bg": "linear-gradient(180deg, rgba(12, 28, 38, 0.98), rgba(16, 31, 45, 0.94))",
    "table_border": "rgba(94, 169, 255, 0.25)",
    "expander_bg": "linear-gradient(180deg, rgba(14, 31, 44, 0.95), rgba(19, 38, 54, 0.92))",
    "button_bg": "linear-gradient(135deg, rgba(94,169,255,0.18), rgba(17,24,39,0.9))",
    "button_border": "rgba(94,169,255,0.28)",
    "tab_bg": "rgba(12, 24, 34, 0.7)",
    "tab_active_bg": "rgba(94, 169, 255, 0.12)",
    "alert_bg": "rgba(17, 24, 39, 0.6)",
    "chart_bg": "#0b1725",
    "chart_text": "#e5eefb",
    "chart_tick": "#dfeaf9",
    "chart_annotation": "#f8fbff",
    "grid": "rgba(148,163,184,0.16)",
    "axis_line": "rgba(148,163,184,0.30)",
    # A solid navy-blue "card" tint (matches the metric_bg gradient's second
    # stop -- Plotly's legend.bgcolor can't render a CSS gradient) so the
    # legend reads as a distinct floating chip instead of nearly disappearing
    # into chart_bg (#0b1725), which the previous near-identical rgba did.
    "legend_bg": "rgba(18, 78, 140, 0.55)",
    "legend_border": "rgba(94, 169, 255, 0.35)",
    "heatmap_zero": "#0b1725",
    "heatmap_mid": "#0b1725",
    "reference_line": "rgba(255, 255, 255, 0.75)",
    "reference_line_subtle": "rgba(255, 255, 255, 0.35)",
    "point_border": "#ffffff",
}


def _active_theme_type() -> str:
    """Return the single supported dashboard theme."""
    return "dark"


def _tokens() -> dict[str, str]:
    return _DARK_TOKENS


def theme_tokens() -> dict[str, str]:
    """Public accessor for the active dark-theme token set.

    Use this from a page/component that needs to color its own inline HTML
    or a Plotly Indicator (which `polish_figure_layout` doesn't touch) to
    match the current theme -- see `components.hero_banner` or
    `quality._embedding_readiness` for examples.
    """
    return _tokens()


def apply_dashboard_theme() -> None:
    """Apply the fixed executive dark visual treatment."""
    t = _tokens()
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {t["bg_top"]};
            --bg-2: {t["bg_bottom"]};
            --border: {t["border"]};
            --text: {t["text"]};
            --muted: {t["muted"]};
            --accent: {t["accent"]};
        }}

        .stApp {{
            background: linear-gradient(180deg, {t["bg_top"]} 0%, {t["bg_mid"]} 28%, {t["bg_bottom"]} 100%);
            color: var(--text);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        h1, h2, h3, h4, h5, h6,
        p, div, span, label,
        .stCaption, .stMarkdown {{
            color: var(--text) !important;
        }}

        .stMarkdown h1 {{
            font-size: 2.2rem;
            letter-spacing: -0.04em;
            margin-bottom: .25rem;
        }}

        .stMarkdown h2 {{
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        }}

        .stMarkdown p, .stMarkdown li, .stMarkdown div {{
            color: var(--muted) !important;
        }}

        [data-testid="stSidebar"] {{
            background: {t["sidebar_bg"]};
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {{
            color: var(--text) !important;
        }}

        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stMultiSelect,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stButton > button {{
            background: {t["input_bg"]};
            color: var(--text);
            border-color: {t["input_border"]};
        }}

        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] {{
            background: {t["input_bg"]} !important;
            color: var(--text) !important;
            border-color: {t["input_border"]} !important;
        }}

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: var(--text) !important;
        }}

        .stSlider [data-testid="stThumbValue"],
        .stSlider .stMarkdown p,
        .stSlider label,
        .stSlider span {{
            color: var(--text) !important;
        }}

        .stSlider > div > div > div {{
            background: {t["input_bg"]};
            border: 1px solid {t["input_border"]};
        }}

        [data-testid="stMetric"] {{
            background: {t["metric_bg"]};
            border: 1px solid {t["metric_border"]};
            border-radius: 0.8rem;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 10px 22px rgba(8, 19, 31, 0.18);
        }}

        [data-testid="stMetricLabel"] {{
            font-size: .82rem;
            font-weight: 600;
            color: {t["metric_label"]} !important;
            letter-spacing: .02em;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {t["metric_value"]} !important;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 0.6rem;
            border: 1px solid {t["table_border"]};
            overflow: hidden;
            background: {t["table_bg"]};
        }}

        div[data-testid="stExpander"] > details {{
            border: 1px solid {t["table_border"]};
            border-radius: 0.7rem;
            background: {t["expander_bg"]};
        }}

        div[data-testid="stExpander"] summary {{
            color: var(--text) !important;
            font-weight: 600;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: {t["button_bg"]};
            color: var(--text);
            border: 1px solid {t["button_border"]};
            border-radius: 0.65rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: {t["accent"]};
            box-shadow: 0 0 0 1px {t["button_border"]};
            transform: translateY(-1px);
        }}

        .stTabs [role="tablist"] {{
            background: {t["tab_bg"]};
            border: 1px solid var(--border);
            border-radius: 0.7rem;
            padding: 0.1rem;
        }}

        .stTabs [role="tab"] {{
            color: var(--muted) !important;
            border-radius: 0.55rem;
        }}

        .stTabs [role="tab"][aria-selected="true"] {{
            background: {t["tab_active_bg"]};
            color: var(--text) !important;
            border: 1px solid {t["button_border"]};
        }}

        .stTabs [role="tab"]:hover {{
            color: var(--text) !important;
        }}

        .stAlert {{
            background: {t["alert_bg"]};
            border: 1px solid var(--border);
            color: var(--text) !important;
        }}

        .stException {{
            background: rgba(255, 107, 107, 0.12);
            border: 1px solid rgba(255, 107, 107, 0.3);
        }}

        /* The figure itself is transparent (see CHART_PAPER_BG); the
           containers Streamlit wraps it in have to be too, or they'd repaint
           the flat slab we just removed. */
        [data-testid="stPlotlyChart"],
        .stPlotlyChart,
        .js-plotly-plot,
        .plot-container,
        .js-plotly-plot .svg-container {{
            background: {CHART_PAPER_BG} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def venue_color_map(venue_rank_list: list[str], others_label: str = "Others") -> dict[str, str]:
    """Assign fixed categorical colors in rank order; the catch-all bucket always gray."""
    colors: dict[str, str] = {}
    slot = 0
    for venue in venue_rank_list:
        if venue == others_label:
            colors[venue] = OTHER_COLOR
        else:
            colors[venue] = CATEGORICAL_PALETTE[slot % len(CATEGORICAL_PALETTE)]
            slot += 1
    return colors


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


@lru_cache(maxsize=4)
def _figure_template(theme_type: str, has_title: bool):
    """The shared chart styling as a Plotly template, built once per
    (theme, has_title) combination.

    Passing this nested styling to `update_layout` on every figure cost ~9ms
    each in Plotly's per-property validation -- with ~20 charts per page that
    was the single largest cost in rendering a page. A template is validated
    once here and afterwards applied by reference (~0.8ms). It layers on top
    of Plotly's own `plotly` template so the built-in colorscales the Express
    charts rely on survive.

    The legend offset depends on whether the figure has a title, which is why
    that's part of the cache key rather than applied per figure.
    With `yanchor="bottom"`, a larger `y` means the legend's bottom edge sits
    higher up, so pushing `y` up to avoid a long title actually put the legend
    *above* the title instead of below it. `yanchor="top"` makes `y` the
    legend's own top edge (it extends downward from there), so a lower `y`
    than the title's `y` reliably reads as "legend below title".
    """
    t = _DARK_TOKENS
    axis = dict(
        showgrid=True,
        gridcolor=t["grid"],
        zeroline=False,
        linecolor=t["axis_line"],
        tickfont=dict(color=t["chart_tick"]),
        automargin=True,
        title_font=dict(color=t["chart_text"]),
    )
    layout = dict(
        font=dict(family=_CHART_FONT_FAMILY, size=12, color=t["chart_text"]),
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PAPER_BG,
        # The fallback for any chart that doesn't pass its own colors, so those
        # match the palette the rest of the dashboard uses instead of Plotly's
        # stock one.
        colorway=CATEGORICAL_PALETTE,
        xaxis=axis,
        yaxis=axis,
        polar=dict(
            bgcolor=CHART_PAPER_BG,
            radialaxis=dict(
                gridcolor=t["grid"],
                linecolor=t["axis_line"],
                tickfont=dict(color=t["chart_tick"]),
            ),
            angularaxis=dict(
                gridcolor=t["grid"],
                linecolor=t["axis_line"],
                tickfont=dict(color=t["chart_text"]),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.86 if has_title else 1.06,
            xanchor="left",
            x=0,
            font=dict(color=t["chart_text"]),
            bgcolor=t["legend_bg"],
            bordercolor=t["legend_border"],
        ),
        annotationdefaults=dict(font=dict(weight="bold", color=t["chart_annotation"])),
        # Plotly derives the hover box's fill from plot_bgcolor/paper_bgcolor,
        # which are now transparent -- an "x unified" tooltip would come out as
        # unreadable text floating over the chart. The trade-off of naming it
        # here is that every hover box gets the same themed fill instead of the
        # hovered trace's own color; the border still follows the trace.
        hoverlabel=dict(
            bgcolor=t["chart_bg"],
            font=dict(color=t["chart_text"], family=_CHART_FONT_FAMILY),
        ),
        # Same reason: the modebar's icon color is picked by contrast against
        # paper_bgcolor, and a transparent one reads as black -- which made the
        # icons invisible on the light theme's white page.
        modebar=dict(bgcolor=CHART_PAPER_BG, color=t["muted"], activecolor=t["accent"]),
    )
    if has_title:
        layout["title"] = dict(y=0.98, yanchor="top", x=0, xanchor="left")
    # Explicitly `plotly`, NOT `pio.templates.default`: importing streamlit
    # rewrites that default to its own "streamlit" template, whose colorway
    # and colorscales are sentinel near-black values (#000001..#000010) that
    # only mean anything once the frontend swaps them out. Inheriting those
    # here made every chart that falls back to the colorway render black.
    base = pio.templates["plotly"]
    return go.layout.Template(base).update(layout=layout)


def polish_figure_layout(fig, height: int | None = None, margin: dict | None = None) -> None:
    """Apply the unified dark chart styling to `fig`, in place."""
    t = _tokens()
    has_title = bool(fig.layout.title and fig.layout.title.text)
    if not has_title:
        # `charts.py`'s builders always pass `title=title` to `update_layout`,
        # even when the caller didn't supply one -- that explicit `None`
        # still serializes `layout.title` as `{}` (Plotly's layout objects
        # have schema defaults for every sub-field) rather than omitting the
        # key. Streamlit's "streamlit" chart theme reads `title.text`
        # whenever the key is present and renders the resulting `undefined`
        # as literal text, so drop the key entirely when there's no real
        # title to show.
        fig.layout.pop("title", None)

    # Margin stays out of the template: Plotly Express sets `margin.t` on the
    # figure itself, and a figure-level value wins over a template default,
    # so a templated margin would silently lose to px's own.
    #
    # The background and font are here for a related reason, and must NOT be
    # collapsed back into the template: Streamlit's frontend runs
    # `layoutWithThemeDefaults` over every Plotly spec -- including with
    # `theme=None` -- and it fills `paper_bgcolor`, `plot_bgcolor` and `font`
    # from *Streamlit's own* theme whenever the figure's layout doesn't carry
    # them. It reads the figure's layout, never the template, so a
    # template-only background lost to Streamlit's near-black `bgColor`
    # (which follows the browser/system setting, not our sidebar toggle) and
    # every chart rendered as a black slab. Spelling them out on the figure is
    # what makes the dashboard's own theme win.
    default_margin = dict(l=40, r=40, t=105 if has_title else 55, b=40)
    fig.update_layout(
        template=_figure_template(_active_theme_type(), has_title),
        margin=margin or default_margin,
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PAPER_BG,
        font=dict(family=_CHART_FONT_FAMILY, size=12, color=t["chart_text"]),
    )
    # Annotations already on the figure don't pick up the template's
    # `annotationdefaults`, so they still need an explicit pass.
    fig.update_annotations(font=dict(weight="bold", color=t["chart_annotation"]))
    try:
        # Don't force a single textfont color onto heatmap cells so Plotly's
        # intelligent text_auto per-cell contrast (black on light, white on dark)
        # continues to operate cleanly in both themes.
        fig.update_traces(
            selector=lambda tr: getattr(tr, "type", None) != "heatmap",
            textfont=dict(weight="bold", color=t["chart_annotation"]),
        )
    except Exception:
        # Not every trace type accepts a bold textfont weight; this is a
        # cosmetic best-effort, not a correctness concern.
        logger.debug(
            "polish_figure_layout: textfont update unsupported for this trace type", exc_info=True
        )
    if height:
        fig.update_layout(height=height)

    is_polar = (hasattr(fig.layout, "polar") and fig.layout.polar is not None) or any(
        getattr(tr, "type", None) in ("scatterpolar", "barpolar") for tr in fig.data
    )
    if is_polar:
        fig.update_layout(
            polar=dict(
                bgcolor=CHART_PAPER_BG,
                radialaxis=dict(
                    gridcolor=t["grid"],
                    linecolor=t["axis_line"],
                    tickfont=dict(color=t["chart_tick"]),
                ),
                angularaxis=dict(
                    gridcolor=t["grid"],
                    linecolor=t["axis_line"],
                    tickfont=dict(color=t["chart_text"]),
                ),
            )
        )
