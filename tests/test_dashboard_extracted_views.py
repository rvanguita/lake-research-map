"""Headless renders of the views whose aggregation moved out of the pages (2026-11-06).

The page-registry smoke tests open each page on its default tab, which never
reaches these views; each is rendered here directly, against a corpus with
semantic signals, so the page side of every extracted function (and its cached
loader wrapper) runs at least once.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

_FIXTURE = """
import numpy as np
import pandas as pd
import streamlit as st
from apps.dashboard import loaders

# A fresh cache per script: the wrappers are keyed by the (empty) filter
# signature, so another test's frame would otherwise be served from cache.
st.cache_data.clear()

rng = np.random.default_rng(7)
n = 120
years = [2000 + (i % 25) for i in range(n)]
articles = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(n)],
    "title": [f"Distribution planning study {i}" for i in range(n)],
    "year": years,
    "source": ["ieee", "elsevier"] * (n // 2),
    "venue": [f"Journal {i % 5}" for i in range(n)],
    "authors": [[f"Author {i % 7}", f"Author {(i + 3) % 11}"] for i in range(n)],
    "keywords": [
        ["distribution planning", "smart grid" if y > 2012 else "radial feeders"]
        for y in years
    ],
    "citation_count": rng.integers(0, 90, size=n),
    "reference_count": rng.integers(5, 60, size=n),
    "has_pdf": [bool(i % 3) for i in range(n)],
    "abstract": ["Optimal planning of distribution networks."] * n,
})
semantic = pd.DataFrame({
    "doi": articles["doi"],
    "map_x": rng.normal(size=n),
    "map_y": rng.normal(size=n),
    "theme_id": [i % 3 for i in range(n)],
    "theme_label": [f"Theme {i % 3}" for i in range(n)],
    "relevance_score": rng.uniform(0.2, 0.8, size=n),
})
author_rows = articles[["doi", "authors"]].explode("authors").rename(columns={"authors": "author"})
author_rows["author_key"] = author_rows["author"].str.lower()
author_rows["author_display"] = author_rows["author"]

loaders.require_articles = lambda: articles
loaders.filtered_articles = lambda *a, **k: ("gold", articles)
loaders.semantics = lambda: semantic
loaders.with_semantics = lambda frame: frame.merge(semantic, on="doi", how="left")
loaders.author_table = lambda *a, **k: author_rows
"""

_VIEWS = {
    "rising_falling": "from apps.dashboard.pages import topics\ntopics._rising_falling()\n",
    "team_size": (
        "from apps.dashboard.pages import researchers\n"
        "researchers._render_team_collaboration_stats(articles)\n"
    ),
    "cognitive_distance": (
        "from apps.dashboard.pages import researchers\nresearchers._cognitive_distance_analysis()\n"
    ),
    "themes": (
        "from apps.dashboard.pages import semantics\n"
        "semantics._themes(loaders.with_semantics(articles))\n"
    ),
    "pdf_bias": "from apps.dashboard.pages import quality\nquality._pdf_selection_bias()\n",
}


@pytest.mark.parametrize("view", list(_VIEWS))
def test_extracted_view_renders(view):
    app = AppTest.from_string(_FIXTURE + _VIEWS[view]).run(timeout=60)

    assert not app.exception, f"{view}: {app.exception}"
    assert app.get("plotly_chart"), f"{view} drew no chart"


def test_team_size_trend_axis_says_mean_not_median():
    app = AppTest.from_string(_FIXTURE + _VIEWS["team_size"]).run(timeout=60)

    specs = [chart.proto.spec for chart in app.get("plotly_chart")]
    assert any("Mean authors per article" in spec for spec in specs)
    assert not any("Median authors" in spec for spec in specs)


def test_themes_drift_caption_names_the_derived_windows():
    app = AppTest.from_string(_FIXTURE + _VIEWS["themes"]).run(timeout=60)

    captions = " ".join(caption.value for caption in app.caption)
    # 2000-2024 split into three equal-count windows; no fixed era literal.
    assert "3 chronological windows" in captions
    assert "2019–2026" not in captions
