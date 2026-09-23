"""Application-wide headless coverage for every registered dashboard page.

WP-22. `test_dashboard_app.py` and `test_production_page.py` each exercise one
surface; this file walks the whole `PAGES` registry so a page cannot be added,
or a shared loader changed, without something rendering it. Two states are
covered because they fail differently: a populated corpus exercises the charts,
and an empty one exercises the guard clauses that are easy to break and never
seen in local development.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from lake_research_map.dashboard.app import PAGE_GROUPS, PAGES

PAGE_IDS = [url_path for _render, _title, _icon, url_path in PAGES]

# Enough columns to satisfy every page's `require_columns` guard at once.
_FIXTURE = """
import numpy as np
import pandas as pd
from lake_research_map.dashboard import loaders

rng = np.random.default_rng(0)
n = 40
years = list(range(2011, 2011 + n))
articles = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(n)],
    "title": [f"Distribution system planning study {i}" for i in range(n)],
    "abstract": ["Optimal planning of distribution networks with distributed generation."] * n,
    "year": [2011 + (i % 14) for i in range(n)],
    "source": ["ieee", "elsevier"] * (n // 2),
    "venue": [f"Journal {i % 5}" for i in range(n)],
    "authors": [["Author A", f"Author {chr(66 + i % 6)}"] for i in range(n)],
    "keywords": [["distribution planning", f"topic {i % 4}"] for i in range(n)],
    "citation_count": rng.integers(0, 90, size=n),
    "reference_count": rng.integers(5, 60, size=n),
    "has_pdf": [bool(i % 3) for i in range(n)],
    "publication_category": ["journal", "conference", "review", "other"] * (n // 4),
    "has_abstract": [True] * n,
})

chunks = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(n)],
    "chunk_type": ["abstract"] * n,
    "seq": [0] * n,
    "text": ["Optimal planning of distribution networks."] * n,
    "char_len": [42] * n,
    "has_embedding": [True] * n,
})

# `articles`/`filtered_articles` return (layer, frame), not a bare frame --
# a fixture that forgets that fails in the page rather than in the loader.
loaders.require_articles = lambda: articles
loaders.articles = lambda: ("gold", articles)
loaders.filtered_articles = lambda *a, **k: ("gold", articles)
loaders.filtered_chunks = lambda *a, **k: chunks
loaders.chunks = lambda: chunks
loaders.chunk_search_data = lambda: chunks
loaders.semantics = lambda: pd.DataFrame()
loaders.with_semantics = lambda frame: frame
loaders.all_venue_qualis_map = lambda: pd.DataFrame({"venue": [], "estrato": []})

author_rows = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(n)],
    "author": [f"Author {chr(65 + i % 6)}" for i in range(n)],
    "author_display": [f"Author {chr(65 + i % 6)}" for i in range(n)],
    "author_key": [f"author {chr(97 + i % 6)}" for i in range(n)],
    "year": [2011 + (i % 14) for i in range(n)],
    "source": ["ieee", "elsevier"] * (n // 2),
    "position": [1] * n,
    "citation_count": rng.integers(0, 90, size=n),
})

# Every loader that would otherwise open a MySQL connection. Without these the
# suite still passes in isolation but each page spends ~30 s waiting for a
# connection to time out, which is both slow and a false green: the page would
# be rendering the degraded path, not the one under test.
_empty = pd.DataFrame()
loaders.articles_by_layer = lambda: {}
loaders.author_table = lambda *a, **k: author_rows
loaders.author_year_matrix_cached = lambda *a, **k: _empty
loaders.alternative_projections = lambda *a, **k: {}
loaders.chunk_search_data = lambda: chunks
loaders.dataset_versions = lambda: _empty
loaders.duplicate_overrides = lambda: _empty
loaders.duplicate_pairs = lambda: _empty
loaders.keyword_forecasts = lambda *a, **k: {}
loaders.layer_funnel = lambda: _empty
loaders.pipeline_executions = lambda: _empty
loaders.pipeline_runs = lambda: _empty
loaders.publication_state = lambda: _empty
loaders.quality_results = lambda: _empty
loaders.rejected_records = lambda: _empty
loaders.row_counts = lambda: _empty
loaders.search_configs = lambda: _empty
loaders.semantic_novelty_scores = lambda *a, **k: _empty
loaders.semantic_stability = lambda *a, **k: None
loaders.source_changes = lambda: _empty
from lake_research_map.dashboard.forecasting import fit_and_forecast, yearly_counts
_forecast = fit_and_forecast(yearly_counts(articles))
loaders.volume_forecast = lambda *a, **k: _forecast
"""

_EMPTY_FIXTURE = """
import pandas as pd
import streamlit as st
from lake_research_map.dashboard import loaders

def _stop():
    st.warning("No data found in the medallion layers yet.")
    st.stop()

empty = pd.DataFrame()
loaders.require_articles = _stop
loaders.articles = lambda: ("bronze", empty)
loaders.filtered_articles = lambda *a, **k: ("bronze", empty)
loaders.filtered_chunks = lambda *a, **k: empty
loaders.chunks = lambda: empty
loaders.chunk_search_data = lambda: empty
loaders.semantics = lambda: empty
loaders.with_semantics = lambda frame: frame
loaders.all_venue_qualis_map = lambda: pd.DataFrame({"venue": [], "estrato": []})
loaders.articles_by_layer = lambda: {}
loaders.author_table = lambda *a, **k: empty
loaders.author_year_matrix_cached = lambda *a, **k: empty
loaders.alternative_projections = lambda *a, **k: {}
loaders.dataset_versions = lambda: empty
loaders.duplicate_overrides = lambda: empty
loaders.duplicate_pairs = lambda: empty
loaders.keyword_forecasts = lambda *a, **k: {}
loaders.layer_funnel = lambda: empty
loaders.pipeline_executions = lambda: empty
loaders.pipeline_runs = lambda: empty
loaders.publication_state = lambda: empty
loaders.quality_results = lambda: empty
loaders.rejected_records = lambda: empty
loaders.row_counts = lambda: empty
loaders.search_configs = lambda: empty
loaders.semantic_novelty_scores = lambda *a, **k: empty
loaders.semantic_stability = lambda *a, **k: None
loaders.source_changes = lambda: empty
from lake_research_map.dashboard.forecasting import fit_and_forecast
_forecast = fit_and_forecast(pd.Series(dtype=float))
loaders.volume_forecast = lambda *a, **k: _forecast
"""


def _script(fixture: str, url_path: str) -> str:
    module = {
        "overview": "overview",
        "output-over-time": "production",
        "topics": "topics",
        "highlights": "highlights",
        "researchers": "researchers",
        "synthesis": "synthesis",
        "forecast": "forecasting",
        "semantics": "semantics",
        "quality": "quality",
        "pipeline-layers": "pipeline_layers",
    }[url_path]
    return f"{fixture}\nfrom lake_research_map.dashboard.pages import {module}\n{module}.render()\n"


def test_registry_is_flat_and_unique():
    """The URL slugs are public bookmarks, so a collision is a broken link."""
    assert len(PAGES) == sum(len(group) for group in PAGE_GROUPS.values())
    assert len(PAGE_IDS) == len(set(PAGE_IDS))
    assert set(PAGE_IDS) == {
        "overview",
        "output-over-time",
        "topics",
        "highlights",
        "researchers",
        "synthesis",
        "forecast",
        "semantics",
        "quality",
        "pipeline-layers",
    }


@pytest.mark.parametrize("url_path", PAGE_IDS)
def test_every_page_renders_with_a_populated_corpus(url_path):
    app = AppTest.from_string(_script(_FIXTURE, url_path)).run(timeout=60)
    assert not app.exception, f"{url_path}: {app.exception}"


@pytest.mark.parametrize("url_path", PAGE_IDS)
def test_every_page_survives_an_empty_corpus(url_path):
    """The degraded path must stop cleanly, not raise.

    `require_articles` calls `st.stop()`, which AppTest surfaces as a normal
    end of script rather than an exception; anything else means a page reads
    the frame before guarding it.
    """
    app = AppTest.from_string(_script(_EMPTY_FIXTURE, url_path)).run(timeout=60)
    assert not app.exception, f"{url_path}: {app.exception}"
