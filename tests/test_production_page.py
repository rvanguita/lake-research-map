"""Headless smoke coverage for the Production and venues page."""

from streamlit.testing.v1 import AppTest


def test_production_page_renders_three_sections_and_reconciled_metrics():
    script = """
import pandas as pd

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.pages import production

articles = pd.DataFrame(
    {
        "doi": [f"10.1000/{i}" for i in range(5)],
        "year": [2020, 2020, 2021, 2021, 2022],
        "source": ["ieee", "elsevier", "ieee", "elsevier", "ieee"],
        "venue": ["Venue A", "Venue B", "Venue A", "Venue B", "Venue C"],
        "publication_category": ["journal", "conference", "review", "other", "legacy"],
    }
)
loaders.require_articles = lambda: articles
loaders.all_venue_qualis_map = lambda: pd.DataFrame(
    {
        "venue": ["Venue A", "Venue B", "Venue C"],
        "estrato": ["A1", "A2", "B1"],
    }
)

production.render()
"""

    app = AppTest.from_string(script).run(timeout=10)

    assert not app.exception
    assert [subheader.value for subheader in app.subheader] == [
        "Corpus summary and total trend",
        "Publication volume by year",
        "Publication types",
        "Venues and CAPES/Qualis",
        "Publication volume by year — CAPES/Qualis classification through B2",
    ]
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Total", "5"),
        ("Articles", "1"),
        ("Conference", "1"),
        ("Review", "1"),
        ("Other", "2"),
    ]
    assert app.segmented_control(key="prod_cumulative_scope").value == "Total"
