"""Headless smoke coverage for the Production and venues page.

The page owns every CAPES/Qualis view (they moved here from Topics on
2026-11-06), so the tests also walk the Qualis view selector and the
missing-workbook state.
"""

import json

from streamlit.testing.v1 import AppTest

QUALIS_AVAILABLE = "loaders.qualis_available = lambda: True\n"
QUALIS_MISSING = "loaders.qualis_available = lambda: False\n"


def _script(qualis_stub: str) -> str:
    return _SCRIPT.replace("# QUALIS_STUB\n", qualis_stub)


_SCRIPT = """
import pandas as pd

from apps.dashboard import loaders
from apps.dashboard.pages import production

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
# QUALIS_STUB

production.render()
"""


def test_production_page_renders_three_sections_and_reconciled_metrics():
    app = AppTest.from_string(_script(QUALIS_AVAILABLE)).run(timeout=10)

    assert not app.exception
    assert [subheader.value for subheader in app.subheader] == [
        "Corpus summary and total trend",
        "Publication volume by year",
        "Publication types",
        "Venues and CAPES/Qualis",
        "Publication volume by year — CAPES/Qualis classification through B2",
        "🎓 CAPES/Qualis classification",
    ]
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Total", "5"),
        ("Articles", "1"),
        ("Conference", "1"),
        ("Review", "1"),
        ("Other", "2"),
        # The default Qualis view (detailed table) adds its own three cards.
        ("📚 Periodicals classified", "3/3"),
        ("🥇 A1 journals", "1"),
        ("📄 Articles in A1 journals", "2"),
    ]
    assert app.segmented_control(key="prod_cumulative_scope").value == "Total"
    assert app.segmented_control(key="qualis_view_selector").value == "Detailed table"


def test_every_qualis_view_renders_on_the_production_page():
    app = AppTest.from_string(_script(QUALIS_AVAILABLE)).run(timeout=10)
    expected = {
        "Totals by stratum": "Total publications by classification",
        "Cumulative evolution": "Cumulative by classification",
        "A1–A3 subset": "A1–A3 journals",
    }
    for view, heading in expected.items():
        app.segmented_control(key="qualis_view_selector").set_value(view).run(timeout=10)
        assert not app.exception, f"{view}: {app.exception}"
        assert app.subheader[-1].value == heading


def test_the_a1_a3_view_draws_one_ranking_not_two():
    app = AppTest.from_string(_script(QUALIS_AVAILABLE)).run(timeout=10)
    app.segmented_control(key="qualis_view_selector").set_value("A1–A3 subset").run(timeout=10)

    # Annual volume, the tier-coloured ranking, and the source x tier split.
    # The source-stacked "Top 15" bars that duplicated the ranking are gone.
    charts = app.get("plotly_chart")
    a1_a3 = [chart for chart in charts if "production_qualis_top_tier" in chart.proto.id]
    assert len(a1_a3) == 3
    titles = [
        json.loads(chart.proto.spec)["layout"].get("title", {}).get("text") for chart in a1_a3
    ]
    assert titles.count("Top 15 A1–A3 journals by article count") == 1
    assert not any("and source" in str(title) for title in titles)


def test_qualis_views_say_once_that_the_workbook_is_missing():
    from apps.dashboard.qualis import SOURCE_UNAVAILABLE_MESSAGE

    app = AppTest.from_string(_script(QUALIS_MISSING)).run(timeout=10)

    assert not app.exception
    assert [info.value for info in app.info].count(SOURCE_UNAVAILABLE_MESSAGE) == 1
    assert "🎓 CAPES/Qualis classification" not in [s.value for s in app.subheader]
