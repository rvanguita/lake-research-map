"""Headless smoke coverage for the new Streamlit review workflow."""

from streamlit.testing.v1 import AppTest


def test_screening_calibration_panel_empty_state():
    script = """
import pandas as pd
from lake_research_map.dashboard.pages.semantics import _screening_calibration_panel
from lake_research_map.dashboard.pages.pipeline_layers import _render_contract_status

scored = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(8)],
    "title": [f"Paper {i}" for i in range(8)],
    "year": [2020] * 8,
    "venue": ["Journal"] * 8,
    "relevance_margin": [-0.3, -0.2, -0.08, -0.01, 0.01, 0.08, 0.2, 0.3],
    "theme_label": ["Theme"] * 8,
})
_screening_calibration_panel(scored)
results = pd.DataFrame([
    {
        "dataset_version_id": "v1",
        "stage": "silver",
        "check_id": "silver.doi_unique",
        "severity": "error",
        "passed": False,
        "observed": 1,
        "expected": 0,
        "checked_at": "2026-09-21T12:00:00",
    },
    {
        "dataset_version_id": "v1",
        "stage": "raw",
        "check_id": "raw.snapshot_coverage",
        "severity": "error",
        "passed": True,
        "observed": 3,
        "expected": 3,
        "checked_at": "2026-09-21T12:00:00",
    },
])
_render_contract_status(results)
"""

    app = AppTest.from_string(script).run(timeout=10)

    assert not app.exception
    assert app.download_button
    assert app.file_uploader
    assert any("independent decisions" in info.value for info in app.info)
    assert len(app.metric) == 3
    assert app.error
    assert app.dataframe


def test_citation_determinants_panel_renders_family_and_age_sensitivity():
    """WP-15: the Impact panel must show how the family was chosen and whether
    the associations survive a different age specification."""
    script = """
import numpy as np
import pandas as pd
from lake_research_map.dashboard.pages.highlights import _citation_determinants_glm_view

rng = np.random.default_rng(3)
n = 80
articles = pd.DataFrame({
    "doi": [f"10.1000/{i}" for i in range(n)],
    "year": rng.integers(2012, 2024, size=n),
    "citation_count": rng.integers(0, 60, size=n),
    "reference_count": rng.integers(5, 50, size=n),
    "authors": [["A", "B"], ["A", "B", "C"]] * (n // 2),
    "source": ["ieee"] * (n // 2) + ["elsevier"] * (n // 2),
})
_citation_determinants_glm_view(articles)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    captions = " ".join(caption.value for caption in app.caption)
    assert "selected by AIC" in captions
    # The expander body renders the age-sensitivity table alongside the
    # coefficient table, so both are present.
    assert len(app.dataframe) >= 2
    markdown = " ".join(block.value for block in app.markdown)
    assert "Sensitivity to the age specification" in markdown
