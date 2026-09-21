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
