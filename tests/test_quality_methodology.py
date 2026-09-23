"""Tests for canonical population selection and quality-bias diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lake_research_map.dashboard import data, loaders
from lake_research_map.dashboard.analytics import metadata_coverage_matrix, pdf_selection_bias


def _gold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": ["10.1000/a", "10.1000/b"],
            "sources": [["ieee"], ["elsevier"]],
            "title": ["A", "B"],
            "authors": [["Ada"], ["Grace", "Edsger"]],
            "year": [2020, 2021],
            "venue": ["J1", "J2"],
            "keywords": [["grid"], []],
            "abstract": ["abstract", None],
            "citation_count": [2, None],
            "reference_count": [10, 12],
            "has_pdf": [True, False],
            "is_non_article": [False, False],
            "publication_category": ["journal", "journal"],
        }
    )


def test_gold_is_selected_when_analytical_contract_passes(monkeypatch):
    gold = _gold_frame()
    silver = gold.copy()
    monkeypatch.setattr(data, "load_articles", lambda layer: gold if layer == "gold" else silver)

    layer, selected, status = data.select_articles_layer()

    assert layer == "gold"
    assert selected.equals(gold)
    assert status["is_canonical"] is True


def test_invalid_gold_falls_back_with_reason(monkeypatch):
    gold = _gold_frame().drop(columns="reference_count")
    silver = _gold_frame()
    monkeypatch.setattr(
        data,
        "load_articles",
        lambda layer: gold if layer == "gold" else silver if layer == "silver" else pd.DataFrame(),
    )

    layer, _, status = data.select_articles_layer()

    assert layer == "silver"
    assert status["is_canonical"] is False
    assert "reference_count" in status["fallback_reasons"][0]


def test_article_normalization_derives_abstract_presence():
    normalized = loaders._normalize_article_frame(_gold_frame())

    assert normalized["source"].tolist() == ["ieee", "elsevier"]
    assert normalized["has_abstract"].tolist() == [True, False]


def test_metadata_coverage_uses_source_denominators():
    coverage = metadata_coverage_matrix(_gold_frame())
    keyword = coverage[coverage["field"] == "keywords"].set_index("source")

    assert keyword.loc["ieee", "coverage"] == 1.0
    assert keyword.loc["elsevier", "coverage"] == 0.0
    citation = coverage[coverage["field"] == "citation_count"].set_index("source")
    assert citation.loc["elsevier", "n_present"] == 0


def test_pdf_selection_bias_is_deterministic_and_keeps_missing_counts_missing():
    n = 40
    frame = pd.DataFrame(
        {
            "has_pdf": [True] * 20 + [False] * 20,
            "year": np.r_[np.arange(2000, 2020), np.arange(1990, 2010)],
            "citation_count": np.r_[np.arange(20), [np.nan] * 5, np.arange(15)],
            "reference_count": np.arange(n),
            "authors": [["a", "b"]] * 20 + [["a"]] * 20,
            "abstract": ["long abstract"] * 20 + ["short"] * 20,
            "keywords": [["x", "y"]] * 20 + [["x"]] * 20,
        }
    )

    first = pdf_selection_bias(frame, n_bootstrap=30, seed=5)
    second = pdf_selection_bias(frame, n_bootstrap=30, seed=5)

    pd.testing.assert_frame_equal(first, second)
    citation = first.set_index("metric").loc["citation_count"]
    assert citation["n_pdf"] == 20
    assert citation["n_no_pdf"] == 15
    assert np.isfinite(first.set_index("metric").loc["year", "smd"])
    assert np.isnan(first.set_index("metric").loc["team_size", "smd"])
    assert first.set_index("metric").loc["team_size", "status"] == "zero_variance"


def test_a_recorded_blocking_failure_refuses_canonical_gold(monkeypatch):
    gold = _gold_frame()
    monkeypatch.setattr(data, "load_articles", lambda layer: gold if layer == "gold" else gold)
    monkeypatch.setattr(data, "active_dataset_version", lambda: "v1")
    monkeypatch.setattr(
        data,
        "active_version_blocking_failures",
        lambda version: ("Recorded blocking check failed for this version: doi_unique.",),
    )

    layer, _, status = data.select_articles_layer()

    # A well-shaped frame is not enough: the pipeline's own verdict wins.
    assert layer == "silver"
    assert status["is_canonical"] is False
    assert any("doi_unique" in reason for reason in status["fallback_reasons"])


def test_only_the_latest_run_of_a_check_decides():
    results = pd.DataFrame(
        {
            "check_id": ["doi_unique", "doi_unique", "abstract_present"],
            "stage_run_id": [1, 2, 1],
            "passed": [False, True, False],
        }
    )

    assert data.blocking_failures(results) == (
        "Recorded blocking check failed for this version: abstract_present.",
    )
    assert data.blocking_failures(results.iloc[:0]) == ()


def test_blocking_failures_are_read_from_the_version_only(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from lake_research_map.db.gold_models import Base, QualityResult

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for version, run, check, severity, passed in [
            ("v1", 1, "doi_unique", "error", False),
            ("v1", 1, "abstract_rate", "warning", False),
            ("v0", 1, "title_present", "error", False),
        ]:
            session.add(
                QualityResult(
                    execution_id="e",
                    stage_run_id=run,
                    dataset_version_id=version,
                    stage="gold",
                    check_id=check,
                    severity=severity,
                    passed=passed,
                )
            )
        session.commit()
    monkeypatch.setattr(data, "get_engine", lambda layer: engine)

    assert data.active_version_blocking_failures("v1") == (
        "Recorded blocking check failed for this version: doi_unique.",
    )
    assert data.active_version_blocking_failures(None) == ()
