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
