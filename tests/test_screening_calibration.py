"""Tests for SLR screening calibration and stratified sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lake_research_map.transform.screening_calibration import (
    calibrate_screening_threshold,
    evaluate_screening_threshold,
    find_optimal_screening_threshold,
    generate_stratified_screening_sample,
    resolve_review_consensus,
    reviewer_agreement,
    validate_review_labels,
)


def test_generate_stratified_screening_sample_creates_balanced_strata():
    n = 80
    margins = np.linspace(-0.3, 0.3, n)
    df = pd.DataFrame(
        {
            "doi": [f"10.1000/{i}" for i in range(n)],
            "title": [f"Paper {i}" for i in range(n)],
            "year": [2020] * n,
            "venue": ["Test Journal"] * n,
            "relevance_margin": margins,
            "theme_label": ["Theme A"] * n,
        }
    )

    sample = generate_stratified_screening_sample(df, n_samples=20, seed=42)
    assert len(sample) == 20
    assert "manual_label" in sample.columns
    assert "stratum" in sample.columns
    # Check that multiple strata are populated
    assert sample["stratum"].nunique() >= 3


def test_evaluate_screening_threshold_metrics():
    # 4 in-scope, 4 out-of-scope
    y_true = np.array([True, True, True, True, False, False, False, False])
    # Margins: in-scope have positive, out-of-scope have negative
    margins = np.array([0.2, 0.1, 0.05, -0.01, 0.02, -0.1, -0.2, -0.3])
    # Cutoff at 0.0:
    # y_pred = [True, True, True, False, True, False, False, False]
    # TP: 3, FN: 1, FP: 1, TN: 3
    res = evaluate_screening_threshold(y_true, margins, threshold=0.0)
    assert res["tp"] == 3
    assert res["fn"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 3
    assert np.isclose(res["recall"], 3 / 4)
    assert np.isclose(res["precision"], 3 / 4)
    assert np.isclose(res["specificity"], 3 / 4)
    assert res["f1"] > 0
    assert res["f2"] > 0


def test_find_optimal_screening_threshold_guarantees_min_recall():
    y_true = np.array([True, True, True, True, False, False, False, False])
    margins = np.array([0.5, 0.4, 0.3, 0.1, -0.1, -0.2, -0.3, -0.4])

    opt = find_optimal_screening_threshold(y_true, margins, min_recall=1.0)
    assert opt["metrics_at_optimal"]["recall"] == 1.0
    # The lowest positive is 0.1, so threshold should be <= 0.1 to get 100% recall
    assert opt["optimal_threshold"] <= 0.1
    # Specificity should still be positive (excluding the negative ones)
    assert opt["metrics_at_optimal"]["specificity"] > 0.5


def test_sample_redistributes_quota_from_empty_stratum():
    df = pd.DataFrame(
        {
            "doi": [f"10.1000/{i}" for i in range(12)],
            "relevance_margin": [-0.2] + list(np.linspace(-0.08, 0.08, 11)),
        }
    )

    sample = generate_stratified_screening_sample(df, n_samples=10, seed=7)

    assert len(sample) == 10
    assert sample["doi"].is_unique
    assert {"reviewer", "protocol_version"}.issubset(sample.columns)


def test_review_validation_consensus_and_adjudication():
    labels = pd.DataFrame(
        {
            "doi": [
                "https://doi.org/10.1000/A",
                "10.1000/a",
                "10.1000/b",
                "10.1000/b",
                "10.1000/b",
                "10.1000/unknown",
            ],
            "reviewer": ["r1", "r2", "r1", "r2", "adjudicated", "r1"],
            "manual_label": ["1", "include", "include", "exclude", "exclude", "1"],
        }
    )

    valid, issues = validate_review_labels(labels, known_dois={"10.1000/a", "10.1000/b"})
    resolved = resolve_review_consensus(valid)

    assert "unknown_doi" in issues["code"].tolist()
    assert resolved.set_index("doi").loc["10.1000/a", "resolution"] == "consensus"
    assert resolved.set_index("doi").loc["10.1000/b", "resolution"] == "adjudicated"
    assert not bool(resolved.set_index("doi").loc["10.1000/b", "y_true"])


def test_reviewer_agreement_requires_support_and_reports_kappa():
    labels = pd.DataFrame(
        [
            {
                "doi": f"10.1000/{index}",
                "reviewer": reviewer,
                "manual_label": "include" if index % 2 else "exclude",
            }
            for index in range(24)
            for reviewer in ("r1", "r2")
        ]
    )

    agreement = reviewer_agreement(labels)

    assert agreement.iloc[0]["status"] == "ok"
    assert agreement.iloc[0]["raw_agreement"] == 1.0
    assert agreement.iloc[0]["kappa"] == 1.0


def test_calibration_uses_holdout_and_is_reproducible():
    n = 60
    labels = pd.DataFrame(
        {
            "doi": [f"10.1000/{index}" for index in range(n)],
            "y_true": [index >= n // 2 for index in range(n)],
            "resolved": True,
        }
    )
    scored = pd.DataFrame(
        {
            "doi": labels["doi"],
            "relevance_margin": np.r_[np.linspace(-0.5, -0.01, 30), np.linspace(0.01, 0.5, 30)],
        }
    )

    first = calibrate_screening_threshold(labels, scored, n_bootstrap=50, seed=9)
    second = calibrate_screening_threshold(labels, scored, n_bootstrap=50, seed=9)

    assert first["valid"] is True
    assert first["n_calibration"] + first["n_holdout"] == n
    assert first["metrics"]["recall"] == 1.0
    assert first["confidence_intervals"] == second["confidence_intervals"]
