"""Tests for advanced statistics, bibliometric laws, and network analytics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from lake_research_map.dashboard.analytics import (
    age_normalized_citations,
    benjamini_hochberg,
    bradford_zones,
    citation_determinants_glm,
    coauthorship_community_detection,
    fit_heavy_tail_distributions,
    graph_advanced_metrics,
    lotka_law_analysis,
    mann_kendall_trend,
    network_null_model_diagnostics,
    semantic_stability_diagnostics,
    zipf_law_analysis,
)


def test_fit_heavy_tail_distributions():
    # Power-law synthetic sample
    rng = np.random.default_rng(42)
    # Pareto draws: (1 - U)^(-1 / (alpha - 1))
    u = rng.uniform(0.01, 0.99, size=100)
    cites = (1.0 - u) ** (-1.0 / 1.5)
    res = fit_heavy_tail_distributions(cites)
    assert res["valid"] is True
    assert res["best_fit"] in ("power_law", "log_normal")
    assert "models" in res
    assert "power_law" in res["models"]
    assert res["models"]["power_law"]["alpha"] > 1.0


def test_age_normalized_citations():
    df = pd.DataFrame(
        {
            "citation_count": [100, 10, 50, 5],
            "year": [2010, 2025, 2010, 2025],
        }
    )
    norm = age_normalized_citations(df, observation_year=2026)
    assert "citation_rate_annual" in norm.columns
    assert "cohort_citation_percentile" in norm.columns
    # 10 citations in 2025 (age ~2, rate ~5.0) vs 50 in 2010 (age ~17, rate ~2.9)
    assert norm.loc[1, "citation_rate_annual"] > norm.loc[2, "citation_rate_annual"]


def test_mann_kendall_trend_strictly_increasing():
    series = np.array([1, 2, 4, 7, 11, 16, 22, 29, 37])
    res = mann_kendall_trend(series)
    assert res["trend"] == "growing"
    assert res["s"] > 0
    assert res["p_value"] < 0.05


def test_benjamini_hochberg_is_monotone_and_preserves_missing_values():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_semantic_stability_recovers_separated_clusters():
    rng = np.random.default_rng(7)
    left = rng.normal(-3, 0.1, size=(20, 4))
    right = rng.normal(3, 0.1, size=(20, 4))
    matrix = np.vstack([left, right])
    labels = np.array([0] * 20 + [1] * 20)
    projection = matrix[:, :2]

    result = semantic_stability_diagnostics(matrix, labels, projection, n_bootstrap=5, seed=3)

    assert result["valid"] is True
    assert result["bootstrap_ari_mean"] > 0.95
    # `projection` drops two of the four dimensions, so some neighbours are
    # genuinely lost: the bound proves trustworthiness responds to well-separated
    # structure, not that a truncation preserves it perfectly.
    assert result["projection_trustworthiness"] > 0.9


def test_network_null_model_preserves_degree_comparison():
    graph = nx.watts_strogatz_graph(20, 4, 0.05, seed=2)

    result = network_null_model_diagnostics(graph, n_simulations=5, seed=2)

    assert result["valid"] is True
    assert result["simulations"] == 5
    assert 0 <= result["empirical_p_value"] <= 1


def test_mann_kendall_trend_flat():
    series = np.array([5, 5, 5, 5, 5, 5])
    res = mann_kendall_trend(series)
    assert res["trend"] == "stable"
    assert res["s"] == 0
    assert res["p_value"] == 1.0


def test_lotka_law_analysis():
    # 100 authors with 1 paper, 25 with 2 papers, 11 with 3 papers (inverse square)
    author_counts = pd.Series([1] * 100 + [2] * 25 + [3] * 11 + [4] * 6)
    res = lotka_law_analysis(author_counts)
    assert res["alpha"] > 1.0
    assert res["r2"] > 0.8
    assert not res["table"].empty


def test_bradford_zones():
    # Top venue has 50 papers, others fewer
    venues = ["Venue A"] * 50 + ["Venue B"] * 30 + ["Venue C"] * 20 + ["Venue D"] * 10
    df = pd.DataFrame({"venue": venues})
    res = bradford_zones(df, n_zones=3)
    assert len(res["zone_summary"]) == 3
    assert res["multiplier_mean"] > 0


def test_coauthorship_community_detection_and_metrics():
    # Two disjoint triangles (cliques)
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])
    communities = coauthorship_community_detection(g)
    assert len(communities) == 6
    # 1, 2, 3 in one community, 4, 5, 6 in another
    assert communities[1] == communities[2] == communities[3]
    assert communities[4] == communities[5] == communities[6]
    assert communities[1] != communities[4]

    metrics = graph_advanced_metrics(g)
    assert metrics["density"] > 0
    assert metrics["avg_clustering"] > 0


def test_citation_determinants_glm():
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "year": rng.integers(2010, 2025, size=n),
            "citation_count": rng.integers(0, 100, size=n),
            "reference_count": rng.integers(5, 50, size=n),
            "authors": [["A", "B"]] * n,
            "source": ["ieee"] * (n // 2) + ["elsevier"] * (n // 2),
        }
    )
    res = citation_determinants_glm(df, observation_year=2026)
    assert res["valid"] is True
    assert res["family"] in {
        "poisson",
        "negative_binomial",
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }
    assert res["n_used"] == n
    assert res["coverage"] == 1.0
    assert len(res["features"]) == len(res["coefficients"]) == len(res["irr"])
    assert len(res["irr_lower"]) == len(res["irr_upper"]) == len(res["features"])
    assert "tamanho_equipe" not in res["features"]  # constant predictor is not identifiable
    assert res["condition_number"] > 0
    assert set(res["candidate_aic"]) >= {"poisson"}
    assert 0 <= res["observed_zero_fraction"] <= 1
    assert res["influential_count"] >= 0

    # WP-15: the family is chosen by AIC over every candidate that converged,
    # not by the old dispersion > 1.5 rule.
    assert res["family_selection"] == "aic"
    assert res["family"] == min(res["candidate_aic"], key=res["candidate_aic"].get)
    assert res["zero_inflated_status"] in {"fitted", "did_not_converge"}

    # A selected family must be able to state its own uncertainty: a fit whose
    # Hessian could not be inverted returns NaN intervals and is not eligible.
    assert all(np.isfinite(value) for value in res["p_values"])
    assert all(np.isfinite(value) for value in res["irr_lower"])
    assert all(np.isfinite(value) for value in res["irr_upper"])

    # WP-15: log(age + 1) as a fixed offset is an assumption, so the exposure
    # choice is reported rather than hidden.
    assert len(res["age_specifications"]) >= 2
    assert {spec["specification"] for spec in res["age_specifications"]} <= {
        "offset_log_age",
        "covariate_log_age",
        "covariate_linear_age",
    }
    for spec in res["age_specifications"]:
        assert set(spec["coefficients"]) == set(res["features"])
        assert np.isfinite(spec["aic"])
    assert isinstance(res["age_specification_signs_agree"], bool)

    with_missing = df.copy()
    with_missing.loc[:4, "reference_count"] = np.nan
    incomplete = citation_determinants_glm(with_missing, observation_year=2026)
    assert incomplete["n_used"] == n - 5
    assert incomplete["coverage"] == (n - 5) / n
    assert incomplete["missingness"]["reference_count"] == 5 / n


def test_zipf_law_analysis():
    titles = [
        "Optimal planning of distribution systems considering distributed generation",
        "A multi-objective framework for distribution network expansion planning",
        "Stochastic planning of electric distribution systems with high penetration of renewable energy",
        "Resilience-oriented planning of active distribution networks against extreme weather events",
    ] * 25
    df = pd.DataFrame({"title": titles})
    res = zipf_law_analysis(df)
    assert res["valid"] is True
    assert res["gamma"] > 0.0
    assert res["r_squared"] > 0.4
    assert res["vocab_size"] >= 10
    assert not res["top_words_df"].empty
    assert "Rank (r)" in res["top_words_df"].columns


def test_zipf_law_analysis_empty():
    df = pd.DataFrame({"title": []})
    res = zipf_law_analysis(df)
    assert res["valid"] is False


def test_graph_advanced_metrics_centralities_and_small_world():
    g = nx.watts_strogatz_graph(n=20, k=4, p=0.1, seed=42)
    for u, v in g.edges():
        g[u][v]["weight"] = 1

    metrics = graph_advanced_metrics(g)
    assert "closeness" in metrics
    assert "pagerank" in metrics
    assert len(metrics["closeness"]) == 20
    assert len(metrics["pagerank"]) == 20
    assert metrics["avg_path_length"] is not None
    assert metrics["avg_path_length"] > 0
    assert metrics["small_world_sigma"] is not None


def test_detect_structural_breaks():
    from lake_research_map.dashboard.analytics import detect_structural_breaks

    # Regime 1: mean 10, Regime 2: mean 50
    years = np.arange(2000, 2020)
    values = np.array(
        [10, 12, 11, 9, 10, 11, 10, 12, 10, 11, 48, 52, 50, 51, 49, 53, 50, 52, 51, 50]
    )
    series = pd.Series(values, index=years)

    res = detect_structural_breaks(series)
    assert res["has_break"] is True
    assert res["break_year"] == 2010
    assert res["f_stat"] > 10.0
    assert res["post_mean"] > res["pre_mean"]

    # WP-16: the breakpoint is searched, so the reported p-value is empirical.
    # It cannot resolve below 1/(draws+1), and asserting otherwise would be
    # asking a permutation test for precision it does not have.
    assert res["p_value"] < 0.05
    assert res["p_value"] >= res["p_value_resolution"]
    assert res["p_value_naive"] < 0.001  # the F-table value this replaces
    assert res["bootstrap_samples"] == 200


def test_searched_breakpoint_p_value_controls_false_positives():
    """The F table is the wrong null for a maximum taken over every split.

    Read against it, pure noise looks like a regime change about a third of
    the time. The permutation p-value is what brings that back near nominal,
    and that gap is the whole reason this statistic is bootstrapped.
    """
    from lake_research_map.dashboard.analytics import detect_structural_breaks

    rng = np.random.default_rng(3)
    trials = 60
    naive_hits = 0
    bootstrap_hits = 0
    for seed in range(trials):
        res = detect_structural_breaks(rng.normal(0, 1, 20), n_bootstrap=100, seed=seed)
        naive_hits += res["p_value_naive"] < 0.05
        bootstrap_hits += res["p_value"] < 0.05

    assert naive_hits / trials > 0.20  # the defect: far above the nominal 5%
    assert bootstrap_hits / trials < 0.15  # near nominal, allowing for 60 trials
    assert bootstrap_hits < naive_hits


def test_conceptual_atypicality_analysis():
    from lake_research_map.dashboard.analytics import conceptual_atypicality_analysis

    df = pd.DataFrame(
        [
            {"keywords": ["solar", "planning"], "citation_count": 100, "doi": "d1"},
            {"keywords": ["solar", "planning"], "citation_count": 80, "doi": "d2"},
            {"keywords": ["storage", "ev"], "citation_count": 150, "doi": "d3"},
            {"keywords": ["storage", "ev"], "citation_count": 20, "doi": "d4"},
            {
                "keywords": ["solar", "ev"],
                "citation_count": 200,
                "doi": "d5",
            },  # atypical combination
            {"keywords": ["planning", "storage"], "citation_count": 30, "doi": "d6"},
        ]
    )
    res = conceptual_atypicality_analysis(df, top_n_keywords=10)
    assert res["valid"] is True
    assert not res["articles_df"].empty
    assert "median_z" in res["articles_df"].columns
    assert "is_atypical" in res["articles_df"].columns


def test_citation_determinants_glm_reports_when_no_zero_inflated_fit_converges():
    """A corpus with no zeros gives the zero-inflation part nothing to explain.

    The panel must still be able to say which families were on the table, so
    `zero_inflated_status` is always populated and the selected family always
    carries finite uncertainty -- never a NaN interval from a fit whose Hessian
    could not be inverted.
    """
    rng = np.random.default_rng(11)
    n = 60
    df = pd.DataFrame(
        {
            "year": rng.integers(2015, 2024, size=n),
            # Strictly positive: there is no zero mass to inflate.
            "citation_count": rng.integers(5, 40, size=n),
            "reference_count": rng.integers(5, 50, size=n),
            "authors": [["A", "B"], ["A", "B", "C"]] * (n // 2),
            "source": ["ieee"] * (n // 2) + ["elsevier"] * (n // 2),
        }
    )
    res = citation_determinants_glm(df, observation_year=2026)

    assert res["valid"] is True
    assert res["observed_zero_fraction"] == 0.0
    assert res["zero_inflated_status"] in {"fitted", "did_not_converge"}
    assert "poisson" in res["candidate_aic"]
    assert res["family"] == min(res["candidate_aic"], key=res["candidate_aic"].get)
    assert all(np.isfinite(value) for value in res["irr_lower"])
    assert all(np.isfinite(value) for value in res["irr_upper"])
    # Influence always names the fit it was measured on, because a zero-inflated
    # fit has no hat matrix and falls back to the Poisson GLM.
    assert res["influence_basis"] in {
        "poisson",
        "negative_binomial",
        "unavailable",
    }


def test_citation_determinants_glm_rejects_undersized_sample():
    df = pd.DataFrame(
        {
            "year": [2020] * 10,
            "citation_count": list(range(10)),
            "reference_count": list(range(10, 20)),
            "authors": [["A"]] * 10,
            "source": ["ieee"] * 10,
        }
    )
    res = citation_determinants_glm(df, observation_year=2026)
    assert res["valid"] is False
    assert res["n_used"] == 10
    assert "20" in res["warning"]
    # Nothing was fitted, so the panel must not find diagnostics to display.
    assert "age_specifications" not in res
    assert "candidate_aic" not in res
    assert res["coefficients"] == []


def test_mann_kendall_serial_correction_deflates_a_random_walk():
    """A random walk has no trend, but Mann-Kendall reads one anyway.

    That is the whole point of the Hamed-Rao correction: the independence
    assumption manufactures significance out of autocorrelation. White noise
    and a genuine trend must both come through it untouched.
    """
    rng = np.random.default_rng(5)

    walk = np.cumsum(rng.normal(0, 1, 40))
    res = mann_kendall_trend(walk)
    assert res["serial_correction_factor"] > 1.5
    assert res["p_value_serial_corrected"] > res["p_value_independent"]
    # `p_value` stays the independent test unless the caller opts in.
    assert res["p_value"] == res["p_value_independent"]
    assert mann_kendall_trend(walk, serial_correction=True)["p_value"] == pytest.approx(
        res["p_value_serial_corrected"]
    )

    noise = rng.normal(0, 1, 40)
    assert mann_kendall_trend(noise)["serial_correction_factor"] == pytest.approx(1.0)

    real_trend = np.arange(40) + rng.normal(0, 1, 40)
    corrected = mann_kendall_trend(real_trend, serial_correction=True)
    assert corrected["trend"] == "growing"
    assert corrected["p_value"] < 0.01


def test_mann_kendall_short_series_reports_a_neutral_correction():
    # Below ten points the lag correlations are too noisy to correct with, so
    # the factor must be exactly neutral rather than a guess.
    res = mann_kendall_trend(np.array([1.0, 3.0, 2.0, 5.0, 4.0]))
    assert res["serial_correction_factor"] == 1.0
    assert res["p_value_serial_corrected"] == pytest.approx(res["p_value_independent"])


def test_linear_slope_with_ci_reports_the_spread_a_ranking_hides():
    from lake_research_map.dashboard.analytics import linear_slope_with_ci

    clean = linear_slope_with_ci(np.arange(10), 2.0 * np.arange(10))
    assert clean["slope"] == pytest.approx(2.0)
    assert clean["ci_low"] <= clean["slope"] <= clean["ci_high"]

    # Three noisy points have a slope but no usable precision: the interval
    # has to straddle zero, which is exactly what the chart now shows.
    noisy = linear_slope_with_ci([1, 2, 3], [5, 1, 6])
    assert noisy["ci_low"] < 0 < noisy["ci_high"]
    assert noisy["p_value"] > 0.05

    degenerate = linear_slope_with_ci([1, 2], [1, 2])
    assert np.isnan(degenerate["slope"])  # fewer than three points
