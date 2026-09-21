"""Tests for advanced statistics, bibliometric laws, and network analytics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from lake_research_map.dashboard.analytics import (
    age_normalized_citations,
    bradford_zones,
    citation_determinants_glm,
    coauthorship_community_detection,
    fit_heavy_tail_distributions,
    graph_advanced_metrics,
    lotka_law_analysis,
    mann_kendall_trend,
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
    norm = age_normalized_citations(df, current_year=2026)
    assert "citation_rate_annual" in norm.columns
    assert "cohort_citation_percentile" in norm.columns
    # 10 citations in 2025 (age ~2, rate ~5.0) vs 50 in 2010 (age ~17, rate ~2.9)
    assert norm.loc[1, "citation_rate_annual"] > norm.loc[2, "citation_rate_annual"]


def test_mann_kendall_trend_strictly_increasing():
    series = np.array([1, 2, 4, 7, 11, 16, 22, 29, 37])
    res = mann_kendall_trend(series)
    assert res["trend"] == "crescendo"
    assert res["s"] > 0
    assert res["p_value"] < 0.05
    assert res["slope"] > 0


def test_mann_kendall_trend_flat():
    series = np.array([5, 5, 5, 5, 5, 5])
    res = mann_kendall_trend(series)
    assert res["trend"] == "estável"
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
    res = citation_determinants_glm(df)
    assert res["valid"] is True
    assert res["family"] in {"poisson", "binomial_negativa"}
    assert res["n_used"] == n
    assert res["coverage"] == 1.0
    assert len(res["features"]) == len(res["coefficients"]) == len(res["irr"])
    assert len(res["irr_lower"]) == len(res["irr_upper"]) == len(res["features"])
    assert "tamanho_equipe" not in res["features"]  # constant predictor is not identifiable

    with_missing = df.copy()
    with_missing.loc[:4, "reference_count"] = np.nan
    incomplete = citation_determinants_glm(with_missing)
    assert incomplete["n_used"] == n - 5
    assert incomplete["coverage"] == (n - 5) / n


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
    assert "Posto (r)" in res["top_words_df"].columns


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
    assert res["p_value"] < 0.001
    assert res["post_mean"] > res["pre_mean"]


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


def test_venue_semantic_clusters():
    from lake_research_map.dashboard.analytics import venue_semantic_clusters

    dois = [f"10.1/{i}" for i in range(8)]
    venues = (
        ["IEEE TPWRS"] * 2
        + ["IEEE TSTE"] * 2
        + ["Applied Energy"] * 2
        + ["Electric Power Systems Research"] * 2
    )
    df = pd.DataFrame(
        {"doi": dois, "venue": venues, "citation_count": [10, 20, 15, 25, 30, 40, 5, 12]}
    )

    # Synthetic 8x4 embeddings
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(8, 4)).astype(np.float32)

    res = venue_semantic_clusters(df, embs, dois, n_clusters=2)
    assert not res.empty
    assert "cluster_id" in res.columns
    assert len(res) == 4
