"""Tests for strategic scientometric analytics functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lake_research_map.dashboard.analytics import (
    callon_strategic_diagram,
    geographic_collaboration_stats,
    keyword_cooccurrence_graph,
    multivariate_correlation_matrix,
    shannon_thematic_entropy,
    thematic_centroids_similarity,
    thematic_radar_metrics,
)


def _make_dummy_strategic_data():
    dois = [f"10.1000/{i}" for i in range(12)]
    df = pd.DataFrame(
        {
            "id": list(range(1, 13)),
            "doi": dois,
            "theme_label": [
                "Tema A · Sub1",
                "Tema A · Sub1",
                "Tema A · Sub1",
                "Tema A · Sub1",
                "Tema B · Sub2",
                "Tema B · Sub2",
                "Tema B · Sub2",
                "Tema B · Sub2",
                "Tema C · Sub3",
                "Tema C · Sub3",
                "Tema C · Sub3",
                "Tema C · Sub3",
            ],
            "citation_count": [10, 20, 5, 15, 50, 60, 40, 30, 2, 4, 1, 3],
            "reference_count": [25, 30, 15, 20, 40, 45, 35, 50, 10, 12, 8, 14],
            "year": [2018, 2019, 2020, 2022, 2021, 2022, 2023, 2024, 2010, 2012, 2014, 2015],
            "relevance_score": [
                0.85,
                0.88,
                0.82,
                0.90,
                0.75,
                0.78,
                0.70,
                0.72,
                0.40,
                0.45,
                0.38,
                0.42,
            ],
            "relevance_margin": [
                0.35,
                0.38,
                0.32,
                0.40,
                0.20,
                0.25,
                0.18,
                0.22,
                -0.10,
                -0.05,
                -0.15,
                -0.08,
            ],
            "abstract": [
                f"Distribution system planning with renewable resources number {i}." * (i + 1)
                for i in range(12)
            ],
            "authors": [["Author A", "Author B"]] * 6 + [["Author C", "Author D", "Author E"]] * 6,
            "keywords": [
                ["solar pv", "energy storage", "uncertainty"],
                ["solar pv", "energy storage", "battery"],
                ["solar pv", "uncertainty", "robust optimization"],
                ["energy storage", "battery", "resilience"],
                ["wind power", "uncertainty", "microgrid"],
                ["wind power", "microgrid", "resilience"],
                ["microgrid", "islanded", "resilience"],
                ["wind power", "energy storage", "microgrid"],
                ["logistics", "vehicle routing", "fleet"],
                ["logistics", "supply chain", "depot"],
                ["fleet", "vehicle routing", "depot"],
                ["supply chain", "logistics", "fleet"],
            ],
            "countries": [
                ["China", "USA"],
                ["China"],
                ["USA"],
                ["Brazil", "Spain"],
                ["Brazil"],
                ["Canada", "USA"],
                ["Iran"],
                ["China", "UK"],
                [],
                ["USA"],
                ["Germany", "France"],
                ["Germany"],
            ],
        }
    )

    rng = np.random.default_rng(42)
    # Distinct directional embeddings per theme
    v_a = rng.normal(1.0, 0.1, (4, 384))
    v_b = rng.normal(-1.0, 0.1, (4, 384))
    v_c = rng.normal(0.0, 0.1, (4, 384))
    embs = np.vstack([v_a, v_b, v_c]).astype(np.float32)

    return df, dois, embs


def test_callon_strategic_diagram():
    df, dois, embs = _make_dummy_strategic_data()
    res = callon_strategic_diagram(df, embs, dois)
    themes_df = res["themes_df"]

    assert res["available"] is False
    assert "keyword equivalence" in res["reason"].lower()
    assert themes_df.empty


def test_keyword_cooccurrence_graph():
    df, _, _ = _make_dummy_strategic_data()
    graph = keyword_cooccurrence_graph(df, min_cooccurrence=2, top_n_keywords=10)

    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0
    assert len(graph["edges"]) > 0

    first_node = graph["nodes"][0]
    assert "id" in first_node
    assert "community" in first_node
    assert "x" in first_node
    assert "y" in first_node

    first_edge = graph["edges"][0]
    assert "source" in first_edge
    assert "target" in first_edge
    assert first_edge["weight"] >= 2
    assert 0.0 <= first_edge["jaccard"] <= 1.0


def test_thematic_centroids_similarity():
    df, dois, embs = _make_dummy_strategic_data()
    sim_matrix = thematic_centroids_similarity(df, embs, dois)

    assert not sim_matrix.empty
    assert sim_matrix.shape == (3, 3)

    # Diagonal must be 1.0 (self-cosine)
    for i in range(3):
        assert np.isclose(sim_matrix.iloc[i, i], 1.0, atol=1e-4)

    # Must be symmetric
    assert np.allclose(sim_matrix.values, sim_matrix.values.T, atol=1e-4)


def test_thematic_radar_metrics():
    df, _, _ = _make_dummy_strategic_data()
    radar = thematic_radar_metrics(df)

    assert not radar.empty
    assert len(radar) == 3

    for col in (
        "pct_recent_score",
        "mean_citations_score",
        "mean_refs_score",
        "mean_margin_score",
        "mean_authors_score",
    ):
        assert col in radar.columns
        assert (radar[col] >= 0.0).all()
        assert (radar[col] <= 100.0).all()


def test_multivariate_correlation_matrix():
    df, _, _ = _make_dummy_strategic_data()
    corr = multivariate_correlation_matrix(df)

    assert not corr.empty
    assert corr.shape[0] == corr.shape[1]
    assert "Ano" in corr.columns
    assert "Citações" in corr.columns
    assert "Score Relevância" in corr.columns

    # Diagonals must be 1.0
    for i in range(len(corr)):
        assert np.isclose(corr.iloc[i, i], 1.0)


def test_shannon_thematic_entropy():
    df, _, _ = _make_dummy_strategic_data()
    ent_df = shannon_thematic_entropy(df, min_year=2000)

    assert not ent_df.empty
    assert "shannon_entropy" in ent_df.columns
    assert "gini_simpson" in ent_df.columns
    assert (ent_df["shannon_entropy"] >= 0.0).all()
    assert ((ent_df["gini_simpson"] >= 0.0) & (ent_df["gini_simpson"] <= 1.0)).all()


def test_geographic_collaboration_stats():
    df, _, _ = _make_dummy_strategic_data()
    geo = geographic_collaboration_stats(df)

    assert geo["total_with_country"] > 0
    assert geo["intl_papers"] > 0
    assert 0.0 <= geo["intl_pct"] <= 100.0
    assert not geo["country_counts"].empty
    assert not geo["collaboration_pairs"].empty
    assert "collaborations" in geo["collaboration_pairs"].columns
