"""Availability contracts for frontiers methods and Kleinberg burst tests."""

import pandas as pd

from lake_research_map.dashboard.analytics import technological_burst_detection


def _sample_corpus_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "doi": "10.1109/TPWRS.2015.001",
                "title": "Robust Optimization for Microgrid Islanding with BESS and Solar PV",
                "abstract": "We develop a two-stage robust optimization model for microgrids with battery storage.",
                "year": 2015,
                "citation_count": 120,
                "reference_count": 35,
                "authors": ["A. Padilha", "J. Liu", "D. Das"],
                "theme_label": "DG Optimization",
                "relevance_margin": 0.8,
                "license": "CC-BY",
                "document_type": "Journal Article",
                "venue": "IEEE Trans. Power Syst.",
            },
            {
                "id": 2,
                "doi": "10.1109/TPWRS.2023.002",
                "title": "Deep Learning and Vehicle-to-Grid Integration for Hosting Capacity",
                "abstract": "Deep reinforcement learning applied to v2g electric vehicles to expand hosting capacity.",
                "year": 2023,
                "citation_count": 28,
                "reference_count": 42,
                "authors": ["J. Liu", "X. Wang"],
                "theme_label": "Renewable & Storage",
                "relevance_margin": 1.2,
                "license": "CC BY-NC",
                "document_type": "Journal Article",
                "venue": "IEEE Trans. Smart Grid",
            },
            {
                "id": 3,
                "doi": "10.1016/j.ijepes.2010.003",
                "title": "Early Analytical Formulation of Convex SOCP Relaxation",
                "abstract": "Second-order cone programming formulation for distribution network loss minimization.",
                "year": 2010,
                "citation_count": 95,
                "reference_count": 18,
                "authors": ["M. Costa"],
                "theme_label": "Network Topology",
                "relevance_margin": 0.4,
                "license": None,
                "document_type": "Conference Paper",
                "venue": "Int. J. Electr. Power",
            },
            {
                "id": 4,
                "doi": "10.1109/TSG.2021.004",
                "title": "Peer-to-Peer Energy Trading and Blockchain in Active Distribution Networks",
                "abstract": "P2P transactive energy with smart meters and active distribution adn resilience.",
                "year": 2021,
                "citation_count": 45,
                "reference_count": 38,
                "authors": ["R. Silva", "T. Souza", "E. Santos", "F. Lima"],
                "theme_label": "Electricity Markets",
                "relevance_margin": 0.6,
                "license": "Closed",
                "document_type": "Journal Article",
                "venue": "IEEE Trans. Smart Grid",
            },
            {
                "id": 5,
                "doi": "10.1109/TPWRS.2008.005",
                "title": "Classical Radial Distribution Feeders Load Flow Algorithm",
                "abstract": "Backward forward sweep method on benchmark feeders without distributed energy.",
                "year": 2008,
                "citation_count": 60,
                "reference_count": 15,
                "authors": ["D. Das", "A. Padilha"],
                "theme_label": "Network Topology",
                "relevance_margin": -0.2,
                "license": None,
                "document_type": "Journal Article",
                "venue": "IEEE Trans. Power Del.",
            },
        ]
    )


def test_technological_burst_detection() -> None:
    df = _sample_corpus_df()
    # Replicate multiple rows to trigger counts
    big_df = pd.concat([df] * 6, ignore_index=True)
    res = technological_burst_detection(big_df)

    assert "burst_timeline" in res
    assert "total_bursts" in res
    assert res["total_bursts"] >= 1
    assert "Technology / Concept" in res["burst_timeline"].columns
    assert "Burst's Beginning" in res["burst_timeline"].columns

    # Test empty dataframe
    empty_res = technological_burst_detection(pd.DataFrame())
    assert empty_res["total_bursts"] == 0
