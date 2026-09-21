"""Tests for methodological synthesis and scientometric maturity functions."""

from __future__ import annotations

import pandas as pd

from lake_research_map.dashboard.analytics import (
    author_impact_advanced_indices,
    author_m_quotient_analysis,
    benchmark_feeders_analysis,
    citation_longevity_and_decay,
    computational_solvers_analysis,
    mathematical_complexity_spectrum,
    objective_functions_taxonomy,
    optimization_methods_taxonomy,
    planning_time_horizons_analysis,
    text_readability_and_stylometrics,
    uncertainty_paradigms_analysis,
)


def _make_dummy_synthesis_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "doi": [f"10.1016/j.epsr.202{i}" for i in range(10)],
            "title": [
                "Optimal distribution system planning using genetic algorithm and IEEE 33-bus",
                "MILP formulation for battery storage and IEEE 69-bus network",
                "Robust optimization for distributed generation in real distribution systems",
                "Multi-objective particle swarm optimization for EV charging and IEEE 33-bus",
                "Stochastic programming for renewable integration on IEEE 123-bus",
                "Second-order cone convex relaxation for distribution network reconfiguration",
                "Machine learning and reinforcement learning for microgrid planning and IEEE 33-bus",
                "A novel genetic algorithm for Brazilian real-world distribution systems",
                "Classical heuristic method for IEEE 33-bus capacitor placement",
                "High impact evergreen paper on distribution system planning with IEEE 69-bus",
            ],
            "abstract": [
                "This paper presents an optimal planning framework using genetic algorithm. The IEEE 33-bus radial system is utilized for validation. Results show significant reduction in investment costs and power loss reduction with improved reliability SAIDI.",
                "A mixed-integer linear programming (MILP) model is formulated in GAMS with CPLEX to size battery energy storage systems in the IEEE 69-bus system under multi-stage expansion. The approach guarantees global optimality and reduces voltage deviation.",
                "We propose a two-stage robust optimization approach implemented in MATLAB to accommodate high penetration of photovoltaic generation. Tested on a real-world distribution network considering emission reduction.",
                "A multi-objective particle swarm optimization (PSO) algorithm is implemented in Python to plan electric vehicle charging stations on the IEEE 33-bus system considering planning and operation with representative days.",
                "Stochastic programming with scenario-based Monte Carlo is adopted in GAMS to model uncertainty in wind power and load variation on the IEEE 123-bus benchmark network.",
                "Second-order cone programming (SOCP) relaxation is applied in MATLAB for radial distribution reconfiguration with renewable energy sources and resilience against extreme weather disasters.",
                "Deep reinforcement learning is trained to dispatch distributed generators in IEEE 33-bus systems under dynamic load profiles with power quality.",
                "An improved genetic algorithm is tested on a practical Brazilian distribution network with high reliability constraints and fuzzy logic.",
                "Capacitor placement is solved using differential evolution on standard test feeders for static planning.",
                "A foundational framework for distribution planning that influenced modern literature. Validated on IEEE 69-bus with exact economic cost minimization.",
            ],
            "year": [2012, 2022, 2023, 2021, 2024, 2022, 2023, 2018, 2010, 2011],
            "citation_count": [150, 45, 60, 35, 12, 28, 40, 25, 80, 250],
            "authors": [
                ["A. Silva", "B. Santos"],
                ["A. Silva", "C. Oliveira"],
                ["D. Das", "E. Smith"],
                ["A. Silva", "D. Das"],
                ["B. Santos", "F. Costa"],
                ["C. Oliveira", "G. Perez"],
                ["D. Das", "H. Lee"],
                ["A. Silva", "B. Santos"],
                ["I. Brown", "J. Wilson"],
                ["D. Das", "A. Silva"],
            ],
        }
    )


def test_optimization_methods_taxonomy():
    df = _make_dummy_synthesis_data()
    res = optimization_methods_taxonomy(df)

    assert "summary_df" in res
    assert "temporal_df" in res
    summary = res["summary_df"]
    assert not summary.empty
    assert "method" in summary.columns
    assert "articles" in summary.columns
    assert "pct_recent" in summary.columns
    assert "mean_citations" in summary.columns

    # Verify that GA, MILP, PSO are recognized
    methods_found = set(summary["method"])
    assert "Algoritmos Genéticos (GA)" in methods_found
    assert "Prog. Linear Inteira Mista (MILP)" in methods_found
    assert "Otimização por Enxame (PSO)" in methods_found


def test_benchmark_feeders_analysis():
    df = _make_dummy_synthesis_data()
    res = benchmark_feeders_analysis(df)

    assert "feeders_df" in res
    assert "cross_matrix" in res
    feeders = res["feeders_df"]
    cross = res["cross_matrix"]

    assert not feeders.empty
    assert "feeder" in feeders.columns
    assert "articles" in feeders.columns
    assert any("33-Bus" in f for f in feeders["feeder"])
    assert any("69-Bus" in f for f in feeders["feeder"])

    assert not cross.empty
    assert "Geração Solar (PV)" in cross.columns
    assert "Armazenamento / Baterias" in cross.columns


def test_author_impact_advanced_indices():
    df = _make_dummy_synthesis_data()
    auth_df = author_impact_advanced_indices(df, min_papers=2)

    assert not auth_df.empty
    assert "author" in auth_df.columns
    assert "h_index" in auth_df.columns
    assert "g_index" in auth_df.columns
    assert "e_index" in auth_df.columns
    assert "i10_index" in auth_df.columns

    # Fundamental scientometric rule: g >= h
    for _, r in auth_df.iterrows():
        assert r["g_index"] >= r["h_index"]
        assert r["e_index"] >= 0.0
        assert r["i10_index"] >= 0
        assert r["total_citations"] >= 0


def test_text_readability_and_stylometrics():
    df = _make_dummy_synthesis_data()
    res = text_readability_and_stylometrics(df)

    assert res["mean_fre"] >= 0.0
    assert res["mean_fre"] <= 100.0
    assert res["mean_fkgl"] >= 0.0
    assert 0.0 < res["mean_ttr"] <= 1.0

    assert not res["sample_df"].empty
    assert "fre" in res["sample_df"].columns
    assert "fkgl" in res["sample_df"].columns
    assert "ttr" in res["sample_df"].columns


def test_citation_longevity_and_decay():
    df = _make_dummy_synthesis_data()
    res = citation_longevity_and_decay(df)

    assert res["available"] is False
    assert "citing year" in res["reason"].lower()
    assert res["decay_curve"].empty
    assert res["evergreen_df"].empty


def test_objective_functions_taxonomy():
    df = _make_dummy_synthesis_data()
    res = objective_functions_taxonomy(df)

    assert "summary_df" in res
    assert "co_matrix" in res
    assert "multi_obj_ratio" in res
    assert "temporal_multiobj" in res

    summary = res["summary_df"]
    assert not summary.empty
    assert "objective" in summary.columns
    assert "articles" in summary.columns
    assert "Custos Econômicos" in set(summary["objective"])

    co_matrix = res["co_matrix"]
    assert not co_matrix.empty
    assert "Custos Econômicos" in co_matrix.index
    assert co_matrix.loc["Custos Econômicos", "Custos Econômicos"] > 0
    assert res["multi_obj_ratio"] >= 0.0


def test_uncertainty_paradigms_analysis():
    df = _make_dummy_synthesis_data()
    res = uncertainty_paradigms_analysis(df)

    assert "paradigms_df" in res
    assert "cross_resources" in res
    assert "temporal_paradigms" in res

    paradigms = res["paradigms_df"]
    assert not paradigms.empty
    assert "paradigm" in paradigms.columns
    assert any("Estocástico" in p for p in paradigms["paradigm"])
    assert any("Robusta" in p for p in paradigms["paradigm"])

    cross = res["cross_resources"]
    assert not cross.empty
    assert "Geração Solar (PV)" in cross.columns


def test_planning_time_horizons_analysis():
    df = _make_dummy_synthesis_data()
    res = planning_time_horizons_analysis(df)

    assert "horizons_df" in res
    horizons = res["horizons_df"]
    assert not horizons.empty
    assert "horizon" in horizons.columns
    assert any("Multi-Estágio" in h for h in horizons["horizon"])
    assert any("Operação" in h for h in horizons["horizon"])


def test_computational_solvers_analysis():
    df = _make_dummy_synthesis_data()
    res = computational_solvers_analysis(df)

    assert "solvers_df" in res
    assert "ecosystem_df" in res
    solvers = res["solvers_df"]
    assert not solvers.empty
    assert "tool" in solvers.columns
    assert any("GAMS" in t for t in solvers["tool"])
    assert any("MATLAB" in t for t in solvers["tool"])

    eco = res["ecosystem_df"]
    assert not eco.empty
    assert "category" in eco.columns


def test_mathematical_complexity_spectrum():
    df = _make_dummy_synthesis_data()
    res = mathematical_complexity_spectrum(df)

    assert "spectrum_df" in res
    assert "temporal_spectrum" in res
    spec = res["spectrum_df"]
    assert not spec.empty
    assert "complexity_class" in spec.columns
    assert any("MILP" in c for c in spec["complexity_class"])
    assert any("SOCP" in c for c in spec["complexity_class"])


def test_author_m_quotient_analysis():
    df = _make_dummy_synthesis_data()
    m_df = author_m_quotient_analysis(df, min_papers=2)

    assert not m_df.empty
    assert "author" in m_df.columns
    assert "h_index" in m_df.columns
    assert "m_quotient" in m_df.columns
    assert "career_span_years" in m_df.columns

    for _, r in m_df.iterrows():
        assert r["career_span_years"] >= 1
        assert r["m_quotient"] >= 0.0


def test_synthesis_page_import():
    from lake_research_map.dashboard.pages import synthesis

    assert hasattr(synthesis, "render")
    assert hasattr(synthesis, "_render_methods_tab")
    assert hasattr(synthesis, "_render_objectives_tab")
    assert hasattr(synthesis, "_render_uncertainty_tab")
    assert hasattr(synthesis, "_render_horizons_tab")
    assert hasattr(synthesis, "_render_feeders_tab")
    assert hasattr(synthesis, "_render_solvers_tab")
    assert not hasattr(synthesis, "_render_authors_tab")
    assert not hasattr(synthesis, "_render_longevity_and_text_tab")


def test_retired_page_controllers_are_removed():
    from lake_research_map.dashboard.pages import highlights, production

    assert not hasattr(production, "_venue_comparison")
    assert not hasattr(production, "_collaboration")
    assert not hasattr(highlights, "_collaboration_team_size")
    assert not hasattr(highlights, "_top_authors")
    assert not hasattr(highlights, "_venue_impact")
