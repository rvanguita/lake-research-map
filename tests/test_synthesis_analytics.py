"""Tests for methodological synthesis and scientometric maturity functions."""

from __future__ import annotations

import pathlib

import pandas as pd

from lake_research_map.dashboard.analytics import (
    author_impact_advanced_indices,
    author_m_quotient_analysis,
    benchmark_feeders_analysis,
    computational_solvers_analysis,
    mathematical_complexity_spectrum,
    objective_functions_taxonomy,
    optimization_methods_taxonomy,
    planning_time_horizons_analysis,
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
    assert "Genetic Algorithms (GA)" in methods_found
    assert "Mixed-Integer Linear Programming (MILP)" in methods_found
    assert "Particle Swarm Optimization (PSO)" in methods_found


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
    assert "Solar Generation (PV)" in cross.columns
    assert "Storage / Batteries" in cross.columns


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
    assert "Economic Costs" in set(summary["objective"])

    co_matrix = res["co_matrix"]
    assert not co_matrix.empty
    assert "Economic Costs" in co_matrix.index
    assert co_matrix.loc["Economic Costs", "Economic Costs"] > 0
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
    assert any("Stochastic" in p for p in paradigms["paradigm"])
    assert any("Robust" in p for p in paradigms["paradigm"])

    cross = res["cross_resources"]
    assert not cross.empty
    assert "Solar Generation (PV)" in cross.columns


def test_planning_time_horizons_analysis():
    df = _make_dummy_synthesis_data()
    res = planning_time_horizons_analysis(df)

    assert "horizons_df" in res
    horizons = res["horizons_df"]
    assert not horizons.empty
    assert "horizon" in horizons.columns
    assert any("Multistage" in h for h in horizons["horizon"])
    assert any("Operation" in h for h in horizons["horizon"])


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


def test_taxonomy_coverage_reports_what_the_chart_leaves_out():
    """WP-12: a frequency chart over matched articles only cannot describe a corpus.

    The unclassified share and the multi-label count are the two numbers that
    change how the chart above should be read: the first says how much is
    invisible, the second says the per-class counts are not a partition.
    """
    import pandas as pd

    from lake_research_map.dashboard.analytics import TAXONOMY_REGISTRY, taxonomy_coverage

    frame = pd.DataFrame(
        {
            "doi": ["10.1/a", "10.1/b", "10.1/c", "10.1/d"],
            "title": [
                "A genetic algorithm and MILP hybrid",  # two classes
                "Particle swarm optimization of feeders",  # one class
                "Cold chain logistics routing",  # none
                "Warehouse location planning",  # none
            ],
            "abstract": [""] * 4,
        }
    )

    stats = taxonomy_coverage(frame, TAXONOMY_REGISTRY["Optimization methods"])

    assert stats["population"] == 4
    assert stats["classified"] == 2
    assert stats["unclassified"] == 2
    assert stats["coverage"] == 0.5
    assert stats["multi_label"] == 1  # the GA+MILP paper
    assert sum(stats["per_class"].values()) > stats["classified"]  # not a partition

    empty = taxonomy_coverage(pd.DataFrame(), TAXONOMY_REGISTRY["Optimization methods"])
    assert empty["population"] == 0


def test_every_displayed_taxonomy_is_in_the_registry():
    """`WP-12` says *each* displayed taxonomy discloses coverage. The registry is
    what makes "each" enumerable rather than a promise, so it must stay in step
    with the page that renders them."""
    from lake_research_map.dashboard.analytics import TAXONOMY_REGISTRY

    assert set(TAXONOMY_REGISTRY) == {
        "Optimization methods",
        "Objective functions",
        "Uncertainty paradigms",
        "Planning horizons",
        "Computational solvers",
        "Mathematical complexity",
        "Benchmark feeders",
    }
    page = pathlib.Path("src/lake_research_map/dashboard/pages/synthesis.py").read_text(
        encoding="utf-8"
    )
    for name in TAXONOMY_REGISTRY:
        assert f'taxonomy_disclosure(df, "{name}")' in page, name


def test_precision_scores_only_decided_labels():
    """`ambiguous` is evidence about the class boundary, not a negative.

    Counting it as one would punish the regex for a reviewer's hesitation and
    quietly inflate the false-positive count.
    """
    import pandas as pd

    from lake_research_map.dashboard.analytics import (
        TAXONOMY_REGISTRY,
        taxonomy_precision_from_labels,
    )

    frame = pd.DataFrame(
        {
            "doi": ["10.1/a", "10.1/b", "10.1/c"],
            "title": [
                "A genetic algorithm approach",  # matched
                "A genetic algorithm approach",  # matched
                "Cold chain logistics",  # not matched
            ],
            "abstract": [""] * 3,
        }
    )
    labels = pd.DataFrame(
        {
            "subject_id": [
                "Genetic Algorithms (GA)::10.1/a",  # matched + present -> TP
                "Genetic Algorithms (GA)::10.1/b",  # matched + absent  -> FP
                "Genetic Algorithms (GA)::10.1/c",  # unmatched + present -> FN
                "Genetic Algorithms (GA)::10.1/a",  # ambiguous, excluded
            ],
            "label": ["present", "absent", "present", "ambiguous"],
        }
    )

    scored = taxonomy_precision_from_labels(
        frame, TAXONOMY_REGISTRY["Optimization methods"], labels
    )
    row = scored[scored["class"] == "Genetic Algorithms (GA)"].iloc[0]

    assert (row["true_positive"], row["false_positive"], row["false_negative"]) == (1, 1, 1)
    assert row["ambiguous"] == 1
    assert row["labelled"] == 3  # the ambiguous row is not counted as decided
    assert row["precision"] == 0.5 and row["recall"] == 0.5

    # No labels at all is the current state: an empty frame, not a crash.
    assert taxonomy_precision_from_labels(
        frame, TAXONOMY_REGISTRY["Optimization methods"], pd.DataFrame()
    ).empty
