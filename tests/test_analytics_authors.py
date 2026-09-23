import pandas as pd

from lake_research_map.dashboard.analytics import (
    analyze_coauthorship_partners,
    author_productivity_trend,
    author_year_matrix,
    cumulative_researchers,
    explode_authors,
    explode_authors_with_position,
    gini_coefficient,
    output_impact_correlation,
    researchers_by_year,
)


def _articles_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"authors": ["A. Silva"], "year": 2020, "doi": "10.1/1", "citation_count": 10},
            {"authors": ["A. Silva"], "year": 2021, "doi": "10.1/2", "citation_count": 12},
            {"authors": ["A. Silva"], "year": 2022, "doi": "10.1/3", "citation_count": 8},
            {"authors": ["B. Costa"], "year": 2020, "doi": "10.1/4", "citation_count": 1},
            {
                "authors": ["A. Silva", "B. Costa"],
                "year": 2022,
                "doi": "10.1/5",
                "citation_count": None,
            },
        ]
    )


def test_author_year_matrix_sorted_descending_by_total():
    matrix = author_year_matrix(_articles_df())

    assert matrix.iloc[0]["total"] >= matrix.iloc[1]["total"]
    # Silva: 2020, 2021, 2022, 2022 (coauthored) = 4 distinct DOIs.
    assert matrix.iloc[0]["total"] == 4
    # Costa: 2020, 2022 = 2 distinct DOIs.
    assert matrix.iloc[1]["total"] == 2


def test_author_year_matrix_year_columns_and_totals_consistent():
    matrix = author_year_matrix(_articles_df())
    year_cols = [
        c for c in matrix.columns if c not in ("author", "total", "ieee_total", "elsevier_total")
    ]

    assert year_cols == sorted(year_cols)
    for _, row in matrix.iterrows():
        assert row["total"] == sum(row[c] for c in year_cols)


def test_author_year_matrix_empty_when_no_authors():
    empty = pd.DataFrame({"year": [2020, 2021]})
    matrix = author_year_matrix(empty)
    assert matrix.empty


def test_author_year_matrix_splits_ieee_and_elsevier_totals():
    df = pd.DataFrame(
        [
            {"authors": ["J. Liu"], "year": 2020, "doi": "10.1/1", "source": "ieee"},
            {"authors": ["J. Liu"], "year": 2021, "doi": "10.1/2", "source": "ieee"},
            {"authors": ["Junyong Liu"], "year": 2022, "doi": "10.1/3", "source": "elsevier"},
            {"authors": ["B. Costa"], "year": 2020, "doi": "10.1/4", "source": "elsevier"},
        ]
    )
    matrix = author_year_matrix(df).set_index("author")

    # "J. Liu" (ieee) and "Junyong Liu" (elsevier) fold to the same canonical
    # author -- total must be the sum of both sources' contributions.
    liu = matrix.loc["Junyong Liu"]
    assert liu["ieee_total"] == 2
    assert liu["elsevier_total"] == 1
    assert liu["total"] == liu["ieee_total"] + liu["elsevier_total"] == 3

    costa = matrix.loc["B. Costa"]
    assert costa["ieee_total"] == 0
    assert costa["elsevier_total"] == 1
    assert costa["total"] == 1


def test_author_year_matrix_source_columns_default_to_zero_without_source():
    matrix = author_year_matrix(_articles_df())
    assert (matrix["ieee_total"] == 0).all()
    assert (matrix["elsevier_total"] == 0).all()


def test_gini_coefficient_perfect_equality_is_zero():
    assert gini_coefficient(pd.Series([5, 5, 5, 5])) == 0.0


def test_gini_coefficient_max_inequality_approaches_one():
    # One author has everything, the rest have nothing.
    gini = gini_coefficient(pd.Series([0, 0, 0, 100]))
    assert gini > 0.7


def test_gini_coefficient_handles_small_or_empty_input():
    assert gini_coefficient(pd.Series([])) == 0.0
    assert gini_coefficient(pd.Series([5])) == 0.0


def test_author_productivity_trend_classifies_growth_and_decline():
    matrix = pd.DataFrame(
        [
            {"author": "Growing", "total": 5, "2018": 0, "2019": 1, "2020": 2, "2021": 4},
            {"author": "Declining", "total": 5, "2018": 4, "2019": 2, "2020": 1, "2021": 0},
            {"author": "OnePoint", "total": 1, "2018": 0, "2019": 0, "2020": 0, "2021": 1},
        ]
    )
    trend = author_productivity_trend(matrix, top_n=10)
    by_author = trend.set_index("author")

    assert by_author.loc["Growing", "trend"] == "growing"
    assert by_author.loc["Declining", "trend"] == "falling"
    assert by_author.loc["OnePoint", "trend"] == "insufficient data"
    # Sorted descending by total.
    assert list(trend["total"]) == sorted(trend["total"], reverse=True)


def test_output_impact_correlation_ignores_null_citation_rows():
    df = pd.DataFrame(
        [
            {"author_display": "A", "citation_count": 10},
            {"author_display": "A", "citation_count": 20},
            {"author_display": "B", "citation_count": None},
            {"author_display": "B", "citation_count": None},
            {"author_display": "C", "citation_count": 5},
            {"author_display": "C", "citation_count": 5},
        ]
    )
    result = output_impact_correlation(df, "citation_count")
    assert result["n"] >= 2
    assert result["pearson"] is None or -1.0 <= result["pearson"] <= 1.0


def test_output_impact_correlation_returns_none_below_minimum_sample():
    df = pd.DataFrame(
        [
            {"author_display": "A", "citation_count": 10},
            {"author_display": "B", "citation_count": 20},
        ]
    )
    result = output_impact_correlation(df, "citation_count")
    assert result["pearson"] is None
    assert result["spearman"] is None


def _mixed_source_authors_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"authors": ["J. Liu"], "year": 2019, "doi": "b1", "source": "ieee"},
            {"authors": ["Junyong Liu"], "year": 2020, "doi": "b2", "source": "elsevier"},
            {"authors": ["Junyong Liu"], "year": 2020, "doi": "b3", "source": "ieee"},
            {"authors": ["B. Costa"], "year": 2020, "doi": "b4", "source": "elsevier"},
            {"authors": ["B. Costa"], "year": 2021, "doi": "b5", "source": "elsevier"},
        ]
    )


def test_researchers_by_year_counts_distinct_authors_not_rows():
    by_year = researchers_by_year(_mixed_source_authors_df()).set_index("year")

    # 2020: Liu (both sources) and Costa (elsevier) are active -- Liu counts
    # once in "total" despite appearing in both sources that year.
    assert by_year.loc[2020, "ieee"] == 1
    assert by_year.loc[2020, "elsevier"] == 2
    assert by_year.loc[2020, "total"] == 2
    assert by_year.loc[2019, "total"] == 1
    assert by_year.loc[2021, "total"] == 1


def test_researchers_by_year_empty_when_no_authors():
    empty = pd.DataFrame({"year": [2020, 2021]})
    assert researchers_by_year(empty).empty


def test_cumulative_researchers_counts_each_author_once_at_first_appearance():
    cum = cumulative_researchers(_mixed_source_authors_df()).set_index("year")

    # Liu's first-ever appearance is 2019 (ieee); Costa's is 2020 (elsevier).
    # A researcher active across multiple years must not be re-counted.
    assert list(cum["total"]) == [1, 2, 2]
    assert list(cum["ieee"]) == [1, 1, 1]
    assert list(cum["elsevier"]) == [0, 2, 2]


def test_cumulative_researchers_is_non_decreasing():
    cum = cumulative_researchers(_mixed_source_authors_df())
    for col in ("ieee", "elsevier", "total"):
        assert (cum[col].diff().dropna() >= 0).all()


def _multi_author_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "authors": ["A. Silva", "B. Costa", "C. Souza"],
                "year": 2020,
                "doi": "c1",
                "source": "ieee",
            },
            {"authors": ["C. Souza"], "year": 2021, "doi": "c2", "source": "elsevier"},
        ]
    )


def test_explode_authors_with_position_orders_match_byline():
    exploded = explode_authors_with_position(_multi_author_df())
    by_doi = exploded[exploded["doi"] == "c1"].sort_values("position")
    assert list(by_doi["author"]) == ["A. Silva", "B. Costa", "C. Souza"]
    assert list(by_doi["position"]) == [0, 1, 2]


def test_explode_authors_output_unaffected_by_position_tracking():
    with_position = explode_authors_with_position(_multi_author_df())
    plain = explode_authors(_multi_author_df())
    assert "position" not in plain.columns
    assert list(plain["author"]) == list(with_position["author"])
    assert len(plain) == len(with_position)


def test_researchers_by_year_max_position_excludes_third_author():
    df = _multi_author_df()
    all_positions = researchers_by_year(df).set_index("year")
    lead_only = researchers_by_year(df, max_position=1).set_index("year")

    # 2020: all 3 authors counted without a position filter; only the first
    # two (A, B) count when restricted to 1st/2nd position.
    assert all_positions.loc[2020, "total"] == 3
    assert lead_only.loc[2020, "total"] == 2
    # C. Souza is 1st author of their own solo 2021 article, so they're still
    # counted there even though they were 3rd author in 2020.
    assert lead_only.loc[2021, "total"] == 1


def test_cumulative_researchers_max_position_excludes_third_author():
    df = _multi_author_df()
    cum = cumulative_researchers(df, max_position=1)
    assert cum["total"].iloc[-1] == 3  # A, B (2020) + C (2021, as 1st author there)


def test_analyze_coauthorship_partners_computes_network_and_global_collaborators():
    import networkx as nx

    data = [
        {"author_display": "Junyong Liu", "doi": "10.1/1"},
        {"author_display": "Xiaochun Zhang", "doi": "10.1/1"},
        {"author_display": "Junyong Liu", "doi": "10.1/2"},
        {"author_display": "Xiaochun Zhang", "doi": "10.1/2"},
        {"author_display": "Yangyang Liu", "doi": "10.1/2"},
        {"author_display": "Junyong Liu", "doi": "10.1/3"},
        {"author_display": "Yangyang Liu", "doi": "10.1/3"},
        {"author_display": "Junyong Liu", "doi": "10.1/4"},
        {"author_display": "Yun Wei Li", "doi": "10.1/4"},
        {"author_display": "External Author", "doi": "10.1/4"},
        {"author_display": "Xiaochun Zhang", "doi": "10.1/5"},
        {"author_display": "Yangyang Liu", "doi": "10.1/5"},
    ]
    author_rows = pd.DataFrame(data)

    graph = nx.Graph()
    graph.add_edge("Junyong Liu", "Xiaochun Zhang", weight=2)
    graph.add_edge("Junyong Liu", "Yangyang Liu", weight=2)
    graph.add_edge("Junyong Liu", "Yun Wei Li", weight=1)
    graph.add_edge("Xiaochun Zhang", "Yangyang Liu", weight=2)

    df = analyze_coauthorship_partners(graph, author_rows)
    assert not df.empty

    liu = df[df["author"] == "Junyong Liu"].iloc[0]
    assert liu["articles"] == 4
    # In network: 3 unique coauthors (Xiaochun Zhang, Yangyang Liu, Yun Wei Li)
    assert liu["network_unique_count"] == 3
    # 2 recurrent (weight >= 2)
    assert liu["network_recurrent_count"] == 2
    assert "Xiaochun Zhang (2)" in liu["network_recurrent_names"]
    assert "Yangyang Liu (2)" in liu["network_recurrent_names"]
    assert "Yun Wei Li" not in liu["network_recurrent_names"]
    # 1 occasional (weight == 1)
    assert liu["network_occasional_count"] == 1
    assert "Yun Wei Li" in liu["network_occasional_names"]

    # Global in corpus: 4 unique coauthors (including External Author)
    assert liu["global_unique_count"] == 4
    assert liu["global_recurrent_count"] == 2
    assert "External Author (1)" in liu["global_top_partners"]


def test_analyze_coauthorship_partners_handles_empty_graph():
    import networkx as nx

    graph = nx.Graph()
    author_rows = pd.DataFrame([{"author_display": "A", "doi": "10.1/1"}])
    df = analyze_coauthorship_partners(graph, author_rows)
    assert df.empty
    assert "network_unique_count" in df.columns
    assert "global_unique_count" in df.columns


def test_network_null_model_reports_assortativity_and_robustness():
    """WP-19: a bare assortativity coefficient says nothing without a null,
    and either removal figure alone is just a graph size."""
    import warnings

    import networkx as nx

    from lake_research_map.dashboard.analytics import network_null_model_diagnostics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hubbed = network_null_model_diagnostics(
            nx.barabasi_albert_graph(60, 2, seed=1), n_simulations=15, seed=1
        )
        even = network_null_model_diagnostics(
            nx.watts_strogatz_graph(60, 4, 0.3, seed=1), n_simulations=15, seed=1
        )

    assert hubbed["observed_assortativity"] is not None
    assert hubbed["assortativity_null_mean"] is not None
    assert hubbed["assortativity_z_score"] is not None

    # A hub-dominated graph must lose more of its giant component to targeted
    # removal than an evenly-connected one does. That gap is the whole claim.
    hub_gap = hubbed["robustness_random"] - hubbed["robustness_targeted"]
    even_gap = even["robustness_random"] - even["robustness_targeted"]
    assert hub_gap > even_gap
    assert 0.0 <= hubbed["robustness_targeted"] <= 1.0
    assert hubbed["robustness_removed"] == 6


def test_periodized_ties_separate_new_from_returning_collaborations():
    """A static recurrent-edge count cannot tell recruitment from consolidation."""
    from lake_research_map.dashboard.analytics import periodized_collaboration_ties

    rows = []
    for doi, year, names in [
        ("d1", 2010, ["A", "B"]),
        ("d2", 2012, ["A", "B"]),
        ("d3", 2015, ["A", "B"]),
        ("d4", 2016, ["A", "C"]),
        ("d5", 2019, ["A", "B"]),
        ("d6", 2020, ["D", "E"]),
        ("d7", 2021, ["D", "E"]),
    ]:
        rows.extend({"doi": doi, "year": year, "author_display": name} for name in names)

    ties = periodized_collaboration_ties(pd.DataFrame(rows), n_periods=3)

    assert list(ties["period"]) == ["2010–2015", "2016–2019", "2020–2021"]
    # A-B is new in the first period and returning in the second.
    assert ties.iloc[0]["new_ties"] == 1 and ties.iloc[0]["repeated_ties"] == 0
    assert ties.iloc[1]["repeated_ties"] == 1
    assert (ties["new_ties"] + ties["repeated_ties"] == ties["total_ties"]).all()

    assert periodized_collaboration_ties(pd.DataFrame()).empty
