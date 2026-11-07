"""Pure functions extracted from the Topics, Researchers, Quality and Production pages.

Each was inline page code until 2026-11-06; these pin what the pages render
without starting Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.dashboard.analytics.authors import (
    cognitive_distance_analysis,
    cognitive_distance_by_article,
    team_size_by_year,
    team_size_summary,
)
from apps.dashboard.analytics.counts import pdf_selection_bias, pdf_selection_bias_views
from apps.dashboard.analytics.vocabulary import (
    keyword_share_slopes,
    keyword_trend_frame,
    keyword_trend_tests,
    rising_falling_keywords,
)
from apps.dashboard.qualis import (
    NOT_CLASSIFIED,
    qualis_strata,
    strata_volume_by_year,
    top_tier_summary,
)

YEARS = list(range(2010, 2020))


def _keyword_year_frame() -> pd.DataFrame:
    """A keyword x year frame with one rising, one falling and one rare term.

    "filler" keeps every year's total at 20 mentions, so a term's share is its
    count / 20 and the slopes are known in advance.
    """
    rows = []
    for i, year in enumerate(YEARS):
        rising, falling = 1 + i, 10 - i
        rows += [("smart grid", year)] * rising
        rows += [("radial feeders", year)] * falling
        rows += [("filler", year)] * (20 - rising - falling - (1 if year == 2015 else 0))
        if year == 2015:
            rows.append(("rare term", year))
    return pd.DataFrame(rows, columns=["keyword", "year"])


def test_keyword_share_slopes_rank_rising_and_falling_terms_and_skip_rare_ones():
    slopes = keyword_share_slopes(_keyword_year_frame(), min_occurrences=15).set_index("keyword")

    assert "rare term" not in slopes.index
    # +1 mention of 20 per year is +5 percentage points per year, exactly.
    assert slopes.loc["smart grid", "slope"] == pytest.approx(5.0)
    assert slopes.loc["radial feeders", "slope"] == pytest.approx(-5.0)
    assert slopes.loc["smart grid", "direction"] == "Rising"
    assert slopes.loc["radial feeders", "direction"] == "Falling"
    # A perfect fit has a zero-width interval, which excludes zero.
    assert bool(slopes.loc["smart grid", "excludes_zero"])
    assert slopes.loc["smart grid", "err_plus"] == pytest.approx(0.0)


def test_keyword_share_slopes_keep_distinct_keywords_that_share_a_slope():
    frame = pd.concat(
        [
            _keyword_year_frame(),
            _keyword_year_frame()
            .query("keyword == 'smart grid'")
            .assign(keyword="electric vehicles"),
        ]
    )
    slopes = keyword_share_slopes(frame, min_occurrences=15, n_extremes=2)

    # Each keyword once, even though the family (4 terms) is smaller than 2 * 2.
    assert slopes["keyword"].is_unique
    assert {"smart grid", "electric vehicles"} <= set(slopes["keyword"])


def test_keyword_trend_tests_adjust_over_the_whole_eligible_family():
    family = keyword_trend_tests(_keyword_year_frame(), min_occurrences=15).set_index("Keyword")

    assert set(family.index) == {"smart grid", "radial feeders", "filler"}
    assert family.loc["smart grid", "Trend"] == "Growing"
    assert family.loc["radial feeders", "Trend"] == "Falling"
    assert bool(family.loc["smart grid", "FDR significant"])
    assert (family["Adjusted p-value"] >= family["Raw p-value"] - 1e-12).all()


def test_rising_falling_keywords_narrows_the_table_to_the_charted_terms():
    result = rising_falling_keywords(_keyword_year_frame(), min_occurrences=15, n_extremes=1)

    assert result["slopes"]["keyword"].tolist() == ["radial feeders", "smart grid"]
    assert set(result["tests"]["Keyword"]) == {"radial feeders", "smart grid"}
    assert result["family_size"] == 3
    assert result["flagged"] >= 2
    assert result["survives"] <= result["flagged"]
    assert result["tests"]["Sen slope"].is_monotonic_increasing


def test_keyword_trend_frame_applies_the_year_floor_and_the_minimum_size():
    articles = pd.DataFrame(
        {"keywords": [["a", "b"]] * 20 + [["old"]], "year": [2015] * 20 + [1999]}
    )
    frame = keyword_trend_frame(articles, min_year=2010)

    assert frame is not None and "old" not in set(frame["keyword"])
    assert frame["year"].dtype.kind == "i"
    assert keyword_trend_frame(articles.head(5), min_year=2010) is None


def test_team_size_summary_and_yearly_mean():
    articles = pd.DataFrame(
        {
            "authors": [["a"], ["a", "b"], ["a", "b", "c", "d"], [], ["x", "y"]],
            "source": ["ieee", "ieee", "elsevier", "ieee", "elsevier"],
            "year": [2001, 2001, 2002, 2002, 1990],
        }
    )
    summary = team_size_summary(articles)

    assert summary["n_articles"] == 4  # the article with no author is left out
    assert summary["mean"]["total"] == pytest.approx(9 / 4)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["solo_share"] == pytest.approx(0.25)
    assert summary["distribution"].to_dict("list") == {
        "authors": [1, 2, 4],
        "articles": [1, 2, 1],
    }

    by_year = team_size_by_year(articles, min_year=2000)
    assert by_year.to_dict("records") == [
        {"year": 2001, "source": "ieee", "mean_authors": 1.5},
        {"year": 2002, "source": "elsevier", "mean_authors": 4.0},
    ]


def test_team_size_summary_on_a_layer_without_authors():
    assert team_size_summary(pd.DataFrame({"year": [2020]}))["n_articles"] == 0


def _cognitive_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    articles = pd.DataFrame(
        {
            "doi": ["d1", "d2", "d3", "d4"],
            "title": ["One", "Two", "Three", "Four"],
            "year": [2020, 2021, 2022, 2023],
            "map_x": [0.0, 0.0, 6.0, 9.0],
            "map_y": [0.0, 4.0, 4.0, 9.0],
            "citation_count": [5, 10, 20, np.nan],
        }
    )
    author_rows = pd.DataFrame(
        {
            "doi": ["d1", "d2", "d2", "d3", "d3", "d4", "d4"],
            "author_key": ["a", "a", "b", "b", "c", "a", "c"],
        }
    )
    return author_rows, articles


def test_cognitive_distance_uses_author_centroids_and_mean_pairwise_distance():
    author_rows, articles = _cognitive_fixture()
    out = cognitive_distance_by_article(author_rows, articles).set_index("doi")

    # d1 is single-author; d4 has no citation count, so it is out of scope
    # (and does not move a's centroid either).
    assert list(out.index) == ["d2", "d3"]
    # a = mean of d1 (0, 0) and d2 (0, 4) = (0, 2); b = mean of d2, d3 = (3, 4);
    # c = d3 = (6, 4).
    assert out.loc["d2", "cognitive_distance"] == pytest.approx(round(np.hypot(3, 2), 3))
    assert out.loc["d3", "cognitive_distance"] == pytest.approx(3.0)
    assert out.loc["d2", "team_size"] == 2
    assert out.loc["d3", "citation_count"] == 20
    assert out.loc["d2", "title"] == "Two"


def test_cognitive_distance_mean_over_every_pair_of_a_larger_team():
    articles = pd.DataFrame(
        {
            "doi": ["p1", "p2", "p3", "team"],
            "map_x": [0.0, 3.0, 0.0, 1.0],
            "map_y": [0.0, 0.0, 4.0, 4.0 / 3],
            "citation_count": [1, 1, 1, 7],
        }
    )
    # Each author's centroid is the mean of a solo article and the shared one.
    author_rows = pd.DataFrame(
        {
            "doi": ["p1", "p2", "p3", "team", "team", "team"],
            "author_key": ["x", "y", "z", "x", "y", "z"],
        }
    )
    out = cognitive_distance_by_article(author_rows, articles).set_index("doi")
    centroids = {
        "x": np.array([0.5, 2 / 3]),
        "y": np.array([2.0, 2 / 3]),
        "z": np.array([0.5, 8 / 3]),
    }
    expected = np.mean(
        [
            np.linalg.norm(centroids["x"] - centroids["y"]),
            np.linalg.norm(centroids["x"] - centroids["z"]),
            np.linalg.norm(centroids["y"] - centroids["z"]),
        ]
    )
    assert out.loc["team", "team_size"] == 3
    assert out.loc["team", "cognitive_distance"] == pytest.approx(round(expected, 3))


def test_cognitive_distance_analysis_reports_why_it_cannot_run():
    author_rows, articles = _cognitive_fixture()
    assert cognitive_distance_analysis(author_rows, articles)["status"] == "few_articles"
    assert (
        cognitive_distance_analysis(author_rows, articles, min_articles=2)["status"] == "few_teams"
    )
    result = cognitive_distance_analysis(author_rows, articles, min_articles=2, min_teams=2)
    assert result["status"] == "ok"
    assert result["spearman"] == pytest.approx(-1.0)


def test_pdf_selection_bias_views_label_the_metrics_and_flag_the_band():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "has_pdf": [True] * 20 + [False] * 20,
            "year": list(rng.integers(2000, 2020, 40)),
            "citation_count": [50, 60] * 10 + [1, 2] * 10,
        }
    )
    bias = pdf_selection_bias(frame, n_bootstrap=20, seed=3)
    finite, audit = pdf_selection_bias_views(bias)

    assert set(finite["Metrics"]) == {"Year", "Citations"}
    citations = finite.set_index("Metrics").loc["Citations"]
    assert bool(citations["not_negligible"])
    assert citations["error_upper"] >= 0 and citations["error_lower"] >= 0
    assert list(audit.columns) == [
        "Metrics",
        "Transformation",
        "n with PDF",
        "n without PDF",
        "SMD",
        "CI 2.5%",
        "CI 97.5%",
        "Status",
    ]
    # Same seed, same table: what makes the loader's cache key stable.
    pd.testing.assert_frame_equal(bias, pdf_selection_bias(frame, n_bootstrap=20, seed=3))


def _qualis_articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": [f"10.1/{i}" for i in range(7)],
            "year": [2020, 2020, 2021, 2021, 2022, 2022, 2022],
            "source": ["ieee", "elsevier", "ieee", "ieee", "elsevier", "ieee", "ieee"],
            "venue": ["J1", "J1", "J2", "J3", "J4", "J5", None],
        }
    )


def _venue_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "venue": ["J1", "J2", "J3", "J4", "Out of scope"],
            "matched_title": ["JOURNAL 1", "JOURNAL 2", None, "JOURNAL 4", "X"],
            "estrato": ["A1", "A3", None, "B3", "A1"],
            "score": [100.0, 91.2, None, 88.0, 99.0],
        }
    )


def test_qualis_strata_narrows_to_scope_and_orders_best_tier_first():
    strata = qualis_strata(_qualis_articles(), _venue_matches())

    table = strata.match_table
    assert table["venue"].tolist() == ["J1", "J2", "J4", "J3"]
    assert table["estrato"].tolist() == ["A1", "A3", "B3", NOT_CLASSIFIED]
    assert table["articles"].tolist() == [2, 1, 1, 1]
    assert table.loc[table["venue"] == "J3", ["matched_title", "score"]].iloc[0].tolist() == [
        "—",
        "—",
    ]
    assert strata.order == ["A1", "A3", "B3", NOT_CLASSIFIED]
    assert strata.totals.tolist() == [2, 1, 1, 1]
    # J5 is absent from the match table: no stratum, not a guessed one.
    assert pd.isna(strata.with_estrato.loc[5, "estrato"])


def test_qualis_strata_tolerates_a_bare_match_table():
    strata = qualis_strata(_qualis_articles(), pd.DataFrame({"venue": [], "estrato": []}))
    assert strata.match_table.empty
    assert strata.totals.empty and strata.order == []


def test_strata_volume_through_b2_excludes_lower_and_unclassified_tiers():
    strata = qualis_strata(_qualis_articles(), _venue_matches())
    by_year, order = strata_volume_by_year(strata.with_estrato, through="B2")

    assert order == ["A1", "A3"]
    assert by_year.to_dict("records") == [
        {"year": 2020, "estrato": "A1", "count": 2},
        {"year": 2021, "estrato": "A3", "count": 1},
    ]


def test_top_tier_summary_puts_the_source_split_beside_the_tier():
    strata = qualis_strata(_qualis_articles(), _venue_matches())
    summary = top_tier_summary(strata.with_estrato)

    assert summary["top_venues"].to_dict("records") == [
        {"venue": "J1", "articles": 2, "estrato": "A1", "ieee": 1, "elsevier": 1},
        {"venue": "J2", "articles": 1, "estrato": "A3", "ieee": 1, "elsevier": 0},
    ]
    assert summary["by_year"]["total"].sum() == 3
    assert summary["by_source_tier"]["count"].sum() == 3
