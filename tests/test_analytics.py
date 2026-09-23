import pandas as pd

from lake_research_map.dashboard.analytics import (
    OTHERS_LABEL,
    cumulative_by_category,
    cumulative_by_venue,
    publication_category_counts_by_year,
    publication_category_totals,
)


def _category_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"category": "A1", "year": 2020},
            {"category": "A1", "year": 2021},
            {"category": "A2", "year": 2021},
            {"category": "C", "year": 2022},
        ]
    )


def test_cumulative_by_category_buckets_and_cumsums():
    result = cumulative_by_category(_category_df(), "category", top_n=10)

    a1 = result[result["category"] == "A1"].set_index("year")["cumulative"]
    assert list(a1.reindex([2020, 2021, 2022], fill_value=a1.max())) == [1, 2, 2]

    a2 = result[result["category"] == "A2"].set_index("year")["cumulative"]
    assert a2.loc[2022] == 1


def test_cumulative_by_category_buckets_tail_into_others():
    result = cumulative_by_category(_category_df(), "category", top_n=1)
    # Only the most frequent category ("A1", 2 rows) keeps its own name; the
    # rest collapse into the catch-all bucket.
    assert set(result["category"]) == {"A1", OTHERS_LABEL}


def test_cumulative_by_category_empty_when_no_valid_years():
    empty = pd.DataFrame({"category": ["A1"], "year": [None]})
    assert cumulative_by_category(empty, "category").empty


def test_cumulative_by_venue_is_a_thin_wrapper():
    df = pd.DataFrame(
        [
            {"venue": "Journal A", "year": 2020},
            {"venue": "Journal A", "year": 2021},
        ]
    )
    result = cumulative_by_venue(df, top_n=10)
    assert "venue" in result.columns
    assert list(result.set_index("year")["cumulative"]) == [1, 2]


def test_publication_category_totals_reconcile_unknown_values_as_other():
    frame = pd.DataFrame(
        {"publication_category": ["journal", "conference", "review", "other", None, "legacy"]}
    )

    totals = publication_category_totals(frame)

    assert totals.to_dict() == {"journal": 1, "conference": 1, "review": 1, "other": 3}
    assert int(totals.sum()) == len(frame)


def test_publication_category_counts_by_year_include_zero_categories_and_total():
    frame = pd.DataFrame(
        {
            "year": [2020, 2020, 2021],
            "publication_category": ["journal", "conference", "review"],
        }
    )

    result = publication_category_counts_by_year(frame).set_index("year")

    assert result.loc[2020].to_dict() == {
        "journal": 1,
        "conference": 1,
        "review": 0,
        "other": 0,
        "total": 2,
    }
    assert result.loc[2021].to_dict() == {
        "journal": 0,
        "conference": 0,
        "review": 1,
        "other": 0,
        "total": 1,
    }


def test_valid_years_keeps_in_press_records_dated_next_year():
    """In-press rows carry their future issue year and are real publications.

    The bound used to be pinned at 2026 while the corpus already held 11
    articles dated 2027, so every trend, production and keyword surface
    dropped them without saying so.
    """
    from datetime import UTC, datetime

    from lake_research_map.dashboard.analytics import plausible_year_bound, valid_years

    next_year = datetime.now(UTC).year + 1
    assert plausible_year_bound() == next_year

    df = pd.DataFrame({"year": [1949, 2020, next_year, next_year + 1, 9999, None]})
    kept = valid_years(df).dropna().astype(int).tolist()

    assert kept == [2020, next_year]


def test_valid_years_still_rejects_parse_garbage():
    from lake_research_map.dashboard.analytics import valid_years

    df = pd.DataFrame({"year": [201, 0, -5, "not a year"]})
    assert valid_years(df).dropna().empty


def test_valid_years_honours_an_explicit_upper_bound():
    from lake_research_map.dashboard.analytics import valid_years

    df = pd.DataFrame({"year": [2020, 2024]})
    assert valid_years(df, hi=2021).dropna().astype(int).tolist() == [2020]
