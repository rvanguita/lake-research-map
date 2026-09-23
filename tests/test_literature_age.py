"""WP-21: Price's index, citation half-life and the Sleeping Beauty coefficient."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lake_research_map.dashboard.analytics import (
    citation_half_lives,
    price_index_by_year,
    sleeping_beauty_scores,
)


def _refs(work_years: dict[str, tuple[int, list[float | None]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"doi": doi, "year": year, "reference_year": ref}
            for doi, (year, refs) in work_years.items()
            for ref in refs
        ]
    )


def _trajectory(doi: str, published: int, counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": doi,
            "publication_year": published,
            "year": [published + t for t in range(len(counts))],
            "citations": counts,
        }
    )


# -- Price's index ------------------------------------------------------------


def test_price_index_is_the_share_of_references_at_most_five_years_old():
    refs = _refs({f"w{i}": (2020, [2019, 2015, 2014, 2000]) for i in range(10)})

    result = price_index_by_year(refs, n_boot=50)

    row = result.iloc[0]
    # 2019 (1 year) and 2015 (5 years) are recent; 2014 and 2000 are not.
    assert row["price_index"] == pytest.approx(0.5)
    assert (row["works"], row["references"]) == (10, 40)
    # Identical works leave the bootstrap nothing to vary.
    assert row["ci_low"] == pytest.approx(0.5) and row["ci_high"] == pytest.approx(0.5)


def test_mostly_undated_lists_are_left_out_not_scored_on_their_dated_part():
    works = {f"dated{i}": (2020, [2019, 2000]) for i in range(10)}
    # One recent dated reference among four undated ones: scoring it would
    # report 100% recent for a list that is mostly books.
    works["books"] = (2020, [2019, None, None, None, None])

    result = price_index_by_year(_refs(works), n_boot=50)

    assert result.iloc[0]["works"] == 10
    assert result.iloc[0]["price_index"] == pytest.approx(0.5)


def test_references_from_the_future_are_not_counted_as_recent():
    refs = _refs({f"w{i}": (2020, [2021, 2030, 2000, 2019]) for i in range(10)})

    row = price_index_by_year(refs, n_boot=50).iloc[0]

    # 2021 is an in-press citation (kept), 2030 is a metadata error (dropped).
    assert row["references"] == 30
    assert row["price_index"] == pytest.approx(2 / 3)


def test_years_below_the_minimum_sample_are_not_reported_and_ci_brackets_the_estimate():
    rng = np.random.default_rng(1)
    works = {
        f"w{i}": (2020, [int(y) for y in rng.integers(2005, 2021, size=20)]) for i in range(30)
    }
    works.update({f"few{i}": (2010, [2009]) for i in range(3)})

    result = price_index_by_year(_refs(works), n_boot=200)

    assert result["year"].tolist() == [2020]
    row = result.iloc[0]
    assert row["ci_low"] < row["price_index"] < row["ci_high"]


def test_price_index_of_an_empty_frame_is_empty():
    assert price_index_by_year(pd.DataFrame()).empty


# -- half-life ----------------------------------------------------------------


def test_half_life_is_the_year_cumulative_citations_reach_half():
    early = _trajectory("early", 2010, [10, 0, 0, 0, 0, 10])
    late = _trajectory("late", 2010, [0, 0, 0, 5, 5, 10])

    result = citation_half_lives(
        pd.concat([early, late]), last_complete_year=2015, min_citations=10
    ).set_index("doi")

    assert result.loc["early", "half_life"] == 0
    assert result.loc["late", "half_life"] == 4
    assert result.loc["late", "age"] == 5


def test_young_and_rarely_cited_works_have_no_half_life():
    young = _trajectory("young", 2023, [50, 50])
    rare = _trajectory("rare", 2010, [1, 1, 1, 1, 1, 1])

    result = citation_half_lives(pd.concat([young, rare]), last_complete_year=2025)

    assert result.empty


def test_the_incomplete_current_year_and_pre_publication_citations_are_handled():
    frame = pd.DataFrame(
        {
            "doi": "w",
            "publication_year": 2015,
            "year": [2014, 2015, 2020, 2026],
            "citations": [4, 6, 10, 999],
        }
    )

    result = citation_half_lives(frame, last_complete_year=2025, min_citations=10)

    # 2014's online-first citations join year 0; 2026 is past the horizon.
    assert result.iloc[0]["total_citations"] == 20
    assert result.iloc[0]["half_life"] == 0


# -- Sleeping Beauty ----------------------------------------------------------


def test_beauty_coefficient_matches_ke_et_al_by_hand():
    frame = _trajectory("sb", 2000, [0, 0, 0, 10])

    row = sleeping_beauty_scores(frame, last_complete_year=2003).iloc[0]

    # Line 0 -> 10 over three years: (0 + 10/3 + 20/3 + 0) below it.
    assert row["beauty"] == pytest.approx(10.0)
    assert row["peak_year"] == 2003
    # Distance to the line is largest in year 2 (20 / hypot(10, 3)).
    assert row["awakening_year"] == 2002


def test_linear_growth_and_an_immediate_peak_score_zero():
    linear = _trajectory("linear", 2000, [0, 5, 10, 15])
    immediate = _trajectory("immediate", 2000, [20, 5, 1, 0])

    result = sleeping_beauty_scores(
        pd.concat([linear, immediate]), last_complete_year=2003
    ).set_index("doi")

    assert result.loc["linear", "beauty"] == pytest.approx(0.0)
    assert result.loc["immediate", "beauty"] == 0.0
    assert result.loc["immediate", "awakening_year"] is None or math.isnan(
        result.loc["immediate", "awakening_year"]
    )


def test_a_delayed_work_outranks_an_early_one():
    delayed = _trajectory("delayed", 2000, [0, 0, 0, 0, 0, 1, 2, 30])
    early = _trajectory("early", 2000, [10, 12, 8, 6, 4, 3, 2, 1])

    result = sleeping_beauty_scores(pd.concat([delayed, early]), last_complete_year=2007)

    assert result.sort_values("beauty", ascending=False)["doi"].iloc[0] == "delayed"


def test_trajectory_reader_keeps_the_latest_crawl_and_the_never_cited(monkeypatch):
    from datetime import datetime

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from lake_research_map.dashboard import data
    from lake_research_map.db.bronze_models import (
        Base,
        CitationYearCount,
        EnrichmentObservation,
        ExternalWork,
    )

    old, new = datetime(2026, 1, 1), datetime(2026, 9, 1)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for short, doi, year, cited in [
            ("W1", "10.1/A", 2015, 7),
            ("W2", "10.1/b", 2016, 0),
            ("W3", "10.1/c", 2005, 3),  # before the series starts
        ]:
            session.add(
                ExternalWork(
                    provider="openalex",
                    provider_work_id=short,
                    doi=doi,
                    publication_year=year,
                    first_observed_at=new,
                    last_observed_at=new,
                )
            )
            session.add(
                EnrichmentObservation(
                    provider="openalex",
                    doi=doi,
                    provider_work_id=short,
                    observed_at=new,
                    status="success",
                    citation_count=cited,
                )
            )
        for work, year, count, observed in [
            ("W1", 2016, 2, old),
            ("W1", 2016, 3, new),
            ("W1", 2017, 4, new),
            ("W3", 2012, 3, new),
        ]:
            session.add(
                CitationYearCount(
                    provider="openalex",
                    provider_work_id=work,
                    year=year,
                    citation_count=count,
                    observed_at=observed,
                )
            )
        session.commit()
    monkeypatch.setattr(data, "get_engine", lambda layer: engine)

    frame, stats = data.load_citation_trajectories()

    assert stats == {
        "population": 3,
        "complete_history": 2,
        "left_censored": 1,
        "series_start": 2012,
    }
    by_doi = frame.groupby("doi")["citations"].sum().to_dict()
    # The superseded 2026-01 snapshot is not added to the September one, and
    # the never-cited work stays in with a zero instead of disappearing.
    assert by_doi == {"10.1/a": 7, "10.1/b": 0}
