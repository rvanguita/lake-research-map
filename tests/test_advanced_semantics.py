"""Tests for advanced semantic representations, projections, centroids, drift, and novelty."""

from __future__ import annotations

import numpy as np
import pytest

from transform.semantics import (
    compute_semantic_novelty,
    compute_temporal_drift,
    project_pca_2d,
    project_umap,
)


def test_project_pca_2d_produces_two_columns():
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, 1, (20, 10)).astype("float32")
    coords = project_pca_2d(matrix)
    assert coords.shape == (20, 2)
    assert np.isfinite(coords).all()


def test_project_pca_2d_handles_degenerate_cases():
    assert project_pca_2d(np.zeros((0, 4), dtype="float32")).shape == (0, 2)
    assert project_pca_2d(np.array([[1.0, 2.0]], dtype="float32")).shape == (1, 2)


def test_project_umap_is_explicitly_unavailable_without_dependency():
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, 1, (15, 6)).astype("float32")
    with pytest.raises(ImportError):
        project_umap(matrix, n_neighbors=5)


def test_compute_temporal_drift():
    coords = np.array(
        [
            [1.0, 1.0],  # 2005, theme 0
            [2.0, 2.0],  # 2008, theme 0
            [10.0, 10.0],  # 2015, theme 0
        ],
        dtype="float32",
    )
    labels = np.array([0, 0, 0])
    years = np.array([2005, 2008, 2015])
    windows = [(2000, 2010), (2011, 2020)]

    drift = compute_temporal_drift(coords, labels, years, windows)
    assert 0 in drift
    assert len(drift[0]) == 2
    assert drift[0][0]["name"] == "2000–2010"
    assert np.isclose(drift[0][0]["x"], 1.5)
    assert drift[0][1]["name"] == "2011–2020"
    assert np.isclose(drift[0][1]["x"], 10.0)


def test_compute_semantic_novelty():
    # 4 points close together and 1 distant point
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, 0.02, 0.0],
            [1.0, -0.01, 0.0],
            [0.0, 1.0, 0.0],  # orthogonal / novel
        ],
        dtype="float32",
    )
    novelty = compute_semantic_novelty(matrix, k=2)
    assert len(novelty) == 5
    # The orthogonal vector must have higher novelty than the clustered ones
    assert novelty[4] > novelty[0]
    assert novelty[4] > novelty[1]


def test_temporal_theme_evolution_smoothing_and_grid():
    import pandas as pd

    from apps.dashboard.analytics.themes import theme_evolution

    # Synthetic data with missing year-theme pairs (2019 has no article at all).
    data = pd.DataFrame(
        {
            "year": [2018, 2018, 2020, 2021, 2021],
            "theme_label": ["Theme A", "Theme B", "Theme A", "Theme A", "Theme B"],
        }
    )
    out = theme_evolution(data, start_year=2018, end_year=2021, window=3, relative=True)

    # The grid is complete: 4 years x 2 themes, gap cells filled with zero.
    assert len(out) == 8
    assert not out[["value", "volume_real", "pct_real"]].isna().any().any()
    assert out.loc[out["year"] == 2019, "volume_real"].tolist() == [0, 0]

    # A smoothed share is renormalized: every year still sums to 100%.
    assert np.allclose(out.groupby("year")["value"].sum(), 100.0)
    # The raw share for the tooltip is untouched by smoothing.
    raw_2020 = out[(out["year"] == 2020)].set_index("theme_label")["pct_real"]
    assert raw_2020.to_dict() == {"Theme A": 100.0, "Theme B": 0.0}


def test_a_three_year_window_actually_smooths():
    """Regression: the page compared "3-year ..." labels with "3 years", so the window was 1."""
    import pandas as pd

    from apps.dashboard.analytics.themes import SMOOTHING_WINDOWS, theme_evolution

    data = pd.DataFrame({"year": [2010, 2012, 2012, 2012], "theme_label": ["A", "A", "A", "A"]})
    window = SMOOTHING_WINDOWS["3-year moving average (smooth)"]
    assert window == 3
    smoothed = theme_evolution(data, start_year=2010, end_year=2012, window=window, relative=False)
    raw = theme_evolution(data, start_year=2010, end_year=2012, window=1, relative=False)

    assert raw.set_index("year")["value"].tolist() == [1.0, 0.0, 3.0]
    # Centred 3-year mean with min_periods=1: (1+0)/2, (1+0+3)/3, (0+3)/2.
    assert np.allclose(smoothed.set_index("year")["value"].tolist(), [0.5, 4 / 3, 1.5])
    assert smoothed["volume_real"].tolist() == raw["volume_real"].tolist()


def test_every_smoothing_label_maps_to_a_distinct_window():
    from apps.dashboard.analytics.themes import SMOOTHING_WINDOWS

    assert sorted(SMOOTHING_WINDOWS.values()) == [1, 3, 5]


def test_horizon_options_offer_floors_only_when_the_corpus_reaches_past_them():
    from apps.dashboard.analytics.themes import horizon_options

    assert horizon_options(1926, 2027) == {
        "Since 2000 (recommended)": 2000,
        "Since 1990": 1990,
        "Full history (1926–2027)": 1926,
    }
    assert horizon_options(1995, 2027) == {
        "Since 2000 (recommended)": 2000,
        "Full history (1995–2027)": 1995,
    }
    assert horizon_options(2005, 2027) == {"Full history (2005–2027)": 2005}


def test_drift_windows_split_the_dated_articles_into_equal_count_windows():
    from apps.dashboard.analytics.themes import drift_windows

    # 1980 is below the floor; 30 articles from 1990 on, a third in each block.
    years = [1980] + [1995] * 10 + [2012, 2014] * 5 + [2020] * 10
    assert drift_windows(years) == [(1995, 1995), (2012, 2014), (2020, 2020)]
    # Contiguous and non-overlapping on a skewed, growing corpus.
    growing = [y for y in range(1990, 2027) for _ in range(max(1, y - 1989))]
    windows = drift_windows(growing)
    assert len(windows) == 3
    assert windows[0][0] == 1990 and windows[-1][1] == 2026
    assert all(prev[1] < nxt[0] for prev, nxt in zip(windows, windows[1:], strict=False))
    # Fewer distinct years than windows -> fewer windows, never an empty one.
    assert drift_windows([2020, 2020, 2021]) == [(2020, 2020), (2021, 2021)]
    assert drift_windows([]) == []


def test_theme_drift_reports_one_centroid_per_theme_and_window():
    import pandas as pd

    from apps.dashboard.analytics.themes import theme_drift

    scored = pd.DataFrame(
        {
            "map_x": [0.0, 2.0, 10.0, 12.0],
            "map_y": [0.0, 0.0, 1.0, 1.0],
            "year": [2000, 2001, 2020, 2021],
            "theme_id": [0, 0, 0, 1],
            "theme_label": ["Grid", "Grid", "Grid", "Storage"],
        }
    )
    drift = theme_drift(scored, [(2000, 2010), (2011, 2021)])

    grid = drift[drift["Theme"] == "Grid"].set_index("Time")
    assert grid.loc["2000–2010", ["map_x", "Articles"]].tolist() == [1.0, 2]
    assert grid.loc["2011–2021", ["map_x", "Articles"]].tolist() == [10.0, 1]
    assert drift[drift["Theme"] == "Storage"]["Time"].tolist() == ["2011–2021"]


def test_alternative_projections_and_novelty_mapping():
    import pandas as pd

    dois = ["10.1/a", "10.1/b", "10.1/c"]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    pca_coords = project_pca_2d(matrix)
    novelty = compute_semantic_novelty(matrix, k=2)

    assert pca_coords.shape == (3, 2)
    assert len(novelty) == 3

    df = pd.DataFrame({"doi": dois, "pca_x": pca_coords[:, 0], "novelty": novelty})
    assert len(df) == 3
    assert df["novelty"].iloc[2] > df["novelty"].iloc[0]
