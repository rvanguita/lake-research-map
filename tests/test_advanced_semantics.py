"""Tests for advanced semantic representations, projections, centroids, drift, and novelty."""

from __future__ import annotations

import numpy as np
import pytest

from lake_research_map.transform.semantics import (
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

    # Synthetic data with missing year-theme pairs
    data = pd.DataFrame(
        {
            "year": [2018, 2018, 2020, 2021, 2021],
            "theme_label": ["Theme A", "Theme B", "Theme A", "Theme A", "Theme B"],
        }
    )
    all_years = list(range(2018, 2022))
    all_themes = sorted(data["theme_label"].unique())

    grid = (
        pd.MultiIndex.from_product([all_years, all_themes], names=["year", "theme_label"])
        .to_frame()
        .reset_index(drop=True)
    )
    raw_counts = data.groupby(["year", "theme_label"]).size().reset_index(name="artigos")
    complete = pd.merge(grid, raw_counts, on=["year", "theme_label"], how="left").fillna(
        {"artigos": 0}
    )
    pivot = complete.pivot(index="year", columns="theme_label", values="artigos")

    # Verify grid is complete (4 years x 2 themes = 8 cells)
    assert pivot.shape == (4, 2)
    assert not pivot.isna().any().any()

    # Verify 3-year rolling average preserves 100% total in relative mode
    pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
    smoothed = pct.rolling(window=3, min_periods=1, center=True).mean()
    smoothed = smoothed.div(smoothed.sum(axis=1).replace(0, 1), axis=0) * 100

    row_sums = smoothed.sum(axis=1)
    # Each year with articles should sum to 100%
    assert np.allclose(row_sums, 100.0)


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
