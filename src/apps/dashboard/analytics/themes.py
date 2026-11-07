"""Theme-level aggregations for the Screening and discovery page.

Theme sizes, the (year x theme) evolution grid with its moving-average
smoothing, and the chronological windows the thematic drift is measured over.
The page's segmented controls read their option labels from the dicts below,
so a label and the value it selects cannot drift apart: the smoothing control
once offered "3-year moving average" while the page compared the choice with
"3 years", so the window was always 1 and smoothing never ran.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transform.semantics import compute_temporal_drift

# Smoothing control: label -> centred moving-average window (1 = raw series).
SMOOTHING_WINDOWS: dict[str, int] = {
    "3-year moving average (smooth)": 3,
    "5-year moving average": 5,
    "No smoothing (raw)": 1,
}

# Metric control: label -> whether the series is each theme's share of the year.
THEME_METRICS: dict[str, bool] = {
    "Relative Participation (%)": True,
    "Absolute Volume (Articles)": False,
}

# Fixed horizon floors, offered only when the corpus reaches further back than
# them. The modern period is the default because the sparse early years make a
# yearly share swing between 0% and 100% on one or two articles.
HORIZON_FLOORS: dict[str, int] = {
    "Since 2000 (recommended)": 2000,
    "Since 1990": 1990,
}

# Thematic drift: how many chronological windows, and the earliest year any of
# them may start. Before 1990 the corpus holds a few dozen scattered records,
# too few to place a theme's centroid in any one window.
DRIFT_WINDOW_COUNT = 3
DRIFT_MIN_YEAR = 1990


def theme_sizes(scored: pd.DataFrame) -> pd.DataFrame:
    """Articles and mean relevance per theme, largest theme first."""
    if scored.empty or "theme_label" not in scored.columns:
        return pd.DataFrame(columns=["theme_label", "articles", "mean_relevance"])
    relevance = (
        scored["relevance_score"]
        if "relevance_score" in scored.columns
        else pd.Series(np.nan, index=scored.index)
    )
    return (
        scored.assign(_relevance=relevance)
        .groupby("theme_label")
        .agg(articles=("theme_label", "size"), mean_relevance=("_relevance", "mean"))
        .reset_index()
        .sort_values("articles", ascending=False)
        .reset_index(drop=True)
    )


def theme_years(scored: pd.DataFrame) -> pd.DataFrame:
    """Rows with a theme and a numeric year, `year` cast to int."""
    if scored.empty or not {"year", "theme_label"} <= set(scored.columns):
        return pd.DataFrame(columns=["year", "theme_label"])
    yearly = scored.dropna(subset=["year", "theme_label"]).copy()
    yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
    return yearly.dropna(subset=["year"]).astype({"year": int})


def horizon_options(min_year: int, max_year: int) -> dict[str, int]:
    """Time-horizon control: label -> first year shown, default first.

    A fixed floor is offered only when the corpus starts before it; the full
    history is always the last option.
    """
    options = {label: floor for label, floor in HORIZON_FLOORS.items() if min_year < floor}
    options[f"Full history ({min_year}–{max_year})"] = min_year
    return options


def theme_evolution(
    yearly: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    window: int = 1,
    relative: bool = True,
) -> pd.DataFrame:
    """Per (year, theme) series over a complete year grid, optionally smoothed.

    Every year in `start_year..end_year` gets a row for every theme seen in the
    period, filled with zero, so a gap year is a real zero rather than a line
    interpolated across it. `value` is the share of that year's articles (%)
    when `relative`, else the article count, smoothed by a centred moving average
    of `window` years when `window > 1`. A smoothed share is renormalized so each
    year still sums to 100%. `volume_real` and `pct_real` are always the raw,
    unsmoothed count and share, for the tooltip.
    """
    columns = ["year", "theme_label", "value", "volume_real", "pct_real"]
    scoped = yearly[(yearly["year"] >= start_year) & (yearly["year"] <= end_year)]
    all_years = list(range(start_year, end_year + 1))
    all_themes = sorted(scoped["theme_label"].unique())
    if not all_years or not all_themes:
        return pd.DataFrame(columns=columns)

    grid = pd.MultiIndex.from_product([all_years, all_themes], names=["year", "theme_label"])
    counts = scoped.groupby(["year", "theme_label"]).size().reindex(grid, fill_value=0)
    pivot = counts.unstack("theme_label").astype(float)

    # A year with no article divides by 1, not 0: its shares are all zero.
    year_totals = pivot.sum(axis=1).replace(0, 1)
    raw_share = pivot.div(year_totals, axis=0) * 100
    series = raw_share if relative else pivot
    if window > 1:
        series = series.rolling(window=window, min_periods=1, center=True).mean()
        if relative:
            series = series.div(series.sum(axis=1).replace(0, 1), axis=0) * 100

    out = pd.concat(
        [
            series.stack().rename("value"),
            counts.rename("volume_real"),
            raw_share.stack().rename("pct_real"),
        ],
        axis=1,
    ).reset_index()
    return out.sort_values(["theme_label", "year"]).reset_index(drop=True)[columns]


def drift_windows(
    years: pd.Series | np.ndarray,
    *,
    n_windows: int = DRIFT_WINDOW_COUNT,
    min_year: int = DRIFT_MIN_YEAR,
) -> list[tuple[int, int]]:
    """Up to `n_windows` contiguous, non-overlapping year windows of equal article count.

    The rule: take the article years from `min_year` on, and cut them at the
    article-count terciles (for the default of three), so each window holds
    roughly a third of the dated articles. A year is never split -- each year
    goes to the window its own mid-point of the cumulative count falls in -- so
    a window can come out larger than its share, and a corpus with fewer
    distinct years than `n_windows` gets fewer windows. Equal-*width* windows
    would leave the first one nearly empty, because the corpus grows roughly
    tenfold over the period. A window spans from its first to its last year
    with articles. This replaces the literal (1990-2010, 2011-2018, 2019-2026)
    the page carried, which was tuned to one snapshot of the corpus.
    """
    values = pd.to_numeric(pd.Series(np.asarray(years)), errors="coerce").dropna()
    values = values[values >= min_year].astype(int)
    if values.empty or n_windows < 1:
        return []
    per_year = values.value_counts().sort_index()
    cumulative = per_year.cumsum() / per_year.sum()
    midpoint = cumulative - per_year / per_year.sum() / 2
    index = np.minimum((midpoint * n_windows).astype(int), n_windows - 1)
    windows = (
        pd.DataFrame({"year": per_year.index, "window": index.to_numpy()})
        .groupby("window")["year"]
        .agg(["min", "max"])
    )
    return [(int(lo), int(hi)) for lo, hi in windows.itertuples(index=False)]


def theme_drift(scored: pd.DataFrame, windows: list[tuple[int, int]]) -> pd.DataFrame:
    """Theme centroid on the semantic map in each window.

    Columns `Theme`, `Time` ("start–end"), `map_x`, `map_y`, `Articles`; a theme
    has no row for a window it has no article in. Empty when the frame lacks
    map coordinates, years or theme ids.
    """
    columns = ["Theme", "Time", "map_x", "map_y", "Articles"]
    needed = {"map_x", "map_y", "year", "theme_id"}
    if scored.empty or not windows or not needed <= set(scored.columns):
        return pd.DataFrame(columns=columns)
    valid = scored.dropna(subset=list(needed)).copy()
    valid["year"] = pd.to_numeric(valid["year"], errors="coerce")
    valid = valid.dropna(subset=["year"])
    if valid.empty:
        return pd.DataFrame(columns=columns)
    drift = compute_temporal_drift(
        valid[["map_x", "map_y"]].to_numpy(dtype=float),
        valid["theme_id"].to_numpy(dtype=int),
        valid["year"].to_numpy(dtype=int),
        windows,
    )
    names = (
        dict(zip(valid["theme_id"].astype(int), valid["theme_label"], strict=False))
        if "theme_label" in valid.columns
        else {}
    )
    rows = [
        {
            "Theme": names.get(theme_id, f"Theme {theme_id}"),
            "Time": point["name"],
            "map_x": point["x"],
            "map_y": point["y"],
            "Articles": point["count"],
        }
        for theme_id, points in drift.items()
        for point in points
    ]
    return pd.DataFrame(rows, columns=columns)
