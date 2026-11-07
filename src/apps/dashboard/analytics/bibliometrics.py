"""Classical bibliometric laws: Lotka (author productivity) and Bradford (venue zones)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lotka_law_analysis(author_counts: pd.Series) -> dict:
    """Fit Lotka's Law (f(x) = C / x^alpha) on author productivity distribution."""
    counts = author_counts[author_counts > 0].value_counts().sort_index()
    if len(counts) < 3:
        return {"alpha": 2.0, "c": 1.0, "r2": 0.0, "table": pd.DataFrame()}

    x = np.asarray(counts.index, dtype=float)
    y = np.asarray(counts.values, dtype=float)

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept = np.polyfit(log_x, log_y, 1, w=y)
    alpha = -float(slope)
    c_const = float(np.exp(intercept))

    pred_log_y = intercept + slope * log_x
    res_ss = np.sum((log_y - pred_log_y) ** 2)
    tot_ss = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = float(1.0 - (res_ss / tot_ss)) if tot_ss > 0 else 0.0

    table = pd.DataFrame(
        {
            "papers_x": x.astype(int),
            "empirical_authors": y.astype(int),
            "theoretical_authors": np.round(c_const / (x**alpha)).astype(int),
            "empirical_share": np.round(y / np.sum(y), 4),
            "theoretical_share": np.round((1.0 / (x**alpha)) / np.sum(1.0 / (x**alpha)), 4),
        }
    )
    return {"alpha": alpha, "c": c_const, "r2": r2, "table": table}


def bradford_zones(articles_df: pd.DataFrame, n_zones: int = 3) -> dict:
    """Partition publication venues into Bradford's concentric scattering zones."""
    if articles_df.empty or "venue" not in articles_df.columns:
        return {
            "multiplier_mean": 0.0,
            "zone_summary": pd.DataFrame(),
            "zone_table": pd.DataFrame(),
        }

    counts = articles_df["venue"].dropna().value_counts()
    if counts.empty:
        return {
            "multiplier_mean": 0.0,
            "zone_summary": pd.DataFrame(),
            "zone_table": pd.DataFrame(),
        }

    total_articles = counts.sum()
    target_per_zone = total_articles / n_zones

    current_zone = 1
    cum_articles = 0
    venue_zone_map = {}

    for venue, cnt in counts.items():
        venue_zone_map[venue] = current_zone
        cum_articles += cnt
        if cum_articles >= target_per_zone * current_zone and current_zone < n_zones:
            current_zone += 1

    zone_df = pd.DataFrame({"venue": counts.index, "articles": counts.values})
    zone_df["zone"] = zone_df["venue"].map(venue_zone_map)

    zone_summary = (
        zone_df.groupby("zone")
        .agg(
            venues=("venue", "count"),
            articles=("articles", "sum"),
        )
        .reset_index()
    )
    zone_summary["share_articles"] = zone_summary["articles"] / total_articles

    v_counts = zone_summary["venues"].to_numpy()
    multipliers = (
        [float(v_counts[i] / v_counts[i - 1]) for i in range(1, len(v_counts))]
        if len(v_counts) > 1 and v_counts[0] > 0
        else [1.0]
    )

    return {
        "multiplier_mean": float(np.mean(multipliers)),
        "zone_summary": zone_summary,
        "zone_table": zone_df,
    }
