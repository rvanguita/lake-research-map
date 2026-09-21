"""Tests for advanced machine learning: quantile forecasting and Bass diffusion."""

from __future__ import annotations

import numpy as np

from lake_research_map.dashboard.forecasting import fit_bass_diffusion, fit_quantile_forecast


def test_fit_quantile_forecast():
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    values = np.array([10, 15, 22, 30, 45, 50, 65, 75, 90, 110, 130])
    res = fit_quantile_forecast(years, values, forecast_years=(2026, 2027))
    assert res["valid"] is True
    assert "p10" in res and "p50" in res and "p90" in res
    assert len(res["p10"]) == 2
    # Quantiles must be monotonic: P10 <= P50 <= P90
    assert (res["p10"] <= res["p50"] + 1e-3).all()
    assert (res["p50"] <= res["p90"] + 1e-3).all()


def test_fit_bass_diffusion():
    years = np.arange(2010, 2025)
    # Bell-shaped adoption pattern
    adoptions = np.array([2, 5, 12, 25, 45, 70, 85, 90, 82, 68, 50, 35, 20, 12, 5])
    res = fit_bass_diffusion(years, adoptions)
    assert res["valid"] is True
    assert res["m"] > 0
    assert res["p"] > 0
    assert res["stage"] in ("crescimento", "maturidade")


def test_detect_bibliometric_anomalies():
    import pandas as pd

    from lake_research_map.dashboard.analytics import detect_bibliometric_anomalies

    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "doi": [f"10.1/{i}" for i in range(n)],
            "year": rng.integers(2010, 2024, size=n),
            "citation_count": rng.exponential(10, size=n),
            "reference_count": rng.integers(5, 50, size=n),
            "authors": [["Author A", "Author B"]] * n,
            "relevance_score": rng.uniform(0.5, 0.9, size=n),
            "has_pdf": [True] * n,
        }
    )
    df.loc[0, "citation_count"] = 500
    df.loc[0, "year"] = 2023

    res = detect_bibliometric_anomalies(df, contamination=0.05)
    assert "anomaly_score" in res.columns
    assert "is_anomaly" in res.columns
    assert "anomaly_reason" in res.columns
    assert res["is_anomaly"].sum() > 0
    assert bool(res.loc[0, "is_anomaly"]) is True


def test_dynamic_topic_ctfidf():
    import pandas as pd

    from lake_research_map.dashboard.analytics import dynamic_topic_ctfidf

    df = pd.DataFrame(
        {
            "theme_label": [
                "Expansão",
                "Expansão",
                "Geração Distribuída",
                "Geração Distribuída",
            ]
            * 10,
            "year": [2005, 2008, 2005, 2008] * 5 + [2020, 2022, 2020, 2022] * 5,
            "title": [
                "distribution network expansion transmission lines",
                "substation feeder optimal reinforcement planning",
                "photovoltaic solar generation hosting capacity",
                "distributed renewable energy penetration voltage control",
            ]
            * 10,
        }
    )
    res = dynamic_topic_ctfidf(df, time_windows=[(2000, 2010), (2019, 2026)])
    assert res["valid"] is True
    assert len(res["epochs"]) >= 1
    assert not res["summary_df"].empty
    assert "Characteristic Terms (c-TF-IDF)" in res["summary_df"].columns
