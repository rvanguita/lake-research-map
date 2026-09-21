"""Tests for `dashboard.forecasting` -- the model selection and forecast logic.

Series are synthetic and exactly shaped (a clean straight line, a clean
exponential, a too-short series), so the assertions are about the mechanism --
does it pick the right candidate, does it refuse to guess on thin data -- and
not about the real corpus, which changes whenever the pipeline runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lake_research_map.dashboard.forecasting import (
    _fit_model,
    _mae,
    fit_and_forecast,
    yearly_counts,
)

TRAIN_YEARS = tuple(range(2010, 2026))  # 2010..2025, the full training range


def _series(values_by_year: dict[int, float]) -> pd.Series:
    return pd.Series(values_by_year, dtype=float).sort_index()


def _linear_series(slope: float = 3.0, intercept: float = 10.0) -> pd.Series:
    return _series({y: intercept + slope * (y - 2010) for y in TRAIN_YEARS})


def test_fit_model_linear_recovers_a_straight_line():
    years = np.array(TRAIN_YEARS)
    values = 10 + 3.0 * (years - 2010)

    predict = _fit_model("linear", years, values)

    assert np.allclose(predict([2026]), [10 + 3.0 * 16], atol=1e-6)


def test_fit_model_log_linear_recovers_exponential_growth():
    years = np.array(TRAIN_YEARS)
    values = np.expm1(0.2 * (years - 2010))

    predict = _fit_model("log_linear", years, values)

    assert np.allclose(predict([2026]), [np.expm1(0.2 * 16)], rtol=1e-4)


def test_fit_model_rejects_unknown_kind():
    try:
        _fit_model("wishful_thinking", np.array([2020, 2021]), np.array([1.0, 2.0]))
    except ValueError as exc:
        assert "wishful_thinking" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected ValueError for an unknown model kind")


def test_mae_is_mean_absolute_error():
    assert _mae([1.0, 2.0, 3.0], [1.0, 4.0, 0.0]) == 5.0 / 3.0


def test_fit_and_forecast_flags_insufficient_data():
    result = fit_and_forecast(_series({2023: 4.0, 2024: 5.0, 2025: 6.0}))

    assert result.insufficient_data is True
    assert result.chosen_model == "none"
    assert result.forecast_values.size == 0
    assert result.notes


def test_fit_and_forecast_picks_linear_for_a_straight_line():
    history = _linear_series()
    history.loc[2026] = 10 + 3.0 * 16  # holdout year, perfectly on the line

    result = fit_and_forecast(history.sort_index())

    assert result.insufficient_data is False
    assert result.chosen_model == "linear"
    assert result.r2_train > 0.99


def test_fit_and_forecast_extrapolates_the_trend():
    history = _linear_series(slope=3.0, intercept=10.0)

    result = fit_and_forecast(history)

    assert result.forecast_years == (2027, 2028)
    assert len(result.forecast_values) == 2
    # A rising line must keep rising, and 2028 must exceed 2027.
    assert result.forecast_values[1] > result.forecast_values[0] > history.iloc[-1]


def test_fit_and_forecast_band_brackets_the_point_forecast():
    result = fit_and_forecast(_linear_series())

    assert np.all(result.forecast_lower <= result.forecast_values)
    assert np.all(result.forecast_values <= result.forecast_upper)


def test_fit_and_forecast_compares_every_candidate():
    result = fit_and_forecast(_linear_series())

    assert set(result.model_comparison["model"]) == {"baseline", "linear", "log_linear"}
    assert result.chosen_model in set(result.model_comparison["model"])


def test_fit_and_forecast_ignores_years_before_the_training_window():
    """Pre-2010 years are too sparse to trend on, so they must not reach the fit."""
    history = _linear_series()
    history.loc[1998] = 999.0  # an outlier that would wreck any fit it entered

    result = fit_and_forecast(history.sort_index())

    assert 1998 not in result.history.index
    assert result.r2_train > 0.99


def test_fit_and_forecast_reports_the_holdout_year():
    history = _linear_series()
    history.loc[2026] = 42.0

    result = fit_and_forecast(history.sort_index())

    assert result.holdout_year == 2026
    assert result.holdout_actual == 42.0
    assert result.holdout_predicted is not None


def test_yearly_counts_counts_articles_per_year():
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, None],
            "sources": [["ieee"], ["elsevier"], ["ieee"], ["ieee"]],
        }
    )

    counts = yearly_counts(df)

    assert counts.loc[2020] == 2
    assert counts.loc[2021] == 1


def test_yearly_counts_filters_by_source():
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021],
            "sources": [["ieee"], ["elsevier"], ["ieee"]],
        }
    )

    counts = yearly_counts(df, source="ieee")

    assert counts.loc[2020] == 1
    assert counts.loc[2021] == 1


def test_fit_bass_diffusion_nls():
    from lake_research_map.dashboard.forecasting import fit_bass_diffusion_nls

    years = np.arange(2010, 2021)
    # S-curve adoption pattern
    adoptions = np.array([2, 5, 12, 25, 45, 60, 55, 40, 25, 15, 8], dtype=float)
    res = fit_bass_diffusion_nls(years, adoptions)

    assert res["valid"] is True
    assert res["m"] > 0
    assert res["p"] > 0
    assert res["q"] > 0
    assert res["t_peak"] is not None
    assert res["method"] in ("nls", "ols")


def test_fit_and_forecast_expanding_confidence_interval():
    s = _linear_series()
    s.loc[2015] += 5.0
    result = fit_and_forecast(s)
    # Horizon 2 margin should be strictly larger than horizon 1 margin
    margin_1 = result.forecast_upper[0] - result.forecast_values[0]
    margin_2 = result.forecast_upper[1] - result.forecast_values[1]
    assert margin_2 > margin_1
