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
    FORECAST_YEARS,
    HOLDOUT_YEAR,
    _fit_model,
    _mae,
    fit_and_forecast,
    yearly_counts,
)

TRAIN_YEARS = tuple(range(2010, HOLDOUT_YEAR))


def _series(values_by_year: dict[int, float]) -> pd.Series:
    return pd.Series(values_by_year, dtype=float).sort_index()


def _linear_series(slope: float = 3.0, intercept: float = 10.0) -> pd.Series:
    return _series({y: intercept + slope * (y - 2010) for y in TRAIN_YEARS})


def test_fit_model_linear_recovers_a_straight_line():
    years = np.array(TRAIN_YEARS)
    values = 10 + 3.0 * (years - 2010)

    predict = _fit_model("linear", years, values)

    expected = 10 + 3.0 * (HOLDOUT_YEAR - 2010)
    assert np.allclose(predict([HOLDOUT_YEAR]), [expected], atol=1e-6)


def test_fit_model_log_linear_recovers_exponential_growth():
    years = np.array(TRAIN_YEARS)
    values = np.expm1(0.2 * (years - 2010))

    predict = _fit_model("log_linear", years, values)

    expected = np.expm1(0.2 * (HOLDOUT_YEAR - 2010))
    assert np.allclose(predict([HOLDOUT_YEAR]), [expected], rtol=1e-4)


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
    history.loc[HOLDOUT_YEAR] = 10 + 3.0 * (HOLDOUT_YEAR - 2010)

    result = fit_and_forecast(history.sort_index())

    assert result.insufficient_data is False
    assert result.chosen_model == "linear"
    assert result.r2_train > 0.99


def test_fit_and_forecast_extrapolates_the_trend():
    history = _linear_series(slope=3.0, intercept=10.0)

    result = fit_and_forecast(history)

    assert result.forecast_years == FORECAST_YEARS
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
    history.loc[HOLDOUT_YEAR] = 42.0

    result = fit_and_forecast(history.sort_index())

    assert result.holdout_year == HOLDOUT_YEAR
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


def test_fit_and_forecast_reports_baseline_skill_and_interval_coverage():
    result = fit_and_forecast(_linear_series())

    assert result.baseline_skill is not None
    assert result.baseline_skill > 0
    assert result.empirical_interval_coverage is not None
    assert 0 <= result.empirical_interval_coverage <= 1


def test_interval_coverage_abstains_when_no_folds_can_be_held_out():
    """A radius scored on the errors that produced it always "covers" ~90%.

    The metric must return None rather than that reassurance when the series is
    too short to keep any fold back.
    """
    short = pd.Series(
        {year: float(value) for year, value in zip(range(2016, 2022), range(10, 16), strict=True)}
    )
    result = fit_and_forecast(short)

    assert result.empirical_interval_coverage is None


def test_horizon_backtests_score_each_forecast_year_separately():
    """WP-18: a two-year projection scored one step ahead is not validated.

    The far end of the chart must either carry its own backtest or say it
    cannot be tested -- it must not silently inherit the one-year number.
    """
    import numpy as np

    from lake_research_map.dashboard.forecasting import fit_and_forecast

    years = list(range(2010, 2026))
    series = pd.Series([10 + 3 * i for i in range(len(years))], index=years, dtype=float)

    result = fit_and_forecast(series)

    assert set(result.horizon_backtests) == {1, 2}
    for _step, stats in result.horizon_backtests.items():
        assert set(stats) == {"cv_mae", "mase", "baseline_skill", "coverage"}
        if stats["mase"] is not None:
            assert np.isfinite(stats["mase"])
    # A clean linear series must beat a naive carry-forward at one step.
    assert result.mase is not None and result.mase < 1.0


def test_rolling_origin_cv_hides_the_forecast_window_at_longer_horizons():
    """A 2-step fold must not see the year immediately before its target."""
    import numpy as np

    from lake_research_map.dashboard.forecasting import _rolling_origin_cv

    years = np.arange(2010, 2024)
    # Flat through 2020, then a jump in 2021. Scoring 2022 one step ahead
    # trains through 2021 and sees the jump; two steps ahead stops at 2020 and
    # cannot. That is exactly the information a 2-year claim must not borrow.
    values = np.array([10.0] * 11 + [90.0] * 3)
    cv_years = (2022,)

    one_step = _rolling_origin_cv(years, values, cv_years, horizon=1)
    two_step = _rolling_origin_cv(years, values, cv_years, horizon=2)

    # The baseline carries the last observed value forward, so hiding an extra
    # year strictly costs it accuracy here.
    assert two_step["baseline"] > one_step["baseline"]


def test_bass_diffusion_reports_parameter_uncertainty():
    """`curve_fit` returns a covariance; the peak year must not be a bare point."""
    import numpy as np

    from lake_research_map.dashboard.forecasting import fit_bass_diffusion_nls

    years = np.arange(2005, 2026)
    steps = np.arange(len(years))
    p, q, m = 0.03, 0.4, 1000.0
    cumulative = (1 - np.exp(-(p + q) * steps)) / (1 + (q / p) * np.exp(-(p + q) * steps))
    clean = np.diff(np.concatenate([[0.0], cumulative * m]))

    exact = fit_bass_diffusion_nls(years, clean)
    assert exact["valid"] is True
    assert exact["p_stderr"] is not None and exact["q_stderr"] is not None
    assert exact["t_peak_low"] is not None and exact["t_peak_high"] is not None
    assert exact["t_peak_low"] <= exact["t_peak"] <= exact["t_peak_high"]

    # Noise must widen the interval rather than leave it unchanged.
    rng = np.random.default_rng(1)
    noisy = fit_bass_diffusion_nls(years, np.clip(clean + rng.normal(0, 40, len(clean)), 0, None))
    exact_width = exact["t_peak_high"] - exact["t_peak_low"]
    noisy_width = noisy["t_peak_high"] - noisy["t_peak_low"]
    assert noisy_width > exact_width
