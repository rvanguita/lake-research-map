"""Publication-volume forecasting: candidate regression models, validated by a
held-out year plus rolling-origin cross-validation, then used to project two
years ahead.

Pure pandas/sklearn -- no `streamlit` import, mirroring `analytics.py` -- so
the modeling logic can be reasoned about (and unit-tested) independently of
the page that renders it.

Why regression instead of a heavier ML model: the usable series is short
(~16 yearly points, 2010-2025 -- the same cutoff `topics.TREND_MIN_YEAR` uses
for keyword trends). A random forest or neural net would overfit a series
this small; three simple regressions plus model selection by validation
error is the appropriate amount of machine learning for the amount of data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

MIN_TRAIN_YEAR = 2010  # matches topics.TREND_MIN_YEAR -- earlier years are too sparse to trend
TRAIN_END_YEAR = 2025  # last *complete* year in the corpus
HOLDOUT_YEAR = 2026  # current year at collection time -- a partial year, not a complete one
FORECAST_YEARS = (2027, 2028)
CV_YEARS = (2023, 2024, 2025)  # rolling-origin validation over the last complete years

_CANDIDATES = ("baseline", "linear", "log_linear")


def _fit_model(kind: str, years: np.ndarray, values: np.ndarray):
    x = years.reshape(-1, 1)
    if kind == "baseline":
        level = float(values[-1])
        return lambda yrs: np.full(len(np.asarray(yrs)), level)
    if kind == "linear":
        model = LinearRegression().fit(x, values)
        return lambda yrs: model.predict(np.asarray(yrs).reshape(-1, 1))
    if kind == "log_linear":
        model = LinearRegression().fit(x, np.log1p(values))
        return lambda yrs: np.expm1(model.predict(np.asarray(yrs).reshape(-1, 1)))
    raise ValueError(f"unknown model kind: {kind}")


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


@dataclass
class ForecastResult:
    history: pd.Series  # full observed series, MIN_TRAIN_YEAR..(latest available year)
    train_years: np.ndarray
    fitted_curve: pd.Series  # model's fit over the training range, for display
    holdout_year: int
    holdout_actual: float | None
    holdout_predicted: float | None
    cv_mae: float | None  # mean MAE across CV_YEARS rolling-origin folds
    model_comparison: pd.DataFrame  # one row per candidate: holdout MAE, CV MAE
    chosen_model: str
    forecast_years: tuple[int, ...]
    forecast_values: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    r2_train: float
    insufficient_data: bool = False
    notes: list[str] = field(default_factory=list)


def yearly_counts(df: pd.DataFrame, source: str | None = None) -> pd.Series:
    """Article counts per year, optionally restricted to one source.

    Returns a series indexed by int year, covering every year from the
    earliest to the latest observed (gaps filled with 0) so a regression over
    `year` doesn't silently skip missing years.
    """
    working = df.copy()
    if source is not None:
        if "source" in working.columns:
            working = working[working["source"] == source]
        elif "sources" in working.columns:
            working = working[
                working["sources"].apply(lambda s: isinstance(s, list) and source in s)
            ]
    years = pd.to_numeric(working.get("year"), errors="coerce").dropna().astype(int)
    if years.empty:
        return pd.Series(dtype=float)
    counts = years.value_counts().sort_index()
    full_index = range(int(counts.index.min()), int(counts.index.max()) + 1)
    return counts.reindex(full_index, fill_value=0).astype(float)


def _rolling_origin_cv(
    years: np.ndarray, values: np.ndarray, cv_years: tuple[int, ...]
) -> dict[str, float]:
    """Mean CV MAE per candidate: for each year in `cv_years`, train on every
    earlier year in `years` and score against that year's actual value.
    """
    errors: dict[str, list[float]] = {kind: [] for kind in _CANDIDATES}
    for cv_year in cv_years:
        train_mask = years < cv_year
        if train_mask.sum() < 3 or cv_year not in years:
            continue
        train_x, train_y = years[train_mask], values[train_mask]
        actual = values[years == cv_year][0]
        for kind in _CANDIDATES:
            try:
                predict = _fit_model(kind, train_x, train_y)
                errors[kind].append(_mae([actual], predict([cv_year])))
            except Exception:
                # Some candidate model kinds (e.g. log-linear on non-positive
                # values) can't fit every fold -- skip that kind for this fold.
                logger.debug(
                    "_rolling_origin_cv: %r failed to fit for year %r", kind, cv_year, exc_info=True
                )
                continue
    return {kind: float(np.mean(v)) if v else float("nan") for kind, v in errors.items()}


def fit_and_forecast(
    series: pd.Series,
    *,
    min_train_year: int = MIN_TRAIN_YEAR,
    train_end_year: int = TRAIN_END_YEAR,
    holdout_year: int = HOLDOUT_YEAR,
    forecast_years: tuple[int, ...] = FORECAST_YEARS,
    cv_years: tuple[int, ...] = CV_YEARS,
) -> ForecastResult:
    """Select a parsimonious model by rolling-origin CV and forecast ahead.

    The incomplete holdout year is reported for monitoring but never enters
    model selection or final training. Prediction bands use the empirical
    90th percentile of rolling one-step errors (conformal calibration).
    """
    notes: list[str] = []
    history = series[series.index >= min_train_year]
    train = history[history.index <= train_end_year]

    if len(train) < 4:
        empty = np.array([])
        return ForecastResult(
            history=history,
            train_years=empty,
            fitted_curve=pd.Series(dtype=float),
            holdout_year=holdout_year,
            holdout_actual=None,
            holdout_predicted=None,
            cv_mae=None,
            model_comparison=pd.DataFrame(),
            chosen_model="none",
            forecast_years=forecast_years,
            forecast_values=empty,
            forecast_lower=empty,
            forecast_upper=empty,
            r2_train=float("nan"),
            insufficient_data=True,
            notes=["Years of insufficient training (minimum 4) to adjust a model."],
        )

    train_years = train.index.to_numpy()
    train_values = train.to_numpy()

    holdout_actual = float(history.loc[holdout_year]) if holdout_year in history.index else None

    cv_mae_by_model = _rolling_origin_cv(train_years, train_values, cv_years)

    rows = []
    for kind in _CANDIDATES:
        predict = _fit_model(kind, train_years, train_values)
        holdout_pred = (
            float(predict([holdout_year])[0]) if holdout_actual is not None else float("nan")
        )
        holdout_mae = (
            abs(holdout_pred - holdout_actual) if holdout_actual is not None else float("nan")
        )
        rows.append(
            {
                "model": kind,
                "holdout_mae": holdout_mae,
                "cv_mae": cv_mae_by_model.get(kind, float("nan")),
            }
        )
    comparison = pd.DataFrame(rows)

    comparison["combined_mae"] = comparison["cv_mae"].fillna(float("inf"))
    chosen_model = comparison.loc[comparison["combined_mae"].idxmin(), "model"]

    # Refit on complete years only. The current partial year must not pull the
    # forecast curve down merely because ingestion is still under way.
    final_train = train
    final_years = final_train.index.to_numpy()
    final_values = final_train.to_numpy()
    final_predict = _fit_model(chosen_model, final_years, final_values)

    fitted_curve = pd.Series(final_predict(final_years), index=final_years)
    residuals = final_values - fitted_curve.to_numpy()
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((final_values - final_values.mean()) ** 2))
    r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    forecast_values = final_predict(np.array(forecast_years))
    forecast_values = np.clip(forecast_values, 0, None)  # counts can't be negative

    # Conformal calibration from one-step rolling-origin errors. This avoids a
    # normality assumption that is hard to justify for a short count series.
    calibration_errors = []
    for index in range(3, len(final_years)):
        predictor = _fit_model(chosen_model, final_years[:index], final_values[:index])
        calibration_errors.append(
            abs(float(final_values[index]) - float(predictor([final_years[index]])[0]))
        )
    conformal_radius = (
        float(np.quantile(calibration_errors, 0.9, method="higher"))
        if calibration_errors
        else float(np.max(np.abs(residuals), initial=0.0))
    )
    step_factors = np.sqrt(np.arange(1, len(forecast_years) + 1, dtype=float))
    margin = conformal_radius * step_factors
    forecast_lower = np.clip(forecast_values - margin, 0, None)
    forecast_upper = forecast_values + margin

    # Holdout prediction for the *chosen* model, fit on train-only years (not
    # the final refit, which already includes the holdout year in training).
    holdout_predicted = None
    if holdout_actual is not None:
        train_only_predict = _fit_model(chosen_model, train_years, train_values)
        holdout_predicted = float(train_only_predict([holdout_year])[0])
        notes.append(
            f"{holdout_year} is partial: it is shown only for monitoring and is not used "
            "of selection or final adjustment."
        )
    if (train_values.max() if len(train_values) else 0) < 20:
        notes.append("Low volume series — the confidence band is proportionally wider.")

    return ForecastResult(
        history=history,
        train_years=train_years,
        fitted_curve=fitted_curve,
        holdout_year=holdout_year,
        holdout_actual=holdout_actual,
        holdout_predicted=holdout_predicted,
        cv_mae=cv_mae_by_model.get(chosen_model),
        model_comparison=comparison,
        chosen_model=chosen_model,
        forecast_years=forecast_years,
        forecast_values=forecast_values,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        r2_train=r2_train,
        notes=notes,
    )


# --- Quantile Regression & Bass Diffusion -----------------------------------


def fit_quantile_forecast(
    years: np.ndarray,
    values: np.ndarray,
    forecast_years: tuple[int, ...] = FORECAST_YEARS,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> dict:
    """Fit asymmetric quantile regression curves for empirical uncertainty quantification."""
    from sklearn.linear_model import LinearRegression, QuantileRegressor

    if len(values) < 5 or np.all(values == 0):
        return {
            "valid": False,
            "forecast_years": forecast_years,
            "p10": np.zeros(len(forecast_years)),
            "p50": np.zeros(len(forecast_years)),
            "p90": np.zeros(len(forecast_years)),
        }

    x = years.reshape(-1, 1)
    x_future = np.array(forecast_years).reshape(-1, 1)
    results = {}

    for q in quantiles:
        try:
            model = QuantileRegressor(quantile=q, alpha=0.1, solver="highs").fit(x, values)
            pred = np.clip(model.predict(x_future), 0, None)
        except Exception:
            base_model = LinearRegression().fit(x, values)
            base_pred = base_model.predict(x_future)
            res = values - base_model.predict(x)
            q_res = float(np.quantile(res, q))
            pred = np.clip(base_pred + q_res, 0, None)
        results[f"p{int(q * 100)}"] = pred

    return {
        "valid": True,
        "forecast_years": forecast_years,
        "p10": results.get("p10", np.zeros(len(forecast_years))),
        "p50": results.get("p50", np.zeros(len(forecast_years))),
        "p90": results.get("p90", np.zeros(len(forecast_years))),
    }


def _bass_cumulative(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    """Continuous cumulative Bass diffusion function F(t) * m."""
    pq = p + q
    exp_term = np.exp(-pq * t)
    return m * (1.0 - exp_term) / (1.0 + (q / max(1e-9, p)) * exp_term)


def fit_bass_diffusion_nls(
    years: np.ndarray,
    annual_adoptions: np.ndarray,
) -> dict:
    """Fit the continuous Bass Diffusion Model via Non-Linear Least Squares (NLS).

    Directly estimates innovation (p), imitation (q), and market capacity (m)
    by fitting cumulative adoptions with scipy.optimize.curve_fit. Overcomes
    the collinearity and negative curvature issues of discrete OLS.
    """
    from scipy.optimize import curve_fit

    n = len(annual_adoptions)
    if n < 5 or np.sum(annual_adoptions) <= 0:
        return {"valid": False, "m": 0.0, "p": 0.0, "q": 0.0, "t_peak": None, "method": "nls"}

    t_relative = np.arange(n, dtype=float)
    y_cum = np.cumsum(annual_adoptions).astype(float)
    total_obs = float(y_cum[-1])

    # Initial parameter guess: standard diffusion priors
    p0 = [0.03, 0.38, max(total_obs * 1.3, 10.0)]
    bounds = ([1e-5, 1e-5, total_obs], [0.5, 1.5, total_obs * 20.0])

    try:
        popt, _ = curve_fit(_bass_cumulative, t_relative, y_cum, p0=p0, bounds=bounds, maxfev=2000)
        p_est, q_est, m_est = float(popt[0]), float(popt[1]), float(popt[2])

        t_peak_offset = np.log(q_est / p_est) / (p_est + q_est) if p_est < q_est else 0.0
        t_peak = float(years[0] + t_peak_offset)

        return {
            "valid": True,
            "m": round(m_est, 1),
            "p": round(p_est, 4),
            "q": round(q_est, 4),
            "t_peak": round(t_peak, 1),
            "stage": ("crescimento" if t_peak > float(years[-1]) else "maturidade"),
            "method": "nls",
        }
    except Exception:
        # Fallback to discrete OLS
        return fit_bass_diffusion(years, annual_adoptions)


def fit_bass_diffusion(
    years: np.ndarray,
    annual_adoptions: np.ndarray,
) -> dict:
    """Fit the classic Bass Diffusion Model (Bass, 1969) to emerging topic lifecycles.

    Uses discrete OLS formulation with automatic fallback to NLS optimization.
    """
    if len(annual_adoptions) < 5 or np.sum(annual_adoptions) <= 0:
        return {"valid": False, "m": 0.0, "p": 0.0, "q": 0.0, "t_peak": None}

    y_cum = np.cumsum(annual_adoptions)
    s_t = annual_adoptions[1:]
    y_prev = y_cum[:-1]
    y_prev_sq = y_prev**2

    X = np.column_stack([np.ones_like(y_prev), y_prev, y_prev_sq])
    try:
        betas, _, _, _ = np.linalg.lstsq(X, s_t, rcond=None)
        b0, b1, b2 = float(betas[0]), float(betas[1]), float(betas[2])

        disc = b1**2 - 4 * b0 * b2
        if b2 < 0 and disc > 0:
            m = (-b1 - np.sqrt(disc)) / (2 * b2)
            p = b0 / m
            q = -m * b2
            if p > 0 and q > 0:
                t_peak_offset = np.log(q / p) / (p + q) if p < q else 0.0
                t_peak = float(years[0] + t_peak_offset)
                return {
                    "valid": True,
                    "m": float(m),
                    "p": float(p),
                    "q": float(q),
                    "t_peak": t_peak,
                    "stage": ("crescimento" if t_peak > float(years[-1]) else "maturidade"),
                    "method": "ols",
                }
    except Exception:
        pass

    return {
        "valid": False,
        "m": float(np.sum(annual_adoptions) * 1.5),
        "p": 0.03,
        "q": 0.38,
        "t_peak": None,
        "method": "fallback",
    }
