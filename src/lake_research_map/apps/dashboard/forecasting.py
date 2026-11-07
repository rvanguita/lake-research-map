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
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

MIN_TRAIN_YEAR = 2010  # matches topics.TREND_MIN_YEAR -- earlier years are too sparse to trend
TRAIN_END_YEAR = int(os.environ.get("LAKE_RESEARCH_MAP_COMPLETE_YEAR", datetime.now(UTC).year - 1))
HOLDOUT_YEAR = TRAIN_END_YEAR + 1
FORECAST_YEARS = (HOLDOUT_YEAR + 1, HOLDOUT_YEAR + 2)
CV_YEARS = tuple(range(TRAIN_END_YEAR - 2, TRAIN_END_YEAR + 1))
# Folds kept out of the conformal radius so interval coverage is measurable
# rather than tautological. Small on purpose: annual series are short.
COVERAGE_HOLDOUT_FOLDS = 3

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
    baseline_skill: float | None = None
    empirical_interval_coverage: float | None = None
    # Per-horizon backtest: {h: {"cv_mae":…, "mase":…, "coverage":…}}. The
    # two-year projection used to be scored only one step ahead, so the second
    # year on the chart carried no validation at all.
    horizon_backtests: dict[int, dict[str, float | None]] = field(default_factory=dict)
    mase: float | None = None
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


def _naive_scale(values: np.ndarray) -> float:
    """Mean absolute one-step change: the denominator MASE is scaled by.

    MASE < 1 means the model beats a naive "next year equals this year"
    forecast on its own scale, which is what makes the keyword series and the
    corpus series comparable at all -- a raw MAE of 4 means something very
    different on a series averaging 12 than on one averaging 400.
    """
    if len(values) < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(values))))
    return scale if scale > 0 else float("nan")


def _rolling_origin_cv(
    years: np.ndarray, values: np.ndarray, cv_years: tuple[int, ...], *, horizon: int = 1
) -> dict[str, float]:
    """Mean CV MAE per candidate at a given forecast `horizon`.

    For each year in `cv_years`, train on every year at least `horizon` steps
    earlier and score against that year's actual value. At `horizon=1` this is
    the original one-step-ahead cross-validation; larger horizons are what make
    a two-year projection testable instead of merely plotted.
    """
    errors: dict[str, list[float]] = {kind: [] for kind in _CANDIDATES}
    for cv_year in cv_years:
        # The origin sits `horizon` years back, so nothing within the forecast
        # window is visible to the fit -- otherwise a 2-year claim is scored
        # with 1-year information and always looks better than it is.
        train_mask = years <= cv_year - horizon
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


def _holdout_interval_coverage(
    kind: str,
    years: np.ndarray,
    values: np.ndarray,
    *,
    n_holdout: int,
    quantile: float = 0.9,
    horizon: int = 1,
) -> float | None:
    """Fraction of held-out `horizon`-step errors covered by a radius fitted without them.

    Returns `None` when there is not enough history to both calibrate a radius on
    at least three folds and keep `n_holdout` folds back for scoring: an unknown
    coverage is more honest than one computed in sample.
    """
    errors = []
    for index in range(3, len(years) - horizon + 1):
        target = index + horizon - 1
        if target >= len(years):
            break
        try:
            predictor = _fit_model(kind, years[:index], values[:index])
        except Exception:
            logger.debug("_holdout_interval_coverage: %r failed at %r", kind, years[target])
            continue
        errors.append(abs(float(values[target]) - float(predictor([years[target]])[0])))
    if len(errors) < 3 + n_holdout:
        return None
    calibration, holdout = errors[:-n_holdout], errors[-n_holdout:]
    radius = float(np.quantile(calibration, quantile, method="higher"))
    return float(np.mean(np.asarray(holdout) <= radius))


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
            baseline_skill=None,
            empirical_interval_coverage=None,
            insufficient_data=True,
            notes=["Not enough training years (minimum 4) to fit a model."],
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
    # Coverage has to be measured on errors the radius was NOT fitted on. Taking
    # the 0.9 quantile of a set and then asking what fraction of that same set it
    # covers returns ~0.9 by construction and can never fail, which is exactly the
    # reassurance a conformal band must not manufacture. Hold out the most recent
    # folds instead, and report nothing when the series is too short to spare any.
    empirical_coverage = _holdout_interval_coverage(
        chosen_model, final_years, final_values, n_holdout=COVERAGE_HOLDOUT_FOLDS
    )
    baseline_mae = cv_mae_by_model.get("baseline", float("nan"))
    chosen_mae = cv_mae_by_model.get(chosen_model, float("nan"))
    baseline_skill = (
        float(1.0 - chosen_mae / baseline_mae)
        if np.isfinite(chosen_mae) and np.isfinite(baseline_mae) and baseline_mae > 0
        else None
    )
    # MASE puts the error on the series' own scale, so the corpus series and a
    # sparse keyword series can be compared at all; > 1 means worse than naive.
    naive_scale = _naive_scale(final_values)
    mase = (
        float(chosen_mae / naive_scale)
        if np.isfinite(chosen_mae) and np.isfinite(naive_scale)
        else None
    )

    # Every fold above scores one step ahead, which left the second forecast
    # year unvalidated. Re-score the chosen model at each horizon actually
    # projected, so the chart's far end has evidence behind it or is silent.
    horizon_backtests: dict[int, dict[str, float | None]] = {}
    for step, _year in enumerate(forecast_years, start=1):
        horizon_cv = _rolling_origin_cv(final_years, final_values, cv_years, horizon=step)
        horizon_mae = horizon_cv.get(chosen_model, float("nan"))
        horizon_base = horizon_cv.get("baseline", float("nan"))
        horizon_backtests[step] = {
            "cv_mae": float(horizon_mae) if np.isfinite(horizon_mae) else None,
            "mase": (
                float(horizon_mae / naive_scale)
                if np.isfinite(horizon_mae) and np.isfinite(naive_scale)
                else None
            ),
            "baseline_skill": (
                float(1.0 - horizon_mae / horizon_base)
                if np.isfinite(horizon_mae) and np.isfinite(horizon_base) and horizon_base > 0
                else None
            ),
            "coverage": _holdout_interval_coverage(
                chosen_model,
                final_years,
                final_values,
                n_holdout=COVERAGE_HOLDOUT_FOLDS,
                horizon=step,
            ),
        }
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
            f"{holdout_year} is partial: it is shown for monitoring only and takes no part "
            "in model selection or the final fit."
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
        baseline_skill=baseline_skill,
        empirical_interval_coverage=empirical_coverage,
        horizon_backtests=horizon_backtests,
        mase=mase,
        notes=notes,
    )


# --- Quantile Regression & Bass Diffusion -----------------------------------


def _bass_cumulative(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    """Continuous cumulative Bass diffusion function F(t) * m."""
    pq = p + q
    exp_term = np.exp(-pq * t)
    return m * (1.0 - exp_term) / (1.0 + (q / max(1e-9, p)) * exp_term)


def _bass_standard_errors(pcov: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Parameter standard errors from the fit covariance, or None when unusable.

    `curve_fit` returns inf on the diagonal when a parameter is not identified
    by the data; reporting that as a number would be worse than reporting
    nothing.
    """
    try:
        diagonal = np.diag(np.asarray(pcov, dtype=float))
    except (ValueError, TypeError):
        return (None, None, None)
    out: list[float | None] = []
    for variance in diagonal[:3]:
        out.append(
            round(float(np.sqrt(variance)), 4) if np.isfinite(variance) and variance >= 0 else None
        )
    while len(out) < 3:
        out.append(None)
    return (out[0], out[1], out[2])


def _bass_peak_interval(
    popt: np.ndarray, pcov: np.ndarray, first_year: float, *, draws: int = 500, seed: int = 0
) -> tuple[float | None, float | None]:
    """80% interval for the diffusion peak year, by parametric bootstrap.

    The peak is a non-linear function of p and q, so its uncertainty cannot be
    read off their standard errors directly. Sampling the fitted covariance and
    recomputing the peak each time is the cheapest honest answer.
    """
    covariance = np.asarray(pcov, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return (None, None)
    rng = np.random.default_rng(seed)
    try:
        samples = rng.multivariate_normal(np.asarray(popt, dtype=float), covariance, size=draws)
    except (np.linalg.LinAlgError, ValueError):
        return (None, None)
    peaks = []
    for p_s, q_s, _m_s in samples:
        if p_s <= 0 or q_s <= 0 or p_s >= q_s:
            continue
        peaks.append(first_year + np.log(q_s / p_s) / (p_s + q_s))
    if len(peaks) < draws // 10:
        # Too few draws produced an interior peak for a quantile to mean much.
        return (None, None)
    return (round(float(np.quantile(peaks, 0.1)), 1), round(float(np.quantile(peaks, 0.9)), 1))


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
        popt, pcov = curve_fit(
            _bass_cumulative, t_relative, y_cum, p0=p0, bounds=bounds, maxfev=2000
        )
        p_est, q_est, m_est = float(popt[0]), float(popt[1]), float(popt[2])

        t_peak_offset = np.log(q_est / p_est) / (p_est + q_est) if p_est < q_est else 0.0
        t_peak = float(years[0] + t_peak_offset)

        # `curve_fit` already returns the covariance; discarding it meant the
        # peak year -- the one number the Trends page actually asserts -- was
        # shown as a point estimate with no spread. A parametric bootstrap over
        # the fitted covariance gives that spread without refitting the series.
        standard_errors = _bass_standard_errors(pcov)
        peak_low, peak_high = _bass_peak_interval(popt, pcov, float(years[0]))

        return {
            "valid": True,
            "m": round(m_est, 1),
            "p": round(p_est, 4),
            "q": round(q_est, 4),
            "t_peak": round(t_peak, 1),
            "p_stderr": standard_errors[0],
            "q_stderr": standard_errors[1],
            "m_stderr": standard_errors[2],
            "t_peak_low": peak_low,
            "t_peak_high": peak_high,
            "stage": ("growth" if t_peak > float(years[-1]) else "maturity"),
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
                    "stage": ("growth" if t_peak > float(years[-1]) else "maturity"),
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
