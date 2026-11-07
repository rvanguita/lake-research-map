"""Citation distributions: heavy-tail fits, age normalization and the count GLMs
(Poisson / negative binomial / zero-inflated) behind the citation-determinants panel.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_heavy_tail_distributions(
    citation_counts: np.ndarray | pd.Series,
    *,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict:
    """Compare citation tails with fitted-parameter bootstrap diagnostics.

    Zeros are reported separately and never coerced into the positive tail.
    ``x_min`` is selected by minimizing the Pareto KS distance while retaining
    at least ten observations. Model comparison uses tail log-likelihood/AIC;
    bootstrap p-values refit the Pareto parameter in every simulated sample.
    """
    from scipy import stats

    raw = np.asarray(citation_counts, dtype=float)
    raw = raw[np.isfinite(raw) & (raw >= 0)]
    zero_count = int((raw == 0).sum())
    arr = raw[raw > 0]
    n = len(arr)
    if n < 10:
        return {
            "n": n,
            "zero_count": zero_count,
            "valid": False,
            "best_fit": "insufficient_data",
        }

    candidates = np.unique(np.quantile(arr, np.linspace(0, 0.8, 21)))
    selected: tuple[float, float, float, np.ndarray] | None = None
    for candidate in candidates:
        tail = arr[arr >= candidate]
        if len(tail) < 10:
            continue
        denominator = float(np.log(tail / candidate).sum())
        if denominator <= 0:
            continue
        alpha_candidate = 1.0 + len(tail) / denominator
        ks = float(
            stats.kstest(tail, stats.pareto(b=alpha_candidate - 1.0, scale=candidate).cdf)[0]
        )
        if selected is None or ks < selected[0]:
            selected = (ks, float(candidate), float(alpha_candidate), tail)
    if selected is None:
        return {
            "n": n,
            "zero_count": zero_count,
            "valid": False,
            "best_fit": "degenerate_tail",
        }
    ks_stat, x_min, alpha, tail = selected
    tail_n = len(tail)

    log_arr = np.log(tail)
    mu = float(np.mean(log_arr))
    sigma = float(np.std(log_arr, ddof=1))
    ks_lognorm = (
        stats.kstest(tail, stats.lognorm(s=sigma, scale=np.exp(mu)).cdf)
        if sigma > 0
        else (1.0, 0.0)
    )
    shifted = tail - x_min
    mean_val = float(np.mean(shifted))
    ks_expon = stats.kstest(shifted, stats.expon(scale=mean_val).cdf)
    pareto_dist = stats.pareto(b=alpha - 1.0, scale=x_min)
    lognorm_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    expon_dist = stats.expon(loc=x_min, scale=mean_val)
    log_likelihoods = {
        "power_law": float(np.log(pareto_dist.pdf(tail)).sum()),
        "log_normal": float(np.log(lognorm_dist.pdf(tail)).sum()),
        "exponential": float(np.log(expon_dist.pdf(tail)).sum()),
    }
    parameter_counts = {"power_law": 1, "log_normal": 2, "exponential": 1}

    rng = np.random.default_rng(random_state)
    bootstrap_ks = []
    for _ in range(max(0, n_bootstrap)):
        simulated = pareto_dist.rvs(size=tail_n, random_state=rng)
        denominator = float(np.log(simulated / x_min).sum())
        simulated_alpha = 1.0 + tail_n / denominator
        simulated_dist = stats.pareto(b=simulated_alpha - 1.0, scale=x_min)
        bootstrap_ks.append(float(stats.kstest(simulated, simulated_dist.cdf)[0]))
    bootstrap_p = (
        float(np.mean(np.asarray(bootstrap_ks) >= ks_stat)) if bootstrap_ks else float("nan")
    )

    fits = {
        "power_law": {
            "alpha": float(alpha),
            "x_min": float(x_min),
            "ks_stat": ks_stat,
            "p_value": bootstrap_p,
        },
        "log_normal": {
            "mu": mu,
            "sigma": sigma,
            "ks_stat": float(ks_lognorm[0]),
            "p_value": float(ks_lognorm[1]),
        },
        "exponential": {
            "scale": mean_val,
            "ks_stat": float(ks_expon[0]),
            "p_value": float(ks_expon[1]),
        },
    }
    for name, model in fits.items():
        model["log_likelihood"] = log_likelihoods[name]
        model["aic"] = 2 * parameter_counts[name] - 2 * log_likelihoods[name]
    best = min(fits, key=lambda name: fits[name]["aic"])
    return {
        "n": n,
        "tail_n": tail_n,
        "zero_count": zero_count,
        "valid": True,
        "best_fit": best,
        "models": fits,
        "log_likelihood_ratios": {
            "power_law_vs_log_normal": log_likelihoods["power_law"] - log_likelihoods["log_normal"],
            "power_law_vs_exponential": log_likelihoods["power_law"]
            - log_likelihoods["exponential"],
        },
    }


def age_normalized_citations(df: pd.DataFrame, *, observation_year: int) -> pd.DataFrame:
    """Calculate annual citation rates, cohort z-scores, and cohort percentiles.

    Prevents older papers from structurally dominating impact rankings.
    """
    if df.empty or "citation_count" not in df.columns or "year" not in df.columns:
        return df.copy()

    res = df.copy()
    years = pd.to_numeric(res["year"], errors="coerce")
    cites = pd.to_numeric(res["citation_count"], errors="coerce").fillna(0)

    age = (observation_year - years + 1).clip(lower=1)
    res["citation_rate_annual"] = cites / age

    cohort_mean = res.groupby(years)["citation_count"].transform("mean")
    cohort_std = res.groupby(years)["citation_count"].transform("std").replace(0, 1.0).fillna(1.0)
    res["cohort_citation_zscore"] = (cites - cohort_mean) / cohort_std
    res["cohort_citation_percentile"] = res.groupby(years)["citation_count"].rank(pct=True)
    return res


def _predicted_zero_fraction(family: str, fitted, mu: np.ndarray, alpha: float) -> float:
    """Model-implied share of zero counts, on the family's own terms.

    Each family answers "how often would this model produce a zero?" with a
    different formula, and the zero-inflation gap is only informative when the
    prediction comes from the family that was actually selected.
    """
    if family.startswith("zero_inflated"):
        try:
            probabilities = np.asarray(fitted.predict(which="prob"))
            if probabilities.ndim == 2 and probabilities.shape[1] > 0:
                return float(probabilities[:, 0].mean())
        except (AttributeError, ValueError, TypeError, NotImplementedError):
            pass
    if family == "negative_binomial" and alpha > 0:
        return float(np.mean((1.0 / (1.0 + alpha * mu)) ** (1.0 / alpha)))
    return float(np.mean(np.exp(-mu)))


def _pseudo_r_squared(fitted) -> float:
    """McFadden pseudo-R2, falling back to the GLM deviance ratio."""
    try:
        if getattr(fitted, "null_deviance", 0) > 0:
            return float(1 - fitted.deviance / fitted.null_deviance)
    except (AttributeError, TypeError):
        pass
    try:
        # `llnull` refits an intercept-only model behind the scenes; for a
        # zero-inflated family that refit can fail to invert its own Hessian.
        # The failure is already handled by returning NaN, so its warning is
        # noise rather than information.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llnull = float(fitted.llnull)
            log_likelihood = float(fitted.llf)
        if np.isfinite(llnull) and llnull != 0:
            return float(1 - log_likelihood / llnull)
    except (AttributeError, TypeError, ValueError):
        pass
    return float("nan")


def _fit_count_families(response, design, exposure) -> tuple[dict[str, object], dict[str, float]]:
    """Fit Poisson, negative binomial, ZIP and ZINB on the same design.

    Returns the converged fits and their AICs so the caller can select on
    evidence. The previous dispersion > 1.5 rule chose between only two of
    these and could never see a zero-inflated alternative at all, which is the
    specification a citation count most often needs.
    """
    from statsmodels.discrete.count_model import (
        ZeroInflatedNegativeBinomialP,
        ZeroInflatedPoisson,
    )
    from statsmodels.genmod.families import NegativeBinomial, Poisson
    from statsmodels.genmod.generalized_linear_model import GLM

    fits: dict[str, object] = {}
    aic: dict[str, float] = {}

    poisson = GLM(response, design, family=Poisson(), offset=exposure).fit(cov_type="HC3")
    fits["poisson"] = poisson
    aic["poisson"] = float(poisson.aic)
    dispersion = float(poisson.pearson_chi2 / max(poisson.df_resid, 1))
    alpha = max(dispersion - 1.0, 0.01)

    try:
        negative_binomial = GLM(
            response, design, family=NegativeBinomial(alpha=alpha), offset=exposure
        ).fit(cov_type="HC3")
        fits["negative_binomial"] = negative_binomial
        aic["negative_binomial"] = float(negative_binomial.aic)
    except (ValueError, np.linalg.LinAlgError):
        logger.debug("negative binomial did not converge", exc_info=True)

    # The zero-inflation part is deliberately intercept-only: with three
    # predictors and a few hundred complete rows, a fully specified inflation
    # equation is not identifiable and converges to noise.
    inflation = np.ones((len(response), 1))
    for name, model_class in (
        ("zero_inflated_poisson", ZeroInflatedPoisson),
        ("zero_inflated_negative_binomial", ZeroInflatedNegativeBinomialP),
    ):
        try:
            # `aic` and `bse` are lazy properties, so they must be read inside
            # this block too -- statsmodels raises its convergence warnings on
            # first access, not at fit time.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model_class(response, design, exog_infl=inflation, offset=exposure).fit(
                    disp=False, maxiter=200
                )
                candidate_aic = float(fitted.aic)
                parameters = np.asarray(fitted.params, dtype=float)
                # A fit whose Hessian could not be inverted has point estimates
                # but no standard errors, so its IRR intervals and p-values come
                # back NaN. Winning on AIC while being unable to state any
                # uncertainty is worse than losing to a family that can, so it
                # is not a candidate at all.
                standard_errors = np.asarray(fitted.bse, dtype=float)
            if not np.isfinite(candidate_aic) or not np.all(np.isfinite(parameters)):
                continue
            if standard_errors.size == 0 or not np.all(np.isfinite(standard_errors)):
                logger.debug("%s converged without usable standard errors", name)
                continue
            fits[name] = fitted
            aic[name] = candidate_aic
        except Exception:
            # Zero-inflated likelihoods fail to converge on small or
            # well-behaved samples often enough that this must not be fatal:
            # the caller reports the omission instead of hiding it.
            logger.debug("%s did not converge", name, exc_info=True)

    return fits, aic


def _age_specification_sensitivity(
    family: str,
    response,
    design,
    ages: np.ndarray,
    focal_features: list[str],
    alpha: float,
) -> list[dict]:
    """Refit under three exposure choices and report whether the signs survive.

    `log(age + 1)` as a fixed-coefficient offset is an assumption, not a
    finding: it forces citations to accumulate exactly proportionally to log
    age. The two covariate specifications let the data estimate that slope
    instead, so a predictor whose sign flips between them is not robust.
    """
    from statsmodels.genmod.families import NegativeBinomial, Poisson
    from statsmodels.genmod.generalized_linear_model import GLM

    log_age = np.log(np.clip(ages, 1, None) + 1.0)
    specifications = [
        ("offset_log_age", np.asarray(log_age), None),
        ("covariate_log_age", None, log_age),
        ("covariate_linear_age", None, np.asarray(ages, dtype=float)),
    ]
    # Zero-inflated families are compared under their Poisson/NB counterpart:
    # this block asks about the exposure term, not about the zero process. The
    # dispersion estimated on the real fit is reused -- statsmodels otherwise
    # silently falls back to alpha=1.0, which is a different model.
    base_family = (
        NegativeBinomial(alpha=max(alpha, 1e-6)) if "negative_binomial" in family else Poisson()
    )

    results: list[dict] = []
    for name, offset, covariate in specifications:
        spec_design = design.copy()
        if covariate is not None:
            standard_deviation = float(np.std(covariate))
            spec_design["age_term"] = (covariate - float(np.mean(covariate))) / (
                standard_deviation if standard_deviation > 0 else 1.0
            )
        try:
            fitted = GLM(response, spec_design, family=base_family, offset=offset).fit(
                cov_type="HC3"
            )
        except (ValueError, np.linalg.LinAlgError):
            logger.debug("age specification %r did not converge", name, exc_info=True)
            continue
        results.append(
            {
                "specification": name,
                "aic": float(fitted.aic),
                "coefficients": {
                    feature: float(fitted.params.get(feature, np.nan)) for feature in focal_features
                },
                "p_values": {
                    feature: float(fitted.pvalues.get(feature, np.nan))
                    for feature in focal_features
                },
            }
        )
    return results


def _signs_agree(specifications: list[dict], focal_features: list[str]) -> bool:
    """True when every predictor keeps its sign across all fitted specifications."""
    if len(specifications) < 2:
        return True
    for feature in focal_features:
        signs = {
            np.sign(spec["coefficients"][feature])
            for spec in specifications
            if np.isfinite(spec["coefficients"].get(feature, np.nan))
        }
        if len(signs) > 1:
            return False
    return True


def citation_determinants_glm(df: pd.DataFrame, *, observation_year: int) -> dict:
    """Fit an exposure-adjusted count GLM with robust uncertainty estimates.

    Citation counts accumulate over time, so ``log(article_age + 1)`` is used
    as an exposure offset. Numeric predictors are standardized and rows with
    missing predictors are excluded instead of silently filled. The count
    family is selected by AIC across Poisson, negative binomial and their
    zero-inflated counterparts, and the exposure choice is reported with a
    sensitivity block rather than assumed.
    """
    feature_names = ["qtd_referencias", "tamanho_equipe", "origem_ieee"]
    required = {"citation_count", "year", "reference_count", "authors", "source"}
    if df.empty or not required.issubset(df.columns):
        return {
            "valid": False,
            "features": feature_names,
            "coefficients": [],
            "irr": [],
            "n_total": len(df),
            "n_used": 0,
            "coverage": 0.0,
            "warning": "Absence of required fields.",
        }

    missingness = {
        column: float(df[column].isna().mean()) if column in df.columns else 1.0
        for column in required
    }
    valid = df.copy()
    valid["citations"] = pd.to_numeric(valid["citation_count"], errors="coerce")
    valid["publication_year"] = pd.to_numeric(valid["year"], errors="coerce")
    valid["references"] = pd.to_numeric(valid["reference_count"], errors="coerce")
    valid["team_size"] = valid["authors"].apply(
        lambda authors: len(authors) if isinstance(authors, list) and authors else np.nan
    )
    valid["is_ieee"] = valid["source"].map({"ieee": 1.0, "elsevier": 0.0})
    valid = valid.dropna(
        subset=["citations", "publication_year", "references", "team_size", "is_ieee"]
    )
    valid = valid[
        (valid["citations"] >= 0)
        & valid["publication_year"].between(1900, observation_year)
        & (valid["references"] >= 0)
        & (valid["team_size"] > 0)
    ]
    n_total = len(df)
    n_used = len(valid)
    coverage = n_used / n_total if n_total else 0.0
    if n_used < 20:
        return {
            "valid": False,
            "features": feature_names,
            "coefficients": [],
            "irr": [],
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": "Less than 20 complete observations.",
        }

    numeric = valid[["references", "team_size"]].astype(float)
    std = numeric.std(ddof=0).replace(0, 1.0)
    standardized = (numeric - numeric.mean()) / std
    design = pd.DataFrame(
        {
            "qtd_referencias": standardized["references"],
            "tamanho_equipe": standardized["team_size"],
            "origem_ieee": valid["is_ieee"].astype(float),
        },
        index=valid.index,
    )
    omitted = [column for column in feature_names if design[column].nunique() < 2]
    active_features = [column for column in feature_names if column not in omitted]
    design = design[active_features]
    if not active_features:
        return {
            "valid": False,
            "features": [],
            "coefficients": [],
            "irr": [],
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": "The predictors do not vary under this filter.",
        }
    from statsmodels.tools.tools import add_constant

    design = add_constant(design, has_constant="add")
    matrix = design.astype(float).to_numpy()
    condition_number = float(np.linalg.cond(matrix))
    vif: dict[str, float] = {}
    if len(active_features) > 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        for index, feature in enumerate(design.columns):
            if feature != "const":
                vif[feature] = float(variance_inflation_factor(matrix, index))
    response = valid["citations"].astype(float)
    ages = (observation_year - valid["publication_year"]).clip(lower=0).to_numpy(dtype=float)
    exposure = np.log(np.clip(ages, 1, None) + 1.0)

    try:
        fits, candidate_aic = _fit_count_families(response, design, exposure)
        poisson = fits["poisson"]
        dispersion = float(poisson.pearson_chi2 / max(poisson.df_resid, 1))
        alpha = max(dispersion - 1.0, 0.01)
        # Selection is now the minimum AIC over every family that converged,
        # not a threshold on dispersion. The dispersion is still reported
        # because it explains *why* a given family wins.
        family = min(candidate_aic, key=candidate_aic.get)
        fitted = fits[family]
        zero_inflated_status = (
            "fitted"
            if any(name.startswith("zero_inflated") for name in fits)
            else "did_not_converge"
        )

        coefficients = fitted.params.reindex(active_features)
        intervals = fitted.conf_int().reindex(active_features)
        # Cook's distance needs a hat matrix, which a zero-inflated MLE fit does
        # not have -- asking it for one yields a failed inversion and a NaN
        # dressed up as a diagnostic. Influence is therefore always reported
        # against the Poisson GLM, and `influence_basis` says so.
        influence_source = fitted if family in {"poisson", "negative_binomial"} else poisson
        influence_basis = family if influence_source is fitted else "poisson"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cooks_distance = np.asarray(
                    influence_source.get_influence().cooks_distance[0], dtype=float
                )
            influential_count = int(np.sum(cooks_distance > (4.0 / max(len(cooks_distance), 1))))
            max_cooks_distance = float(np.nanmax(cooks_distance, initial=0.0))
        except (AttributeError, ValueError, np.linalg.LinAlgError):
            influential_count = 0
            max_cooks_distance = float("nan")
            influence_basis = "unavailable"
        mu = np.exp(np.asarray(design.astype(float)) @ np.asarray(poisson.params) + exposure)
        observed_zero_fraction = float((response == 0).mean())
        predicted_zero_fraction = _predicted_zero_fraction(family, fitted, mu, alpha)
        age_specifications = _age_specification_sensitivity(
            family, response, design, ages, active_features, alpha
        )
        return {
            "valid": True,
            "features": active_features,
            "coefficients": coefficients.astype(float).tolist(),
            "irr": np.exp(coefficients).astype(float).tolist(),
            "irr_lower": np.exp(intervals[0]).astype(float).tolist(),
            "irr_upper": np.exp(intervals[1]).astype(float).tolist(),
            "p_values": fitted.pvalues.reindex(active_features).astype(float).tolist(),
            "family": family,
            "family_selection": "aic",
            "zero_inflated_status": zero_inflated_status,
            "dispersion": dispersion,
            "missingness": missingness,
            "condition_number": condition_number,
            "vif": vif,
            "candidate_aic": candidate_aic,
            "observed_zero_fraction": observed_zero_fraction,
            "predicted_zero_fraction": predicted_zero_fraction,
            "zero_inflation_gap": observed_zero_fraction - predicted_zero_fraction,
            "influential_count": influential_count,
            "max_cooks_distance": max_cooks_distance,
            "influence_basis": influence_basis,
            "age_specifications": age_specifications,
            "age_specification_signs_agree": _signs_agree(age_specifications, active_features),
            "score": _pseudo_r_squared(fitted),
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": (
                "Predictors without variation were omitted: " + ", ".join(omitted)
                if omitted
                else None
            ),
        }
    except (ValueError, np.linalg.LinAlgError, KeyError):
        return {
            "valid": False,
            "features": feature_names,
            "coefficients": [],
            "irr": [],
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": "The model did not converge under this filter.",
        }
