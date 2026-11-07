"""Corpus counts, coverage and the shared year window, plus generic trend statistics.

The year bound (`plausible_year_bound`, `valid_years`), the "count by source" and
cumulative shapes, metadata coverage, the PDF selection-bias table, keyword
explosion, and the trend tests the pages share (Mann-Kendall with the Hamed-Rao
correction, OLS slope with its interval, Benjamini-Hochberg). Every other
module in this package that needs a year window imports `valid_years` from here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from transform.publication_categories import PUBLICATION_CATEGORIES

OTHERS_LABEL = "Others"


# Shared "recent activity" window used by both the Overview and
# Researchers pages, so "recent" means the same thing (last 5 publication
# years, inclusive of the latest) everywhere it's shown.


RECENT_WINDOW_YEARS = 5


METADATA_COVERAGE_FIELDS = (
    "doi",
    "title",
    "year",
    "venue",
    "authors",
    "abstract",
    "keywords",
    "citation_count",
    "reference_count",
    "has_pdf",
)


PDF_BIAS_METRICS = {
    "year": "identity",
    "citation_count": "log1p",
    "reference_count": "log1p",
    "team_size": "identity",
    "abstract_chars": "log1p",
    "keyword_count": "log1p",
}


def plausible_year_bound() -> int:
    """Highest publication year a record may legitimately carry.

    Next calendar year, not this one: publishers assign an in-press record the
    year of its future issue, so the corpus genuinely contains rows ahead of
    today. Computed per call rather than at import, because the dashboard is a
    long-running process that would otherwise keep whichever bound was true
    when it started.
    """
    return datetime.now(UTC).year + 1


def valid_years(df: pd.DataFrame, lo: int = 1950, hi: int | None = None) -> pd.Series:
    """Coerce `year` to numeric and drop rows outside a plausible window.

    The guard exists for parse garbage -- a year of 9999 or 201 would dominate
    any rate or trend calculation -- not for records that are merely ahead of
    the calendar. `hi` was pinned at 2026 and silently discarded the 11 in-press
    2027 articles from every trend, production and keyword surface, which is
    exactly the kind of drop this function's own docstring claimed it spanned.
    It now follows `plausible_year_bound()`.

    Returns a Series aligned to `df`'s index (NaN where invalid), matching
    `pd.to_numeric(..., errors="coerce")` semantics -- callers that need only
    valid rows should `.dropna()` the result themselves.
    """
    years = pd.to_numeric(df.get("year"), errors="coerce")
    return years.where(years.between(lo, plausible_year_bound() if hi is None else hi))


def _field_present(series: pd.Series, field: str) -> pd.Series:
    """Return a boolean presence mask without treating unknown counts as zero."""
    if field in {"authors", "keywords", "sources"}:
        return series.apply(lambda value: isinstance(value, list) and len(value) > 0)
    if field == "has_pdf":
        return series.fillna(False).astype(bool)
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        return series.notna() & series.astype(str).str.strip().ne("")
    return series.notna()


def metadata_coverage_matrix(
    df: pd.DataFrame,
    fields: tuple[str, ...] = METADATA_COVERAGE_FIELDS,
) -> pd.DataFrame:
    """Measure field completeness by source using explicit denominators."""
    columns = ["field", "source", "n_total", "n_present", "coverage"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    if "source" not in working.columns:
        if "sources" in working.columns:
            working["source"] = working["sources"].apply(
                lambda values: values[0] if isinstance(values, list) and values else "unknown"
            )
        else:
            working["source"] = "total"

    rows: list[dict[str, object]] = []
    for source, group in working.groupby("source", dropna=False):
        for field in fields:
            if field not in group.columns:
                continue
            present = _field_present(group[field], field)
            n_total = len(group)
            n_present = int(present.sum())
            rows.append(
                {
                    "field": field,
                    "source": str(source),
                    "n_total": n_total,
                    "n_present": n_present,
                    "coverage": n_present / n_total if n_total else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled_variance = (
        (len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1)
    ) / (len(left) + len(right) - 2)
    if pooled_variance <= 0 or np.isclose(pooled_variance, 0.0):
        return 0.0 if np.isclose(np.mean(left), np.mean(right)) else float("nan")
    return float((np.mean(left) - np.mean(right)) / np.sqrt(pooled_variance))


def pdf_selection_bias(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare PDF and non-PDF subsets using standardized mean differences.

    Positive effects mean the PDF subset has a larger transformed mean. The
    bootstrap is stratified by PDF availability so group sizes remain fixed.
    """
    columns = [
        "metric",
        "transform",
        "n_pdf",
        "n_no_pdf",
        "mean_pdf",
        "mean_no_pdf",
        "smd",
        "ci_low",
        "ci_high",
        "status",
    ]
    if df.empty or "has_pdf" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["has_pdf"] = working["has_pdf"].fillna(False).astype(bool)
    working["team_size"] = working.get(
        "authors", pd.Series(index=working.index, dtype=object)
    ).apply(lambda value: len(value) if isinstance(value, list) else np.nan)
    working["abstract_chars"] = working.get(
        "abstract", pd.Series(index=working.index, dtype=object)
    ).apply(lambda value: len(value.strip()) if isinstance(value, str) else np.nan)
    working["keyword_count"] = working.get(
        "keywords", pd.Series(index=working.index, dtype=object)
    ).apply(lambda value: len(value) if isinstance(value, list) else np.nan)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for metric, transform in PDF_BIAS_METRICS.items():
        if metric not in working.columns:
            continue
        values = pd.to_numeric(working[metric], errors="coerce")
        valid = values.notna() & values.ge(0)
        pdf_values = values[valid & working["has_pdf"]].to_numpy(dtype=float)
        no_pdf_values = values[valid & ~working["has_pdf"]].to_numpy(dtype=float)
        if transform == "log1p":
            pdf_values = np.log1p(pdf_values)
            no_pdf_values = np.log1p(no_pdf_values)

        smd = _standardized_mean_difference(pdf_values, no_pdf_values)
        if len(pdf_values) < 2 or len(no_pdf_values) < 2:
            status = "insufficient_group"
        elif not np.isfinite(smd):
            status = "zero_variance"
        else:
            status = "ok"
        boot = np.array([], dtype=float)
        if n_bootstrap > 0 and len(pdf_values) >= 2 and len(no_pdf_values) >= 2:
            estimates = [
                _standardized_mean_difference(
                    rng.choice(pdf_values, size=len(pdf_values), replace=True),
                    rng.choice(no_pdf_values, size=len(no_pdf_values), replace=True),
                )
                for _ in range(n_bootstrap)
            ]
            boot = np.asarray([value for value in estimates if np.isfinite(value)], dtype=float)
        ci_low, ci_high = (
            np.quantile(boot, [0.025, 0.975]) if len(boot) else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "metric": metric,
                "transform": transform,
                "n_pdf": len(pdf_values),
                "n_no_pdf": len(no_pdf_values),
                "mean_pdf": float(np.mean(pdf_values)) if len(pdf_values) else np.nan,
                "mean_no_pdf": float(np.mean(no_pdf_values)) if len(no_pdf_values) else np.nan,
                "smd": smd,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


PDF_BIAS_LABELS = {
    "year": "Year",
    "citation_count": "Citations",
    "reference_count": "References",
    "team_size": "Team size",
    "abstract_chars": "Abstract length",
    "keyword_count": "Number of keywords",
}
# |SMD| below this is the conventional "negligible" difference.
NEGLIGIBLE_SMD = 0.1


def pdf_selection_bias_views(bias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a `pdf_selection_bias` table into what the chart and the audit table show.

    Returns `(finite, audit)`. `finite` keeps the rows with a finite SMD and adds
    `Metrics` (display label), `error_upper`/`error_lower` (distances from the SMD
    to its CI bounds), `not_negligible` (|SMD| >= `NEGLIGIBLE_SMD`) and
    `magnitude` (a hover label for that band). `audit` is every row, with
    display column names and values rounded to 3 places.
    """
    labelled = bias.assign(Metrics=bias["metric"].map(PDF_BIAS_LABELS))
    finite = labelled.dropna(subset=["smd"]).copy()
    finite["error_upper"] = finite["ci_high"] - finite["smd"]
    finite["error_lower"] = finite["smd"] - finite["ci_low"]
    finite["not_negligible"] = finite["smd"].abs().ge(NEGLIGIBLE_SMD)
    finite["magnitude"] = finite["not_negligible"].map(
        {
            True: f"|SMD| ≥ {NEGLIGIBLE_SMD}, not negligible",
            False: f"|SMD| < {NEGLIGIBLE_SMD}, negligible",
        }
    )
    audit = labelled[
        ["Metrics", "transform", "n_pdf", "n_no_pdf", "smd", "ci_low", "ci_high", "status"]
    ].rename(
        columns={
            "transform": "Transformation",
            "n_pdf": "n with PDF",
            "n_no_pdf": "n without PDF",
            "smd": "SMD",
            "ci_low": "CI 2.5%",
            "ci_high": "CI 97.5%",
            "status": "Status",
        }
    )
    return finite.reset_index(drop=True), audit.round(3)


def source_counts_by(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """Count rows per `index_col`, broken into ieee / elsevier / total columns.

    This is the "3 real series" shape every source-comparison chart in the
    refactored dashboard consumes -- replacing the dotted reference lines
    that used to carry the Total dimension.
    """
    if df.empty or index_col not in df.columns:
        return pd.DataFrame(columns=[index_col, "ieee", "elsevier", "total"])

    has_source = "source" in df.columns
    if has_source:
        pivot = df.groupby([index_col, "source"]).size().unstack(fill_value=0)
        for src in ("ieee", "elsevier"):
            if src not in pivot.columns:
                pivot[src] = 0
        pivot = pivot[["ieee", "elsevier"]]
    else:
        pivot = pd.DataFrame(index=df[index_col].dropna().unique())
        pivot["ieee"] = 0
        pivot["elsevier"] = 0

    pivot["total"] = df.groupby(index_col).size().reindex(pivot.index, fill_value=0)
    return pivot.reset_index().rename(columns={"index": index_col})


def publication_category_totals(df: pd.DataFrame) -> pd.Series:
    """Return mutually exclusive publication-category totals in stable order.

    Unknown or missing values are treated as ``other`` so the displayed
    category cards always reconcile to the corpus total.
    """
    counts = pd.Series(0, index=PUBLICATION_CATEGORIES, dtype="int64")
    if df.empty:
        return counts
    values = df.get("publication_category", pd.Series(index=df.index, dtype="object"))
    normalized = values.where(values.isin(PUBLICATION_CATEGORIES), "other").fillna("other")
    return counts.add(normalized.value_counts().reindex(PUBLICATION_CATEGORIES, fill_value=0))


def publication_category_counts_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Count publication categories by valid year, including a reconciled total."""
    columns = ["year", *PUBLICATION_CATEGORIES, "total"]
    if df.empty or "year" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["year"] = working["year"].astype(int)
    values = working.get("publication_category", pd.Series(index=working.index, dtype="object"))
    working["publication_category"] = values.where(
        values.isin(PUBLICATION_CATEGORIES), "other"
    ).fillna("other")
    grouped = (
        working.groupby(["year", "publication_category"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PUBLICATION_CATEGORIES, fill_value=0)
        .sort_index()
    )
    grouped["total"] = grouped[list(PUBLICATION_CATEGORIES)].sum(axis=1)
    return grouped.reset_index()[columns]


def cumulative_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative article count per year, split ieee / elsevier / total."""
    working = df.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"])
    if working.empty:
        return pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])
    working["year"] = working["year"].astype(int)

    counts = source_counts_by(working, "year").sort_values("year")
    for col in ("ieee", "elsevier", "total"):
        counts[col] = counts[col].cumsum()
    return counts.reset_index(drop=True)


def cumulative_by_category(
    df: pd.DataFrame, category_col: str, top_n: int = 10, scope: str = "total"
) -> pd.DataFrame:
    """Cumulative publications per year for the top values of `category_col` (+ an Outros bucket).

    `scope` restricts the underlying rows to "ieee", "elsevier", or "total"
    (all rows) before ranking categories and accumulating -- lets one chart
    answer "top X overall" vs. "top X within IEEE" without recomputing.
    """
    working = df.copy()
    if scope in ("ieee", "elsevier") and "source" in working.columns:
        working = working[working["source"] == scope]
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year", category_col])
    if working.empty:
        return pd.DataFrame(columns=["year", category_col, "cumulative"])
    working["year"] = working["year"].astype(int)

    top_values = working[category_col].value_counts().head(top_n).index.tolist()
    working["_bucket"] = working[category_col].where(
        working[category_col].isin(top_values), OTHERS_LABEL
    )

    by_year_bucket = working.groupby(["year", "_bucket"]).size().rename("count").reset_index()
    years = sorted(by_year_bucket["year"].unique())
    buckets = list(by_year_bucket["_bucket"].unique())
    full_index = pd.MultiIndex.from_product([years, buckets], names=["year", "_bucket"])
    filled = (
        by_year_bucket.set_index(["year", "_bucket"])["count"]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    filled["cumulative"] = filled.groupby("_bucket")["count"].cumsum()
    return filled.rename(columns={"_bucket": category_col})


def cumulative_by_venue(df: pd.DataFrame, top_n: int = 10, scope: str = "total") -> pd.DataFrame:
    """Cumulative publications per year for the top venues (+ an Outros bucket)."""
    return cumulative_by_category(df, "venue", top_n=top_n, scope=scope)


def source_means(df: pd.DataFrame, col: str) -> dict[str, float | None]:
    """Mean of `col` for ieee / elsevier / total, or None where unavailable.

    Replaces the `ieee_sub = df[df["source"] == "ieee"]; ... .mean()` pattern
    that used to feed the removed reference-line helper.
    """
    result: dict[str, float | None] = {"ieee": None, "elsevier": None, "total": None}
    if df.empty or col not in df.columns:
        return result
    if "source" in df.columns:
        for src in ("ieee", "elsevier"):
            sub = df[df["source"] == src]
            if not sub.empty:
                val = sub[col].mean()
                result[src] = float(val) if val == val else None
    total_val = df[col].mean()
    result["total"] = float(total_val) if total_val == total_val else None
    return result


def explode_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, keyword), keeping year/source/venue/doi."""
    if df.empty or "keywords" not in df.columns:
        return pd.DataFrame(columns=["keyword", "year", "source", "venue", "doi"])
    keep = [c for c in ("year", "source", "venue", "doi") if c in df.columns]
    working = df[["keywords", *keep]].copy()
    working = working.explode("keywords").rename(columns={"keywords": "keyword"})
    working = working[
        working["keyword"].notna() & (working["keyword"].astype(str).str.strip() != "")
    ]
    working["keyword"] = working["keyword"].astype(str).str.strip().str.lower()
    return working.reset_index(drop=True)


def _hamed_rao_variance_factor(y: np.ndarray, sen_slope: float) -> float:
    """Variance inflation for Mann-Kendall under serial dependence.

    Mann-Kendall assumes independent observations. Annual publication and
    keyword counts are not independent -- a busy year follows a busy year --
    and positive autocorrelation makes the unadjusted test reject far too
    often. Hamed and Rao (1998) rescale the variance by an effective sample
    size read off the autocorrelation of the *detrended ranks*, counting only
    the lags whose correlation clears the white-noise bound.

    Returns 1.0 when there is nothing to correct, so the caller can always
    multiply by it.
    """
    n = len(y)
    if n < 10:
        # Below ten points the lag correlations are themselves too noisy to
        # correct with, and trading one bias for another is not an improvement.
        return 1.0
    detrended = y - sen_slope * np.arange(n, dtype=float)
    ranks = pd.Series(detrended).rank().to_numpy()
    centered = ranks - ranks.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 1.0

    total = 0.0
    for lag in range(1, n - 2):
        rho = float(np.sum(centered[: n - lag] * centered[lag:]) / denominator)
        # Keep only autocorrelation distinguishable from white noise at 5%;
        # summing every lag would inflate the variance with pure noise.
        bound = 1.96 * np.sqrt(n - lag - 1) / (n - lag)
        if abs(rho + 1.0 / (n - lag)) <= bound:
            continue
        total += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho

    factor = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * total
    # A non-positive factor is a numerical artifact of a short series, not a
    # finding that the variance vanished.
    return float(factor) if factor > 0 else 1.0


def mann_kendall_trend(
    series: np.ndarray | pd.Series, *, serial_correction: bool = False
) -> dict[str, float | str]:
    """Returns trend direction ('growing', 'stable', 'falling'), p-value, S statistic, and slope.

    `p_value` stays the classical independent-observations test so existing
    callers are unchanged, and `p_value_serial_corrected` reports the same test
    under the Hamed-Rao variance inflation. Pass ``serial_correction=True`` to
    make `trend` and `p_value` follow the corrected value instead.
    """
    from scipy import stats

    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4:
        return {
            "trend": "stable",
            "p_value": 1.0,
            "s": 0.0,
            "slope": 0.0,
            "z": 0.0,
            "p_value_serial_corrected": 1.0,
            "serial_correction_factor": 1.0,
        }

    i_indices, j_indices = np.triu_indices(n, k=1)
    diffs = y[j_indices] - y[i_indices]
    s = float(np.sum(np.sign(diffs)))
    slopes = diffs / (j_indices - i_indices)

    sen_slope = float(np.median(slopes)) if len(slopes) > 0 else 0.0

    unique, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    def _z_and_p(variance: float) -> tuple[float, float]:
        if variance <= 0:
            return 0.0, 1.0
        if s > 0:
            statistic = (s - 1.0) / np.sqrt(variance)
        elif s < 0:
            statistic = (s + 1.0) / np.sqrt(variance)
        else:
            statistic = 0.0
        return float(statistic), float(2.0 * stats.norm.sf(abs(statistic)))

    z, p_value = _z_and_p(var_s)
    correction_factor = _hamed_rao_variance_factor(y, sen_slope)
    z_corrected, p_corrected = _z_and_p(var_s * correction_factor)

    # The corrected test is the one that holds when the series is
    # autocorrelated, but it stays opt-in: every existing caller reads
    # `p_value`, and silently changing what that means would rewrite published
    # trend tables without anybody asking for it.
    decisive_p = p_corrected if serial_correction else p_value
    if decisive_p < 0.05 and sen_slope > 0:
        trend = "growing"
    elif decisive_p < 0.05 and sen_slope < 0:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "p_value": decisive_p if serial_correction else p_value,
        "p_value_independent": p_value,
        "p_value_serial_corrected": p_corrected,
        "serial_correction_factor": correction_factor,
        "s": float(s),
        "z": float(z_corrected if serial_correction else z),
        "slope": sen_slope,
    }


def linear_slope_with_ci(
    x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, *, confidence: float = 0.95
) -> dict[str, float]:
    """OLS slope with the standard error and confidence interval it implies.

    A ranked table of bare slopes invites reading the order as a finding. Most
    of these series are a handful of years long, so the interval is frequently
    wide enough to contain zero even at the top of the ranking -- which is the
    thing the reader needs to see.
    """
    from scipy import stats

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    n = len(xs)
    nan = {
        "slope": float("nan"),
        "stderr": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "n": n,
    }
    if n < 3:
        return nan
    centered = xs - xs.mean()
    sxx = float(np.sum(centered**2))
    if sxx <= 0:
        return nan
    slope = float(np.sum(centered * (ys - ys.mean())) / sxx)
    intercept = float(ys.mean() - slope * xs.mean())
    residuals = ys - (intercept + slope * xs)
    dof = n - 2
    residual_var = float(np.sum(residuals**2)) / dof if dof > 0 else 0.0
    stderr = float(np.sqrt(residual_var / sxx)) if residual_var > 0 else 0.0
    if stderr == 0:
        # A perfect fit has no sampling spread to report; saying the interval
        # is the point estimate is honest, inventing one is not.
        return {
            "slope": slope,
            "stderr": 0.0,
            "ci_low": slope,
            "ci_high": slope,
            "p_value": 0.0,
            "n": n,
        }
    margin = float(stats.t.ppf(0.5 + confidence / 2.0, dof)) * stderr
    return {
        "slope": slope,
        "stderr": stderr,
        "ci_low": slope - margin,
        "ci_high": slope + margin,
        "p_value": float(2.0 * stats.t.sf(abs(slope / stderr), dof)),
        "n": n,
    }


def benjamini_hochberg(p_values: np.ndarray | pd.Series) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted
