"""Pure pandas aggregations shared by every dashboard page.

No `streamlit` import here on purpose -- these functions are plain data
transforms and can be exercised/cached independently of the UI layer. Pages
call these instead of open-coding a `groupby`/`explode`, so the same year
window, the same "count by source" shape, and the same author-name
normalization are used everywhere.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

OTHERS_LABEL = "Outros"

# Shared "recent activity" window used by both the Visão Geral and
# Pesquisadores pages, so "recent" means the same thing (last 5 publication
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


def valid_years(df: pd.DataFrame, lo: int = 1950, hi: int = 2026) -> pd.Series:
    """Coerce `year` to numeric and drop rows outside a plausible window.

    The corpus spans 1926-2027 (in-press records included), but a handful of
    very old or future-dated rows would otherwise dominate any rate/trend
    calculation. Returns a Series aligned to `df`'s index (NaN where invalid),
    matching `pd.to_numeric(..., errors="coerce")` semantics -- callers that
    need only valid rows should `.dropna()` the result themselves.
    """
    years = pd.to_numeric(df.get("year"), errors="coerce")
    return years.where(years.between(lo, hi))


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


def author_count_series(df: pd.DataFrame) -> pd.Series:
    """Authors per row, from the `authors` list column.

    A list counts by its length; a non-null non-list scalar (a defensive
    fallback for a stray single-author value that never got wrapped in a
    list) counts as 1; null counts as 0. Shared by every page that needs an
    "authors per article" distribution or mean, so this edge case is handled
    the same way everywhere instead of drifting between pages.
    """
    if "authors" not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df["authors"].apply(
        lambda a: len(a) if isinstance(a, list) else (1 if pd.notna(a) else 0)
    )


def keyword_count_series(df: pd.DataFrame) -> pd.Series:
    """Keywords per row, from the `keywords` list column -- mirrors `author_count_series`."""
    if "keywords" not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df["keywords"].apply(
        lambda k: len(k) if isinstance(k, list) else (1 if pd.notna(k) else 0)
    )


def explode_authors_with_position(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, author), plus the author's 0-based byline position.

    Position 0 is the first-listed author, 1 the second, etc. -- the order the
    source `.bib`/CSV author field was written in (not alphabetical; see
    `transform.bronze_articles._split_bibtex_authors`/`_split_ieee_csv_authors`).
    It's a positional count only, not a claim about authorship role -- what a
    given position conventionally means varies by field, and this corpus
    doesn't record roles.
    """
    columns = ["author", "position", "year", "source", "venue", "doi"]
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame(columns=columns)
    keep = [c for c in ("year", "source", "venue", "doi", "citation_count") if c in df.columns]
    working = df[["authors", *keep]].copy()
    working["authors"] = working["authors"].apply(
        lambda lst: list(enumerate(lst)) if isinstance(lst, list) else np.nan
    )
    working = working.explode("authors")
    working = working[working["authors"].apply(lambda v: isinstance(v, tuple))]
    if working.empty:
        return pd.DataFrame(columns=columns)
    # `.explode()` repeats the original row index for every exploded entry, so a
    # positional (not index-aligned) assignment is required here -- joining two
    # frames that both carry duplicate index labels would cross-join within each
    # duplicated label instead of pairing rows one-to-one.
    tuples = working["authors"].tolist()
    working = working.drop(columns=["authors"])
    working["position"] = [t[0] for t in tuples]
    working["author"] = [t[1] for t in tuples]
    working = working[working["author"].notna() & (working["author"].astype(str).str.strip() != "")]
    return working.reset_index(drop=True)


def explode_authors(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, author), keeping year/source/citation_count/venue."""
    return explode_authors_with_position(df).drop(columns=["position"])


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


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


_INITIAL_RE = re.compile(r"^[A-Za-z]\.?$")


def canonical_author(name: str) -> str:
    """Fold an author name to a rough identity key: "initial surname", lowercase.

    Handles the three formats present in the corpus: "M. Parvania" (IEEE
    CSV/bib), "Fernando Postigo" (Elsevier, full first name), and
    "Liu, Junyong" (IEEE bib, "Last, First"). This deliberately merges
    "Junyong Liu" and "J. Liu" into the same key -- a real simplification,
    not an identity match. Pages using this must say so: it can also merge
    distinct people who share an initial and surname.
    """
    if not name or not isinstance(name, str):
        return ""
    cleaned = name.strip().strip(".")
    if not cleaned:
        return ""

    if "," in cleaned:
        last, _, first = cleaned.partition(",")
        parts = [first.strip(), last.strip()]
    else:
        parts = cleaned.split()

    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        surname = parts[0]
        initial = ""
    else:
        surname = parts[-1]
        first_token = parts[0]
        initial = first_token[0] if first_token else ""

    surname = _strip_accents(surname).lower()
    initial = _strip_accents(initial).lower()
    return f"{initial} {surname}".strip()


def author_display_name(names: pd.Series) -> str:
    """Pick the most informative spelling among variants that share a key."""
    if names.empty:
        return ""
    return max(names.unique(), key=len)


_MATRIX_SUMMARY_COLS = ("total", "ieee_total", "elsevier_total")


def author_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical author, one column per valid year, plus source totals.

    Explodes `authors`, folds names via `canonical_author` (see its docstring
    for the "initial surname" identity-merge caveat -- any page showing this
    table must disclose it), and counts distinct articles per author per year
    via `doi` where available (falls back to row count otherwise, since two
    exploded rows for the same article/author pair would otherwise double-count
    a co-authored paper). Sorted descending by `total` per the page's "highest
    to lowest" requirement. Years outside `valid_years`' plausible window are
    dropped before pivoting, same as every other year-based aggregation here.

    `ieee_total`/`elsevier_total` break `total` down by source (0 when a
    `source` column isn't present), using the same distinct-DOI counting rule
    as the year columns -- the "3 real series" convention this dashboard uses
    everywhere else (see `source_counts_by`).
    """
    exploded = explode_authors(df)
    if exploded.empty:
        return pd.DataFrame(columns=["author", *_MATRIX_SUMMARY_COLS])

    exploded["author_key"] = exploded["author"].apply(canonical_author)
    exploded = exploded[exploded["author_key"] != ""]
    exploded["year"] = valid_years(exploded)
    exploded = exploded.dropna(subset=["year"]).astype({"year": int})
    if exploded.empty:
        return pd.DataFrame(columns=["author", *_MATRIX_SUMMARY_COLS])

    display_names = exploded.groupby("author_key")["author"].apply(author_display_name)
    exploded["author_display"] = exploded["author_key"].map(display_names)

    count_col = "doi" if "doi" in exploded.columns else "author"
    agg = "nunique" if count_col == "doi" else "size"
    pivot = (
        exploded.groupby(["author_display", "year"])[count_col]
        .agg(agg)
        .unstack(fill_value=0)
        .astype(int)
    )
    pivot.columns = [str(int(c)) for c in pivot.columns]
    pivot["total"] = pivot.sum(axis=1)

    if "source" in exploded.columns:
        src_pivot = (
            exploded.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in src_pivot.columns:
                src_pivot[src] = 0
        pivot["ieee_total"] = src_pivot["ieee"].reindex(pivot.index, fill_value=0).astype(int)
        pivot["elsevier_total"] = (
            src_pivot["elsevier"].reindex(pivot.index, fill_value=0).astype(int)
        )
    else:
        pivot["ieee_total"] = 0
        pivot["elsevier_total"] = 0

    pivot = pivot.sort_values("total", ascending=False)
    pivot.index.name = "author"
    year_cols = sorted((c for c in pivot.columns if c not in _MATRIX_SUMMARY_COLS), key=int)
    return pivot.reset_index()[["author", *_MATRIX_SUMMARY_COLS, *year_cols]]


def _exploded_author_years(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Shared prep for author-by-year aggregations: explode, canonicalize, valid years only.

    `max_position`, when given, restricts to authors at or before that 0-based
    byline position (e.g. `max_position=1` keeps only the 1st/2nd author of
    each article) -- see `explode_authors_with_position`.
    """
    exploded = (
        explode_authors_with_position(df) if max_position is not None else explode_authors(df)
    )
    if max_position is not None and not exploded.empty:
        exploded = exploded[exploded["position"] <= max_position]
    if exploded.empty:
        return exploded
    unique_authors = exploded["author"].unique()
    canonical_map = {a: canonical_author(a) for a in unique_authors}
    exploded["author_key"] = exploded["author"].map(canonical_map)
    exploded = exploded[exploded["author_key"] != ""]
    exploded["year"] = valid_years(exploded)
    return exploded.dropna(subset=["year"]).astype({"year": int})


def researchers_by_year(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Distinct canonical-author count per year, split ieee/elsevier/total.

    Unlike `source_counts_by` (which counts rows), `total` here is counted
    independently as the number of distinct authors active that year
    regardless of source -- an author publishing in both IEEE and Elsevier
    the same year must count once in `total`, not twice. `ieee`/`elsevier`
    default to 0 when a `source` column isn't present. `max_position` restricts
    to authors at or before that byline position (see `_exploded_author_years`).
    """
    exploded = _exploded_author_years(df, max_position=max_position)
    if exploded.empty:
        return pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])

    years = sorted(exploded["year"].unique())
    result = pd.DataFrame({"year": years})
    has_source = "source" in exploded.columns
    for src in ("ieee", "elsevier"):
        if has_source:
            counts = exploded[exploded["source"] == src].groupby("year")["author_key"].nunique()
            result[src] = result["year"].map(counts).fillna(0).astype(int)
        else:
            result[src] = 0
    total_counts = exploded.groupby("year")["author_key"].nunique()
    result["total"] = result["year"].map(total_counts).fillna(0).astype(int)
    return result


def cumulative_researchers(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Cumulative count of distinct researchers introduced by each year.

    Each canonical author is counted once, in the year of their earliest
    valid-year appearance (within that source, for `ieee`/`elsevier`; across
    all sources, for `total`) -- summing each year's *active* researcher
    count would double-count an author active across multiple years, which
    isn't what a cumulative researcher count should mean. As with every other
    ieee/elsevier/total triple here, `total` isn't required to equal
    `ieee + elsevier` -- an author's first IEEE year and first Elsevier year
    can differ from their first-ever appearance. `max_position` restricts to
    authors at or before that byline position (see `_exploded_author_years`).
    """
    exploded = _exploded_author_years(df, max_position=max_position)
    if exploded.empty:
        return pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])

    years = sorted(exploded["year"].unique())

    def cumulative_new(sub: pd.DataFrame) -> pd.Series:
        if sub.empty:
            return pd.Series(0, index=years, dtype="int64")
        first_year = sub.groupby("author_key")["year"].min()
        by_year = first_year.value_counts().reindex(years, fill_value=0)
        return by_year.cumsum()

    result = pd.DataFrame({"year": years})
    has_source = "source" in exploded.columns
    for src in ("ieee", "elsevier"):
        sub = exploded[exploded["source"] == src] if has_source else exploded.iloc[0:0]
        result[src] = cumulative_new(sub).to_numpy()
    result["total"] = cumulative_new(exploded).to_numpy()
    return result


def gini_coefficient(values: pd.Series) -> float:
    """Gini coefficient of a distribution of non-negative values (0..1).

    0 = every author has the same output; close to 1 = output is concentrated
    in very few authors. Standard mean-absolute-difference formulation, no
    external stats dependency needed. Returns 0.0 for fewer than 2 authors or
    an all-zero series (nothing to be unequal about).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    arr = arr[arr >= 0]
    n = arr.size
    if n < 2 or arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    index = np.arange(1, n + 1, dtype="float64")
    return float((2 * (index * sorted_arr).sum() / (n * sorted_arr.sum())) - (n + 1) / n)


def lorenz_curve(values: pd.Series) -> pd.DataFrame:
    """Cumulative share of output vs. cumulative share of authors, sorted ascending.

    Includes the (0, 0) origin point. `charts.lorenz_chart` plots this against
    the perfect-equality diagonal as a second real trace (never a reference
    line, per the chart contract).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    arr = np.sort(arr[arr >= 0])
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return pd.DataFrame({"share_of_authors": [0.0], "share_of_output": [0.0]})

    cum_output = arr.cumsum() / arr.sum()
    cum_authors = np.arange(1, n + 1, dtype="float64") / n
    return pd.DataFrame(
        {
            "share_of_authors": [0.0, *cum_authors.tolist()],
            "share_of_output": [0.0, *cum_output.tolist()],
        }
    )


def author_productivity_trend(matrix: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Linear-trend classification of yearly output for the top `top_n` authors.

    Fits `count ~ year` with `numpy.polyfit` (degree 1) per author, over that
    author's own active years only (a career that ended in 2015 shouldn't be
    scored against years it has no data for). Classifies the slope as
    "crescendo" / "estável" / "caindo" against a small fixed threshold (0.15
    articles/year) rather than a significance test -- most authors here have
    under 15 active years, too short a series for a p-value to be meaningful
    (the same reasoning `forecasting.py` uses to prefer plain regression over
    a heavier model). Sorted descending by `total`, matching the main table.
    """
    columns = [
        "author",
        "total",
        "first_year",
        "last_year",
        "active_years",
        "slope",
        "trend",
    ]
    if matrix.empty:
        return pd.DataFrame(columns=columns)

    year_cols = [c for c in matrix.columns if c not in ("author", *_MATRIX_SUMMARY_COLS)]
    top = matrix.sort_values("total", ascending=False).head(top_n)

    rows = []
    for _, row in top.iterrows():
        active = [(int(y), row[y]) for y in year_cols if row[y] > 0]
        if len(active) < 2:
            first_year = active[0][0] if active else None
            rows.append(
                {
                    "author": row["author"],
                    "total": int(row["total"]),
                    "first_year": first_year,
                    "last_year": first_year,
                    "active_years": len(active),
                    "slope": 0.0,
                    "trend": "dados insuficientes",
                }
            )
            continue
        # Build the FULL year range (first_year to last_year) filling gaps
        # with zeros so that periods of inactivity count against the slope
        # instead of being silently omitted from the regression.
        active_map = dict(active)
        first_y, last_y = min(active_map), max(active_map)
        full_years = list(range(first_y, last_y + 1))
        full_counts = [active_map.get(y, 0) for y in full_years]
        slope = float(_linear_slope(full_years, full_counts))
        if slope > 0.15:
            trend = "crescendo"
        elif slope < -0.15:
            trend = "caindo"
        else:
            trend = "estável"
        rows.append(
            {
                "author": row["author"],
                "total": int(row["total"]),
                "first_year": first_y,
                "last_year": last_y,
                "active_years": len(active),
                "slope": round(slope, 3),
                "trend": trend,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("total", ascending=False)


def _linear_slope(x: list[int], y: list[int]) -> float:
    """`numpy.polyfit` degree-1 slope, isolated for testability."""
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def output_impact_correlation(
    author_rows: pd.DataFrame, impact_col: str = "citation_count"
) -> dict[str, float | int | None]:
    """Pearson and Spearman correlation between an author's total output and
    their mean `impact_col`, plus the sample size actually used.

    Rows with a null `impact_col` are dropped before averaging -- per this
    corpus's documented data pitfall, null means "not collected", not zero
    impact, and must not be averaged in as 0. Returns `None`s when fewer than
    3 authors have usable data (a correlation over 1-2 points is noise).
    """
    result: dict[str, float | int | None] = {"pearson": None, "spearman": None, "n": 0}
    if impact_col not in author_rows.columns or "author_display" not in author_rows.columns:
        return result

    by_author = author_rows.groupby("author_display").agg(
        articles=("author_display", "size"), mean_impact=(impact_col, "mean")
    )
    by_author = by_author.dropna(subset=["mean_impact"])
    result["n"] = int(len(by_author))
    if len(by_author) < 3:
        return result

    result["pearson"] = float(
        by_author["articles"].corr(by_author["mean_impact"], method="pearson")
    )
    result["spearman"] = float(
        by_author["articles"].corr(by_author["mean_impact"], method="spearman")
    )
    return result


LAYER_ORDER = ("raw", "bronze", "silver", "gold")
LAYER_LABELS = {"raw": "Raw", "bronze": "Bronze", "silver": "Silver", "gold": "Gold"}


def layer_source_counts(
    bronze_df: pd.DataFrame, silver_df: pd.DataFrame, gold_df: pd.DataFrame
) -> pd.DataFrame:
    """IEEE / Elsevier / total article counts for bronze, silver, and gold.

    Bronze has a scalar `source`; silver/gold only carry a `sources` list --
    normalize both to the same ieee/elsevier/total shape so the funnel chart
    can compare layers directly.
    """
    rows = []
    for layer, frame, col in (
        ("bronze", bronze_df, "source"),
        ("silver", silver_df, "sources"),
        ("gold", gold_df, "sources"),
    ):
        if frame.empty:
            rows.append({"layer": layer, "ieee": 0, "elsevier": 0, "total": 0})
            continue
        if col == "source":
            ieee = int((frame["source"] == "ieee").sum())
            elsevier = int((frame["source"] == "elsevier").sum())
        else:
            src_lists = frame[col] if col in frame.columns else pd.Series([[]] * len(frame))
            ieee = int(src_lists.apply(lambda s: isinstance(s, list) and "ieee" in s).sum())
            elsevier = int(src_lists.apply(lambda s: isinstance(s, list) and "elsevier" in s).sum())
        rows.append({"layer": layer, "ieee": ieee, "elsevier": elsevier, "total": len(frame)})
    return pd.DataFrame(rows)


# --- Advanced Statistics, Bibliometrics & Complex Networks -----------------


def fit_heavy_tail_distributions(citation_counts: np.ndarray | pd.Series) -> dict:
    """Fit and compare Power-law, Log-normal, and Exponential distributions to citation counts.

    Returns estimated parameters, Kolmogorov-Smirnov statistics (D), and p-values.
    """
    from scipy import stats

    arr = np.asarray(citation_counts, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = len(arr)
    if n < 10:
        return {"n": n, "valid": False, "best_fit": "insufficient_data"}

    x_min = float(np.min(arr))
    sum_log = float(np.sum(np.log(arr / x_min)))
    alpha = 1.0 + n / sum_log if sum_log > 0 else 1.0
    ks_pareto = (
        stats.kstest(arr, stats.pareto(b=alpha - 1.0, scale=x_min).cdf) if alpha > 1 else (1.0, 0.0)
    )

    log_arr = np.log(arr)
    mu = float(np.mean(log_arr))
    sigma = float(np.std(log_arr, ddof=1)) if n > 1 else float(np.std(log_arr))
    ks_lognorm = (
        stats.kstest(arr, stats.lognorm(s=sigma, scale=np.exp(mu)).cdf) if sigma > 0 else (1.0, 0.0)
    )

    mean_val = float(np.mean(arr))
    ks_expon = stats.kstest(arr, stats.expon(scale=mean_val).cdf)

    fits = {
        "power_law": {
            "alpha": float(alpha),
            "x_min": float(x_min),
            "ks_stat": float(ks_pareto[0]),
            "p_value": float(ks_pareto[1]),
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
    best = min(fits.keys(), key=lambda k: fits[k]["ks_stat"])
    return {"n": n, "valid": True, "best_fit": best, "models": fits}


def age_normalized_citations(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """Calculate annual citation rates, cohort z-scores, and cohort percentiles.

    Prevents older papers from structurally dominating impact rankings.
    """
    if df.empty or "citation_count" not in df.columns or "year" not in df.columns:
        return df.copy()

    res = df.copy()
    years = pd.to_numeric(res["year"], errors="coerce")
    cites = pd.to_numeric(res["citation_count"], errors="coerce").fillna(0)

    age = (current_year - years + 1).clip(lower=1)
    res["citation_rate_annual"] = cites / age

    cohort_mean = res.groupby(years)["citation_count"].transform("mean")
    cohort_std = res.groupby(years)["citation_count"].transform("std").replace(0, 1.0).fillna(1.0)
    res["cohort_citation_zscore"] = (cites - cohort_mean) / cohort_std
    res["cohort_citation_percentile"] = res.groupby(years)["citation_count"].rank(pct=True)
    return res


def mann_kendall_trend(series: np.ndarray | pd.Series) -> dict[str, float | str]:
    """Compute Mann-Kendall non-parametric monotonic trend test and Sen's slope.

    Returns trend direction ('crescendo', 'estável', 'caindo'), p-value, S statistic, and slope.
    """
    from scipy import stats

    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4:
        return {
            "trend": "estável",
            "p_value": 1.0,
            "s": 0.0,
            "slope": 0.0,
            "z": 0.0,
        }

    i_indices, j_indices = np.triu_indices(n, k=1)
    diffs = y[j_indices] - y[i_indices]
    s = float(np.sum(np.sign(diffs)))
    slopes = diffs / (j_indices - i_indices)

    sen_slope = float(np.median(slopes)) if len(slopes) > 0 else 0.0

    unique, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s > 0:
        if s > 0:
            z = (s - 1.0) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1.0) / np.sqrt(var_s)
        else:
            z = 0.0
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    else:
        z = 0.0
        p_value = 1.0

    if p_value < 0.05 and sen_slope > 0:
        trend = "crescendo"
    elif p_value < 0.05 and sen_slope < 0:
        trend = "caindo"
    else:
        trend = "estável"

    return {
        "trend": trend,
        "p_value": p_value,
        "s": float(s),
        "z": float(z),
        "slope": sen_slope,
    }


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


def coauthorship_community_detection(graph) -> dict[str, int]:
    """Detect research communities in a coauthorship network using Louvain modularity."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}
    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        node_comm = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = comm_id
        return node_comm
    except Exception:
        return {
            node: idx for idx, comp in enumerate(nx.connected_components(graph)) for node in comp
        }


def graph_advanced_metrics(graph) -> dict:
    """Calculate betweenness, closeness, pagerank, clustering, density, and small-world metrics."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {
            "density": 0.0,
            "avg_clustering": 0.0,
            "betweenness": {},
            "closeness": {},
            "pagerank": {},
            "degree": {},
            "avg_path_length": None,
            "small_world_sigma": None,
        }
    # In coauthorship networks, edge weight = collaboration intensity (higher
    # means closer).  Betweenness and closeness treat weight as *distance*, so
    # we invert: distance = 1 / intensity.  PageRank treats weight as strength
    # (no inversion needed).
    dist_graph = graph.copy()
    for _u, _v, d in dist_graph.edges(data=True):
        w = d.get("weight", 1)
        d["distance"] = 1.0 / w if w > 0 else 1.0
    betweenness = nx.betweenness_centrality(dist_graph, weight="distance")
    closeness = nx.closeness_centrality(dist_graph, distance="distance")
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except Exception:
        pagerank = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}

    degree = dict(graph.degree(weight="weight"))
    density = float(nx.density(graph))
    avg_clustering = float(nx.average_clustering(graph, weight="weight"))

    # Small-world analysis on the largest connected component (LCC)
    avg_path_length = None
    small_world_sigma = None
    if graph.number_of_nodes() >= 4 and graph.number_of_edges() >= 3:
        components = list(nx.connected_components(graph))
        if components:
            largest_cc = max(components, key=len)
            sub_lcc = graph.subgraph(largest_cc)
            if sub_lcc.number_of_nodes() >= 3:
                try:
                    avg_path_length = float(nx.average_shortest_path_length(sub_lcc))
                    n_lcc = sub_lcc.number_of_nodes()
                    m_lcc = sub_lcc.number_of_edges()
                    k_mean = (2.0 * m_lcc) / n_lcc
                    if k_mean > 1.0 and n_lcc > k_mean:
                        c_rand = k_mean / n_lcc
                        l_rand = np.log(n_lcc) / np.log(k_mean)
                        c_actual = float(nx.average_clustering(sub_lcc))
                        if c_rand > 0 and l_rand > 0 and avg_path_length > 0:
                            gamma = c_actual / c_rand
                            lambda_val = avg_path_length / l_rand
                            if lambda_val > 0:
                                small_world_sigma = float(gamma / lambda_val)
                except Exception:
                    pass

    return {
        "density": density,
        "avg_clustering": avg_clustering,
        "betweenness": betweenness,
        "closeness": closeness,
        "pagerank": pagerank,
        "degree": degree,
        "avg_path_length": avg_path_length,
        "small_world_sigma": small_world_sigma,
    }


def analyze_coauthorship_partners(
    graph,
    author_rows: pd.DataFrame,
    communities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Analyze unique vs. recurrent coauthorships for authors in the network and corpus.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph of top authors (nodes are author display names, edges have 'weight').
    author_rows : pd.DataFrame
        DataFrame with at least 'author_display' (or 'author') and 'doi' columns.
    communities : dict[str, int] | None
        Optional mapping of author -> community id. If None, Louvain communities are computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with detailed breakdown of unique and recurrent collaborators
        both within the top-authors network subgraph and across the global corpus.
    """
    import collections

    columns = [
        "author",
        "articles",
        "community",
        "community_id",
        "network_unique_count",
        "network_recurrent_count",
        "network_recurrent_names",
        "network_occasional_count",
        "network_occasional_names",
        "global_unique_count",
        "global_recurrent_count",
        "global_recurrent_names",
        "global_top_partners",
    ]

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    if communities is None:
        communities = coauthorship_community_detection(graph)

    author_col = "author_display" if "author_display" in author_rows.columns else "author"
    has_author_col = author_col in author_rows.columns
    has_doi = "doi" in author_rows.columns

    # Pre-aggregate global collaborations across the corpus
    doi_to_authors: dict[str, set[str]] = collections.defaultdict(set)
    author_to_dois: dict[str, set[str]] = collections.defaultdict(set)
    if has_author_col and has_doi:
        valid = author_rows[[author_col, "doi"]].dropna()
        for a, d in zip(valid[author_col].astype(str), valid["doi"].astype(str), strict=True):
            doi_to_authors[d].add(a)
            author_to_dois[a].add(d)

    rows = []
    for u in graph.nodes():
        # --- 1. In-network metrics ---
        neighbors = list(graph.neighbors(u))
        network_unique_count = len(neighbors)

        # Recurrent in network (weight >= 2)
        recurrent_net = [
            (v, int(graph[u][v].get("weight", 1)))
            for v in neighbors
            if graph[u][v].get("weight", 1) >= 2
        ]
        recurrent_net.sort(key=lambda x: (-x[1], x[0]))
        network_recurrent_count = len(recurrent_net)
        network_recurrent_names = (
            ", ".join(f"{v} ({w})" for v, w in recurrent_net) if recurrent_net else "—"
        )

        # Occasional in network (weight == 1)
        occasional_net = sorted([v for v in neighbors if graph[u][v].get("weight", 1) == 1])
        network_occasional_count = len(occasional_net)
        network_occasional_names = ", ".join(occasional_net) if occasional_net else "—"

        # Community info (1-indexed)
        comm_idx = communities.get(u, 0) + 1
        comm_label = f"#{comm_idx}"

        # --- 2. Global corpus metrics ---
        u_dois = author_to_dois.get(u, set())
        articles_count = len(u_dois)

        partner_counts: dict[str, int] = collections.defaultdict(int)
        for d in u_dois:
            for coauthor in doi_to_authors.get(d, set()):
                if coauthor != u:
                    partner_counts[coauthor] += 1

        global_unique_count = len(partner_counts)
        recurrent_glob = {k: v for k, v in partner_counts.items() if v >= 2}
        global_recurrent_count = len(recurrent_glob)

        sorted_recurrent_glob = sorted(recurrent_glob.items(), key=lambda x: (-x[1], x[0]))
        global_recurrent_names = (
            ", ".join(f"{k} ({v})" for k, v in sorted_recurrent_glob[:5])
            if sorted_recurrent_glob
            else "—"
        )

        sorted_all_glob = sorted(partner_counts.items(), key=lambda x: (-x[1], x[0]))
        global_top_partners = (
            ", ".join(f"{k} ({v})" for k, v in sorted_all_glob[:5]) if sorted_all_glob else "—"
        )

        rows.append(
            {
                "author": u,
                "articles": articles_count,
                "community": comm_label,
                "community_id": comm_idx,
                "network_unique_count": network_unique_count,
                "network_recurrent_count": network_recurrent_count,
                "network_recurrent_names": network_recurrent_names,
                "network_occasional_count": network_occasional_count,
                "network_occasional_names": network_occasional_names,
                "global_unique_count": global_unique_count,
                "global_recurrent_count": global_recurrent_count,
                "global_recurrent_names": global_recurrent_names,
                "global_top_partners": global_top_partners,
            }
        )

    res_df = pd.DataFrame(rows, columns=columns)
    if not res_df.empty:
        res_df = res_df.sort_values(
            by=["articles", "network_unique_count", "network_recurrent_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return res_df


def citation_determinants_glm(df: pd.DataFrame, *, observation_year: int = 2026) -> dict:
    """Fit an exposure-adjusted count GLM with robust uncertainty estimates.

    Citation counts accumulate over time, so ``log(article_age + 1)`` is used
    as an exposure offset. Numeric predictors are standardized and rows with
    missing predictors are excluded instead of silently filled. A negative
    binomial variance is used when the preliminary Poisson fit is materially
    overdispersed.
    """
    from statsmodels.genmod.families import NegativeBinomial, Poisson
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools.tools import add_constant

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
            "warning": "Campos necessários ausentes.",
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
            "warning": "Menos de 20 observações completas.",
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
            "warning": "Os preditores não variam neste recorte.",
        }
    design = add_constant(design, has_constant="add")
    response = valid["citations"].astype(float)
    exposure = np.log((observation_year - valid["publication_year"] + 1).clip(lower=1))

    try:
        poisson = GLM(
            response,
            design,
            family=Poisson(),
            offset=exposure,
        ).fit(cov_type="HC3")
        dispersion = float(poisson.pearson_chi2 / max(poisson.df_resid, 1))
        if dispersion > 1.5:
            fitted = GLM(
                response,
                design,
                family=NegativeBinomial(alpha=max(dispersion - 1.0, 0.01)),
                offset=exposure,
            ).fit(cov_type="HC3")
            family = "binomial_negativa"
        else:
            fitted = poisson
            family = "poisson"
        coefficients = fitted.params.reindex(active_features)
        intervals = fitted.conf_int().reindex(active_features)
        return {
            "valid": True,
            "features": active_features,
            "coefficients": coefficients.astype(float).tolist(),
            "irr": np.exp(coefficients).astype(float).tolist(),
            "irr_lower": np.exp(intervals[0]).astype(float).tolist(),
            "irr_upper": np.exp(intervals[1]).astype(float).tolist(),
            "p_values": fitted.pvalues.reindex(active_features).astype(float).tolist(),
            "family": family,
            "dispersion": dispersion,
            "score": float(1 - fitted.deviance / fitted.null_deviance)
            if fitted.null_deviance > 0
            else float("nan"),
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": (
                "Preditores sem variação foram omitidos: " + ", ".join(omitted) if omitted else None
            ),
        }
    except (ValueError, np.linalg.LinAlgError):
        return {
            "valid": False,
            "features": feature_names,
            "coefficients": [],
            "irr": [],
            "n_total": n_total,
            "n_used": n_used,
            "coverage": coverage,
            "warning": "O modelo não convergiu para este recorte.",
        }


def zipf_law_analysis(df: pd.DataFrame) -> dict:
    """Evaluate Zipf's law for technical vocabulary in titles and abstracts.

    Calculates word frequency vs. rank, fits log(f) = log(C) - gamma * log(r)
    via OLS regression, and returns goodness of fit (R^2), slope gamma (~1.0),
    and a top words table.
    """
    from collections import Counter

    # Aggregate text from title and abstract
    texts = []
    if "title" in df.columns:
        texts.append(df["title"].dropna().astype(str))
    if "abstract" in df.columns:
        texts.append(df["abstract"].dropna().astype(str))

    if not texts:
        return {
            "valid": False,
            "gamma": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "vocab_size": 0,
            "total_tokens": 0,
            "ranks": [],
            "frequencies": [],
            "words": [],
            "top_words_df": pd.DataFrame(),
        }

    full_series = pd.concat(texts, ignore_index=True)
    all_text = " ".join(full_series)
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
    total_tokens = len(tokens)

    stopwords = {
        "the",
        "of",
        "and",
        "in",
        "to",
        "for",
        "with",
        "on",
        "as",
        "by",
        "at",
        "an",
        "be",
        "is",
        "are",
        "from",
        "that",
        "this",
        "these",
        "it",
        "its",
        "or",
        "can",
        "has",
        "have",
        "been",
        "was",
        "were",
        "into",
        "also",
        "such",
        "than",
        "through",
        "which",
        "our",
        "their",
        "paper",
        "presents",
        "proposed",
        "method",
        "based",
        "using",
        "results",
        "study",
        "analysis",
        "model",
        "system",
        "systems",
        "approach",
        "case",
        "different",
        "new",
    }
    filtered_tokens = [w for w in tokens if w not in stopwords]
    counter = Counter(filtered_tokens)

    if len(counter) < 10:
        return {
            "valid": False,
            "gamma": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "vocab_size": len(counter),
            "total_tokens": total_tokens,
            "ranks": [],
            "frequencies": [],
            "words": [],
            "top_words_df": pd.DataFrame(),
        }

    top_items = counter.most_common(500)
    words = [w for w, _ in top_items]
    freqs = np.array([c for _, c in top_items], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    log_r = np.log10(ranks)
    log_f = np.log10(freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    pred = slope * log_r + intercept
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    ss_res = np.sum((log_f - pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    gamma = -float(slope)

    expected_ideal = freqs[0] / ranks

    top_df = pd.DataFrame(
        {
            "Posto (r)": ranks[:30].astype(int),
            "Termo": words[:30],
            "Frequência Real (f)": freqs[:30].astype(int),
            "Previsto Zipf Ideal": expected_ideal[:30].round(1),
            "Ajuste Empírico": (10 ** pred[:30]).round(1),
        }
    )

    return {
        "valid": True,
        "gamma": gamma,
        "intercept": float(intercept),
        "r_squared": float(r2),
        "vocab_size": len(counter),
        "total_tokens": total_tokens,
        "ranks": ranks.tolist(),
        "frequencies": freqs.tolist(),
        "words": words,
        "expected_zipf": (10**pred).tolist(),
        "top_words_df": top_df,
    }


def dynamic_topic_ctfidf(
    df: pd.DataFrame, time_windows: list[tuple[int, int]] | None = None
) -> dict:
    """Compute Class-based TF-IDF (c-TF-IDF) by research theme across chronological epochs.

    Reveals which technical terms distinguish each theme during each epoch.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if time_windows is None:
        time_windows = [(1990, 2010), (2011, 2018), (2019, 2026)]

    req_cols = ["theme_label", "year"]
    if not all(col in df.columns for col in req_cols):
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    text_series = df["title"].fillna("")
    if "abstract" in df.columns:
        text_series = text_series + " " + df["abstract"].fillna("")

    valid_df = df.copy()
    valid_df["_full_text"] = text_series
    valid_df["year"] = pd.to_numeric(valid_df["year"], errors="coerce")
    valid_df = valid_df.dropna(subset=["year", "theme_label"])
    valid_df["year"] = valid_df["year"].astype(int)

    epochs_data = []
    summary_rows = []

    global_docs = []
    for start, end in time_windows:
        sub = valid_df[(valid_df["year"] >= start) & (valid_df["year"] <= end)]
        if len(sub) >= 10:
            grouped = sub.groupby("theme_label")["_full_text"].apply(lambda s: " ".join(s))
            if len(grouped) >= 2:
                global_docs.extend(grouped.values)

    if not global_docs:
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    try:
        vec = CountVectorizer(stop_words="english", min_df=1, max_features=1500)
        vec.fit(global_docs)
        vocab = np.array(vec.get_feature_names_out())
        if len(vocab) < 3:
            return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}
    except Exception:
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    for start, end in time_windows:
        epoch_name = f"{start}–{end}"
        sub = valid_df[(valid_df["year"] >= start) & (valid_df["year"] <= end)]
        if len(sub) < 10:
            continue

        grouped = sub.groupby("theme_label")["_full_text"].apply(lambda s: " ".join(s))
        themes = list(grouped.index)
        docs = list(grouped.values)

        if len(themes) < 2:
            continue

        try:
            X = vec.transform(docs).toarray()

            tf = X
            ft = X.sum(axis=0)
            A = X.sum(axis=1).mean()
            idf = np.log(1 + A / (ft + 1e-9))
            ctfidf = tf * idf

            theme_res = {}
            for i, th in enumerate(themes):
                top_indices = np.argsort(-ctfidf[i])[:5]
                top_kws = [vocab[idx] for idx in top_indices if ctfidf[i, idx] > 0]
                theme_res[th] = top_kws
                summary_rows.append(
                    {
                        "Época": epoch_name,
                        "Tema": th,
                        "Termos Característicos (c-TF-IDF)": ", ".join(top_kws),
                        "Artigos": int((sub["theme_label"] == th).sum()),
                    }
                )

            epochs_data.append(
                {
                    "name": epoch_name,
                    "start": start,
                    "end": end,
                    "themes": theme_res,
                }
            )
        except Exception:
            continue

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return {"valid": len(epochs_data) > 0, "epochs": epochs_data, "summary_df": summary_df}


def detect_bibliometric_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """Identify multidimensional bibliometric outliers using Isolation Forest.

    Audits publication records based on citation velocity, author team size,
    relevance score, and full-text metadata.
    """
    from sklearn.ensemble import IsolationForest

    if len(df) < 20:
        res = df.copy()
        res["anomaly_score"] = 0.0
        res["is_anomaly"] = False
        res["anomaly_reason"] = "—"
        return res

    res = df.copy()
    years = pd.to_numeric(res.get("year"), errors="coerce").fillna(2015).to_numpy()
    cites = (
        pd.to_numeric(res.get("citation_count"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
        .to_numpy()
    )

    if "reference_count" in res.columns:
        refs = pd.to_numeric(res["reference_count"], errors="coerce").fillna(0.0).to_numpy()
    else:
        refs = np.zeros(len(res))

    if "authors" in res.columns:
        author_counts = (
            res["authors"].apply(lambda a: len(a) if isinstance(a, list) else 1).to_numpy()
        )
    else:
        author_counts = np.ones(len(res))

    if "relevance_score" in res.columns:
        rel = pd.to_numeric(res["relevance_score"], errors="coerce").fillna(0.5).to_numpy()
    else:
        rel = np.full(len(res), 0.5)

    if "has_pdf" in res.columns:
        has_pdf = res["has_pdf"].fillna(False).astype(int).to_numpy()
    else:
        has_pdf = np.zeros(len(res))

    X = np.column_stack([years, cites, refs, author_counts, rel, has_pdf])

    try:
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit(
            X
        )
        preds = model.predict(X)
        scores = -model.decision_function(X)

        res["anomaly_score"] = scores.round(3)
        res["is_anomaly"] = preds == -1

        # Derive explanatory rationales
        reasons = []
        cite_p95 = np.percentile(cites, 95)
        author_p98 = np.percentile(author_counts, 98)

        for i in range(len(res)):
            if preds[i] != -1:
                reasons.append("Padrão Típico / Consistente")
                continue
            r_list = []
            if cites[i] >= cite_p95 and years[i] >= 2018:
                r_list.append(f"Hiper-citado recente ({int(cites[i])} citações)")
            elif cites[i] >= cite_p95:
                r_list.append(f"Citação extrema ({int(cites[i])} citações)")
            if author_counts[i] >= author_p98 and author_counts[i] >= 10:
                r_list.append(f"Mega-equipe ({int(author_counts[i])} autores)")
            if years[i] < 1990:
                r_list.append(f"Artigo histórico ({int(years[i])})")
            if rel[i] < 0.25:
                r_list.append(f"Margem semântica divergente ({rel[i]:.2f})")
            if not r_list:
                r_list.append("Atipicidade multidimensional conjunta")
            reasons.append("; ".join(r_list))

        res["anomaly_reason"] = reasons
    except Exception:
        res["anomaly_score"] = 0.0
        res["is_anomaly"] = False
        res["anomaly_reason"] = "Erro no ajuste"

    return res


def callon_strategic_diagram(
    df: pd.DataFrame, embeddings: np.ndarray | None = None, dois: list[str] | None = None
) -> dict:
    """Compute Callon's Strategic Diagram (1991) metrics: Density vs. Centrality.

    - Callon's Density (Y-axis): Average internal cohesion of a theme (mean pairwise cosine similarity of articles within theme).
    - Callon's Centrality (X-axis): Average external interaction / structural importance (mean cosine similarity of theme centroid to other theme centroids).
    Divides themes into four strategic quadrants:
      Q1 (Motor Themes): High Centrality, High Density
      Q2 (Specialized/Niche Themes): Low Centrality, High Density
      Q3 (Emerging or Marginal Themes): Low Centrality, Low Density
      Q4 (Basic/Transversal Themes): High Centrality, Low Density
    """
    return {
        "available": False,
        "reason": "A Callon diagram requires a keyword equivalence network.",
        "themes_df": pd.DataFrame(),
        "median_density": 0.0,
        "median_centrality": 0.0,
    }
    if df.empty or "theme_label" not in df.columns or embeddings is None or dois is None:
        return {"themes_df": pd.DataFrame(), "median_density": 0.0, "median_centrality": 0.0}

    doi_to_idx = {d: i for i, d in enumerate(dois)}
    valid = df.copy()
    valid["emb_idx"] = valid["doi"].map(doi_to_idx)
    valid = valid.dropna(subset=["emb_idx", "theme_label"])
    if valid.empty:
        return {"themes_df": pd.DataFrame(), "median_density": 0.0, "median_centrality": 0.0}

    valid["emb_idx"] = valid["emb_idx"].astype(int)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    theme_stats = []
    centroids = {}
    theme_groups = {}

    for theme, grp in valid.groupby("theme_label"):
        idxs = grp["emb_idx"].to_numpy()
        theme_groups[theme] = idxs
        sub = normed[idxs]
        cent = sub.mean(axis=0)
        c_norm = np.linalg.norm(cent)
        centroids[theme] = cent / (c_norm if c_norm > 0 else 1.0)

    for theme, idxs in theme_groups.items():
        sub = normed[idxs]
        n_docs = len(sub)
        # Internal density
        if n_docs > 1:
            sim_mat = sub @ sub.T
            np.fill_diagonal(sim_mat, 0)
            density = float(sim_mat.sum() / (n_docs * (n_docs - 1)))
        else:
            density = 1.0

        # External centrality: cosine with other theme centroids
        t_cent = centroids[theme]
        other_sims = [float(np.dot(t_cent, centroids[o])) for o in centroids if o != theme]
        centrality = float(np.mean(other_sims)) if other_sims else 0.0

        # Citations & margin
        grp = valid[valid["theme_label"] == theme]
        mean_cites = float(
            pd.to_numeric(grp.get("citation_count"), errors="coerce").fillna(0).mean()
        )
        mean_margin = float(
            pd.to_numeric(grp.get("relevance_margin"), errors="coerce").fillna(0).mean()
        )

        theme_stats.append(
            {
                "theme": theme,
                "density": density,
                "centrality": centrality,
                "n_articles": n_docs,
                "mean_citations": round(mean_cites, 1),
                "mean_margin": round(mean_margin, 3),
            }
        )

    res_df = pd.DataFrame(theme_stats)
    if res_df.empty:
        return {"themes_df": pd.DataFrame(), "median_density": 0.0, "median_centrality": 0.0}

    med_density = float(res_df["density"].median())
    med_centrality = float(res_df["centrality"].median())

    res_df["centrality_centered"] = res_df["centrality"] - med_centrality
    res_df["density_centered"] = res_df["density"] - med_density

    # Assign Callon Quadrant
    def _quadrant(r):
        c, d = r["centrality_centered"], r["density_centered"]
        if c >= 0 and d >= 0:
            return "Q1: Temas Motores (Motor)"
        elif c < 0 and d >= 0:
            return "Q2: Temas Especializados / Nicho"
        elif c < 0 and d < 0:
            return "Q3: Temas Emergentes ou Marginais"
        else:
            return "Q4: Temas Básicos / Transversais"

    res_df["quadrant"] = res_df.apply(_quadrant, axis=1)

    return {
        "themes_df": res_df,
        "median_density": med_density,
        "median_centrality": med_centrality,
    }


def keyword_cooccurrence_graph(
    df: pd.DataFrame, min_cooccurrence: int = 3, top_n_keywords: int = 40
) -> dict:
    """Build a keyword co-occurrence network with Jaccard edge weights and Louvain clustering."""
    import itertools
    from collections import Counter

    import networkx as nx

    if df.empty or "keywords" not in df.columns:
        return {"nodes": [], "edges": [], "n_communities": 0}

    kw_counts = Counter()
    doc_kws = []
    # Stopwords/generic terms in keywords to filter
    generic_kws = {
        "planning",
        "distribution systems",
        "distribution system",
        "distribution network",
        "distribution networks",
        "electric power",
        "power distribution",
        "power systems",
        "paper",
        "method",
        "models",
    }

    for kws in df["keywords"]:
        if not isinstance(kws, list):
            continue
        cleaned = [
            k.strip().lower()
            for k in kws
            if len(k.strip()) > 2 and k.strip().lower() not in generic_kws
        ]
        unique_k = sorted(set(cleaned))
        if unique_k:
            doc_kws.append(unique_k)
            kw_counts.update(unique_k)

    top_keywords = {term for term, _ in kw_counts.most_common(top_n_keywords)}
    if not top_keywords:
        return {"nodes": [], "edges": [], "n_communities": 0}

    pair_counts = Counter()
    for k_list in doc_kws:
        filtered = [k for k in k_list if k in top_keywords]
        for k1, k2 in itertools.combinations(filtered, 2):
            pair_counts[(k1, k2)] += 1

    # Filter by min_cooccurrence
    edges = []
    for (k1, k2), count in pair_counts.items():
        if count >= min_cooccurrence:
            jaccard = count / (kw_counts[k1] + kw_counts[k2] - count)
            edges.append((k1, k2, count, jaccard))

    G = nx.Graph()
    for kw in top_keywords:
        G.add_node(kw, count=kw_counts[kw])
    for k1, k2, count, jaccard in edges:
        G.add_edge(k1, k2, weight=count, jaccard=jaccard)

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    if len(G.nodes) == 0:
        return {"nodes": [], "edges": [], "n_communities": 0}

    # Community detection via Louvain
    try:
        from networkx.algorithms.community import louvain_communities

        communities = list(louvain_communities(G, seed=42))
    except Exception:
        communities = [set(G.nodes())]

    comm_map = {}
    for c_id, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = c_id

    # Spring layout
    pos = nx.spring_layout(G, k=0.4, seed=42, iterations=50)

    nodes_data = []
    for node in G.nodes():
        x, y = pos[node]
        nodes_data.append(
            {
                "id": node,
                "x": float(x),
                "y": float(y),
                "count": kw_counts[node],
                "community": comm_map.get(node, 0),
            }
        )

    edges_data = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edges_data.append(
            {
                "source": u,
                "target": v,
                "weight": data["weight"],
                "jaccard": round(data["jaccard"], 3),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            }
        )

    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "n_communities": len(communities),
    }


def thematic_centroids_similarity(
    df: pd.DataFrame, embeddings: np.ndarray | None = None, dois: list[str] | None = None
) -> pd.DataFrame:
    """Compute pairwise cosine similarity matrix between thematic cluster centroids in R^384."""
    if df.empty or "theme_label" not in df.columns or embeddings is None or dois is None:
        return pd.DataFrame()

    doi_to_idx = {d: i for i, d in enumerate(dois)}
    valid = df.copy()
    valid["emb_idx"] = valid["doi"].map(doi_to_idx)
    valid = valid.dropna(subset=["emb_idx", "theme_label"])
    if valid.empty:
        return pd.DataFrame()

    valid["emb_idx"] = valid["emb_idx"].astype(int)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    centroids = {}
    for theme, grp in valid.groupby("theme_label"):
        idxs = grp["emb_idx"].to_numpy()
        cent = normed[idxs].mean(axis=0)
        c_norm = np.linalg.norm(cent)
        centroids[theme] = cent / (c_norm if c_norm > 0 else 1.0)

    themes = sorted(centroids.keys())
    C = np.vstack([centroids[t] for t in themes])
    matrix = C @ C.T

    return pd.DataFrame(matrix, index=themes, columns=themes)


def thematic_radar_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5 standardized dimensions (0-100) per theme for radar chart visualization:

    1. Momentum Recente (% docs >= 2021)
    2. Densidade Teórica (média de referências por artigo)
    3. Impacto Citatório (média de citações)
    4. Aderência Temática (margem de relevância média)
    5. Tamanho de Equipe (média de autores por artigo)
    """
    if df.empty or "theme_label" not in df.columns:
        return pd.DataFrame()

    res = df.copy()
    res["is_recent"] = (pd.to_numeric(res.get("year"), errors="coerce").fillna(0) >= 2021).astype(
        int
    )
    res["citations"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0)
    res["references"] = pd.to_numeric(res.get("reference_count"), errors="coerce").fillna(0)
    res["margin"] = pd.to_numeric(res.get("relevance_margin"), errors="coerce").fillna(0)
    if "authors" in res.columns:
        res["n_authors"] = res["authors"].apply(lambda a: len(a) if isinstance(a, list) else 1)
    else:
        res["n_authors"] = 1

    stats = (
        res.groupby("theme_label")
        .agg(
            n_articles=("id", "count"),
            pct_recent=("is_recent", "mean"),
            mean_citations=("citations", "mean"),
            mean_refs=("references", "mean"),
            mean_margin=("margin", "mean"),
            mean_authors=("n_authors", "mean"),
        )
        .reset_index()
    )

    # Scale each dimension to 0-100
    for col in ("pct_recent", "mean_citations", "mean_refs", "mean_margin", "mean_authors"):
        min_v = stats[col].min()
        max_v = stats[col].max()
        range_v = max_v - min_v if max_v > min_v else 1.0
        stats[f"{col}_score"] = ((stats[col] - min_v) / range_v * 100).round(1)

    return stats


def multivariate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix across 7 bibliometric and semantic variables."""
    if df.empty:
        return pd.DataFrame()

    numeric_cols = {
        "Ano": "year",
        "Citações": "citation_count",
        "Referências": "reference_count",
        "Score Relevância": "relevance_score",
        "Margem Relevância": "relevance_margin",
    }
    corr_df = pd.DataFrame()
    for label, col in numeric_cols.items():
        if col in df.columns:
            corr_df[label] = pd.to_numeric(df[col], errors="coerce")
        else:
            corr_df[label] = 0.0

    if "abstract" in df.columns:
        corr_df["Tam. Abstract (carac.)"] = df["abstract"].fillna("").astype(str).apply(len)
    else:
        corr_df["Tam. Abstract (carac.)"] = 0

    if "authors" in df.columns:
        corr_df["Nº Autores"] = df["authors"].apply(lambda a: len(a) if isinstance(a, list) else 1)
    else:
        corr_df["Nº Autores"] = 1

    corr = corr_df.corr(method="spearman").round(3)
    # Fill diagonal with 1.0 and off-diagonal NaNs with 0.0 for zero-variance attributes
    for col in corr.columns:
        if pd.isna(corr.loc[col, col]):
            corr.loc[col, col] = 1.0
    corr = corr.fillna(0.0)

    return corr


def shannon_thematic_entropy(df: pd.DataFrame, min_year: int = 2000) -> pd.DataFrame:
    """Compute Shannon Entropy H(t) and Gini-Simpson Index of thematic diversity over time."""
    if df.empty or "theme_label" not in df.columns or "year" not in df.columns:
        return pd.DataFrame()

    res = df.copy()
    res["year"] = pd.to_numeric(res["year"], errors="coerce")
    res = res.dropna(subset=["year", "theme_label"])
    res = res[res["year"] >= min_year]
    if res.empty:
        return pd.DataFrame()

    records = []
    for year, grp in sorted(res.groupby("year")):
        counts = grp["theme_label"].value_counts()
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        shannon = float(-np.sum(probs * np.log2(probs)))
        simpson = float(1.0 - np.sum(probs**2))
        records.append(
            {
                "year": int(year),
                "shannon_entropy": round(shannon, 3),
                "gini_simpson": round(simpson, 3),
                "n_articles": len(grp),
                "dominant_theme": counts.index[0],
                "dominant_share": round(float(counts.iloc[0] / total * 100), 1),
            }
        )

    return pd.DataFrame(records)


def geographic_collaboration_stats(df: pd.DataFrame) -> dict:
    """Compute country production counts, international collaboration rate, and bilateral country co-authorship pairs."""
    import itertools
    from collections import Counter

    if df.empty or "countries" not in df.columns:
        return {
            "country_counts": pd.DataFrame(),
            "collaboration_pairs": pd.DataFrame(),
            "total_with_country": 0,
            "intl_papers": 0,
            "intl_pct": 0.0,
        }

    has_countries = df["countries"].dropna()
    pair_counts = Counter()
    country_counts = Counter()
    intl_papers = 0
    total_with_country = 0

    for c_list in has_countries:
        if not isinstance(c_list, list) or not c_list:
            continue
        total_with_country += 1
        unique_c = sorted(set(str(c).strip() for c in c_list if str(c).strip()))
        country_counts.update(unique_c)
        if len(unique_c) > 1:
            intl_papers += 1
            for c1, c2 in itertools.combinations(unique_c, 2):
                pair_counts[(c1, c2)] += 1

    country_df = pd.DataFrame(
        [{"country": c, "articles": n} for c, n in country_counts.most_common(20)]
    )

    pair_df = pd.DataFrame(
        [
            {
                "country_a": p[0],
                "country_b": p[1],
                "pair": f"{p[0]} ↔ {p[1]}",
                "collaborations": n,
            }
            for p, n in pair_counts.most_common(20)
        ]
    )

    intl_pct = (intl_papers / total_with_country * 100) if total_with_country > 0 else 0.0

    return {
        "country_counts": country_df,
        "collaboration_pairs": pair_df,
        "total_with_country": total_with_country,
        "intl_papers": intl_papers,
        "intl_pct": round(intl_pct, 1),
    }


def optimization_methods_taxonomy(df: pd.DataFrame) -> dict:
    """Extract and quantify optimization methods used across the corpus.

    Identifies 8 major mathematical modeling and optimization paradigms:
    - MILP / Mixed-Integer Linear
    - Genetic Algorithms (GA)
    - Particle Swarm Optimization (PSO)
    - Robust Optimization
    - Stochastic Programming
    - Conic / Convex Relaxation (SOCP)
    - Machine Learning / AI / Reinforcement Learning
    - Multi-Objective Optimization
    - Diverse Metaheuristics
    """

    if df.empty:
        return {"summary_df": pd.DataFrame(), "temporal_df": pd.DataFrame()}

    opt_patterns = {
        "Otimização Multi-Objetivo": re.compile(r"multi-objective|pareto"),
        "Algoritmos Genéticos (GA)": re.compile(r"genetic algorithm|\bga\b"),
        "Otimização por Enxame (PSO)": re.compile(r"particle swarm|\bpso\b"),
        "Prog. Linear Inteira Mista (MILP)": re.compile(r"milp|mixed-integer linear"),
        "Aprendizado de Máquina & IA": re.compile(
            r"machine learning|deep learning|reinforcement learning|neural network"
        ),
        "Meta-heurísticas Diversas": re.compile(
            r"differential evolution|harmony search|simulated annealing|ant colony"
        ),
        "Prog. Estocástica": re.compile(r"stochastic programming|scenario-based"),
        "Otimização Robusta": re.compile(r"robust optimization|robust approach"),
        "Relaxação Cônica / Convexa (SOCP)": re.compile(
            r"second-order cone|conic|convex relaxation|socp"
        ),
    }

    if "_cached_titles_abs" not in df.columns:
        df["_cached_titles_abs"] = (
            df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
            + " "
            + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
        ).str.lower()
    titles_abs = df["_cached_titles_abs"]

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    summary_rows = []
    temporal_dict = {}

    for m_label, pat in opt_patterns.items():
        matched = titles_abs.str.contains(pat, regex=True)
        count = int(matched.sum())
        pct_recent = float((years[matched] >= 2021).mean() * 100) if count > 0 else 0.0
        mean_cites = float(cites[matched].mean()) if count > 0 else 0.0

        summary_rows.append(
            {
                "method": m_label,
                "articles": count,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_cites, 1),
            }
        )

        # Yearly counts for articles >= 2005
        yearly = df[matched & (years >= 2005)].groupby("year")["id"].count()
        temporal_dict[m_label] = yearly

    summary_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)
    temporal_df = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    temporal_df.index = temporal_df.index.astype(int)

    return {
        "summary_df": summary_df,
        "temporal_df": temporal_df,
    }


def benchmark_feeders_analysis(df: pd.DataFrame) -> dict:
    """Analyze IEEE standard benchmark test feeders and real-world distribution networks."""

    if df.empty:
        return {"feeders_df": pd.DataFrame(), "cross_matrix": pd.DataFrame()}

    feeder_patterns = {
        "IEEE 33-Bus (Radial Padrão)": re.compile(r"33-bus|ieee 33|33 bus|33-node"),
        "IEEE 69-Bus": re.compile(r"69-bus|ieee 69|69 bus|69-node"),
        "Redes Reais de Concessionárias": re.compile(
            r"real distribution|real-world|practical distribution|actual distribution|utility network"
        ),
        "Sistemas Regionais (Brasil / Europa)": re.compile(
            r"brazilian|european|california|uk distribution|nordic"
        ),
        "IEEE 123-Bus / 119-Bus": re.compile(r"123-bus|ieee 123|119-bus|ieee 119"),
    }

    resource_patterns = {
        "Geração Solar (PV)": re.compile(r"photovoltaic|\bpv\b|solar"),
        "Armazenamento / Baterias": re.compile(r"energy storage|battery|bess"),
        "Veículos Elétricos (EV)": re.compile(r"electric vehicle|\bev\b|v2g|charging"),
        "Reconfiguração de Redes": re.compile(r"reconfiguration|switching|switch"),
    }

    if "_cached_titles_abs" not in df.columns:
        df["_cached_titles_abs"] = (
            df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
            + " "
            + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
        ).str.lower()
    titles_abs = df["_cached_titles_abs"]

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    cross_data = {r_label: [] for r_label in resource_patterns}

    for f_label, pat in feeder_patterns.items():
        f_match = titles_abs.str.contains(pat, regex=True)
        count = int(f_match.sum())
        pct_recent = float((years[f_match] >= 2021).mean() * 100) if count > 0 else 0.0
        mean_cites = float(cites[f_match].mean()) if count > 0 else 0.0

        rows.append(
            {
                "feeder": f_label,
                "articles": count,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_cites, 1),
            }
        )

        for r_label, r_pat in resource_patterns.items():
            r_match = titles_abs.str.contains(r_pat, regex=True)
            cross_count = int((f_match & r_match).sum())
            cross_data[r_label].append(cross_count)

    feeders_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    cross_matrix = pd.DataFrame(cross_data, index=list(feeder_patterns.keys()))

    return {
        "feeders_df": feeders_df,
        "cross_matrix": cross_matrix,
    }


def author_impact_advanced_indices(df: pd.DataFrame, min_papers: int = 2) -> pd.DataFrame:
    """Compute h-index, Egghe's g-index, Zhang's e-index, and i10-index for researchers.

    - h-index: max h such that h papers have >= h citations.
    - g-index (Egghe 2006): largest g such that top g papers accumulate >= g^2 citations.
    - e-index (Zhang 2009): sqrt(sum(c_i - h) for i in top h papers) measuring excess citations beyond h-core.
    - i10-index: count of papers with >= 10 citations.
    """
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame()

    author_citations: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        raw_c = row.get("citation_count")
        cites = int(raw_c) if pd.notna(raw_c) else 0
        authors = row.get("authors")
        if isinstance(authors, list):
            for a in authors:
                a_clean = str(a).strip()
                if a_clean:
                    author_citations.setdefault(a_clean, []).append(cites)

    records = []
    for author, cites_list in author_citations.items():
        if len(cites_list) < min_papers:
            continue
        cites_sorted = sorted(cites_list, reverse=True)
        n = len(cites_sorted)

        # h-index
        h = 0
        for i, c in enumerate(cites_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break

        # g-index (Egghe)
        cumsum = np.cumsum(cites_sorted)
        g = 0
        for i in range(n):
            if cumsum[i] >= (i + 1) ** 2:
                g = i + 1

        # e-index (Zhang excess citations)
        e2 = sum(cites_sorted[i] - h for i in range(h)) if h > 0 else 0
        e = float(np.sqrt(max(0, e2)))

        # i10-index
        i10 = sum(1 for c in cites_sorted if c >= 10)

        total_c = sum(cites_sorted)
        mean_c = total_c / n if n > 0 else 0.0

        records.append(
            {
                "author": author,
                "papers": n,
                "total_citations": total_c,
                "mean_citations": round(mean_c, 1),
                "h_index": h,
                "g_index": g,
                "e_index": round(e, 2),
                "i10_index": i10,
                "g_h_diff": g - h,
            }
        )

    return pd.DataFrame(records).sort_values(by=["g_index", "h_index"], ascending=[False, False])


def text_readability_and_stylometrics(df: pd.DataFrame) -> dict:
    """Compute linguistic readability (Flesch Reading Ease, Flesch-Kincaid) and Type-Token Ratio."""
    import re

    if df.empty or "abstract" not in df.columns:
        return {
            "mean_fre": 0.0,
            "mean_fkgl": 0.0,
            "mean_ttr": 0.0,
            "temporal_df": pd.DataFrame(),
            "sample_df": pd.DataFrame(),
        }

    records = []
    for _, row in df.iterrows():
        text = str(row.get("abstract") or "")
        if len(text.strip()) < 30:
            continue

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 3]
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        if not sentences or not words:
            continue

        n_s = len(sentences)
        n_w = len(words)

        # Syllable approximation (vowel groups)
        def _count_syllables(word: str) -> int:
            w = word.lower()
            count = len(re.findall(r"[aeiouy]+", w))
            if w.endswith("e") and not w.endswith("le") and count > 1:
                count -= 1
            return max(1, count)

        n_syll = sum(_count_syllables(w) for w in words)
        asl = n_w / n_s
        asw = n_syll / n_w

        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
        fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
        ttr = len(set(words)) / n_w

        year = pd.to_numeric(row.get("year"), errors="coerce")
        cites = pd.to_numeric(row.get("citation_count"), errors="coerce")

        records.append(
            {
                "id": row.get("id"),
                "doi": row.get("doi"),
                "year": int(year) if pd.notna(year) else 2015,
                "citation_count": int(cites) if pd.notna(cites) else 0,
                "fre": round(float(np.clip(fre, 0, 100)), 1),
                "fkgl": round(float(np.clip(fkgl, 0, 30)), 1),
                "ttr": round(float(ttr), 3),
                "n_words": n_w,
            }
        )

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return {
            "mean_fre": 0.0,
            "mean_fkgl": 0.0,
            "mean_ttr": 0.0,
            "temporal_df": pd.DataFrame(),
            "sample_df": pd.DataFrame(),
        }

    temporal = (
        res_df[res_df["year"] >= 2005]
        .groupby("year")
        .agg(
            mean_fre=("fre", "mean"),
            mean_fkgl=("fkgl", "mean"),
            mean_ttr=("ttr", "mean"),
            n_articles=("id", "count"),
        )
        .reset_index()
        .round(2)
    )

    return {
        "mean_fre": round(float(res_df["fre"].mean()), 1),
        "mean_fkgl": round(float(res_df["fkgl"].mean()), 1),
        "mean_ttr": round(float(res_df["ttr"].mean()), 3),
        "temporal_df": temporal,
        "sample_df": res_df,
    }


def citation_longevity_and_decay(df: pd.DataFrame) -> dict:
    """Analyze literature citation decay, half-life, and identify Evergreen classical papers."""
    return {
        "available": False,
        "reason": "Citation longevity requires citations indexed by citing year.",
        "half_life_years": 0.0,
        "decay_curve": pd.DataFrame(),
        "evergreen_df": pd.DataFrame(),
    }
    if df.empty or "year" not in df.columns:
        return {
            "half_life_years": 0.0,
            "decay_curve": pd.DataFrame(),
            "evergreen_df": pd.DataFrame(),
        }

    res = df.copy()
    current_year = 2026
    res["pub_year"] = pd.to_numeric(res["year"], errors="coerce")
    res = res.dropna(subset=["pub_year"])
    res["age"] = (current_year - res["pub_year"]).clip(lower=0)
    res["cites"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0)

    # Citation accumulation by age
    age_agg = res.groupby("age")["cites"].sum().reset_index()
    age_agg = age_agg.sort_values(by="age")
    age_agg["cum_cites"] = age_agg["cites"].cumsum()
    total_cites = age_agg["cites"].sum()
    age_agg["cum_pct"] = (
        (age_agg["cum_cites"] / total_cites * 100).round(1) if total_cites > 0 else 0.0
    )

    # Find half-life age (50% of cumulative citations)
    half_life = 0.0
    for _, r in age_agg.iterrows():
        if r["cum_pct"] >= 50.0:
            half_life = float(r["age"])
            break

    # Evergreen papers: age >= 10, citations >= 40
    evergreen = res[(res["age"] >= 10) & (res["cites"] >= 40)].copy()
    evergreen["annual_velocity"] = (evergreen["cites"] / (evergreen["age"] + 1)).round(1)
    evergreen = evergreen.sort_values(by="cites", ascending=False)

    cols = ["id", "doi", "title", "year", "venue", "citation_count", "age", "annual_velocity"]
    available_cols = [c for c in cols if c in evergreen.columns]

    return {
        "half_life_years": half_life,
        "decay_curve": age_agg,
        "evergreen_df": evergreen[available_cols].head(25),
    }


def objective_functions_taxonomy(df: pd.DataFrame) -> dict:
    """Analyze mathematical objective functions and multi-criteria formulations.

    Mapeia os critérios e objetivos otimizados na literatura:
    - Custos Econômicos (CAPEX/OPEX de ativos)
    - Confiabilidade & Índices de Interrupção (SAIDI/SAIFI/ENS)
    - Perdas Técnicas de Energia (I^2R)
    - Perfil de Tensão & Estabilidade
    - Emissões & Descarbonização (CO2)
    - Resiliência a Eventos Extremos (Blackouts/Clima)

    Calcula:
    - summary_df: [objective, articles, pct_recent, mean_citations]
    - co_matrix: DataFrame simétrico de co-ocorrência dos objetivos
    - multi_obj_ratio: percentual de artigos com >= 2 objetivos
    - temporal_multiobj: série temporal de artigos mono-objetivo vs. multi-objetivo
    """
    if df.empty:
        return {
            "summary_df": pd.DataFrame(),
            "co_matrix": pd.DataFrame(),
            "multi_obj_ratio": 0.0,
            "temporal_multiobj": pd.DataFrame(),
        }

    objs = {
        "Custos Econômicos": r"cost|capex|opex|investment|economic|capital expenditure",
        "Confiabilidade (SAIDI/SAIFI/ENS)": r"reliability|saidi|saifi|ens|energy not supplied|interruption|outage|unserved",
        "Perdas Técnicas": r"power loss|energy loss|technical loss|transmission loss|loss reduction",
        "Perfil de Tensão": r"voltage profile|voltage deviation|voltage stability|power quality|voltage drop|voltage regulation",
        "Descarbonização / Emissões": r"emission|carbon|decarboniz|greenhouse|environmental|co2",
        "Resiliência": r"resilience|extreme weather|disaster|blackout|hardening|restoration",
    }

    titles_abs = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in objs.items()}
    summary_rows = []
    for k, m in matches.items():
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        summary_rows.append(
            {
                "objective": k,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)

    co_matrix = pd.DataFrame(index=list(objs.keys()), columns=list(objs.keys()), dtype=int)
    for k1 in objs:
        for k2 in objs:
            co_matrix.loc[k1, k2] = int((matches[k1] & matches[k2]).sum())

    match_matrix = np.column_stack([matches[k].to_numpy() for k in objs])
    obj_counts = match_matrix.sum(axis=1)
    has_any = obj_counts >= 1
    multi_count = int((obj_counts >= 2).sum())
    multi_ratio = float(multi_count / has_any.sum() * 100) if has_any.sum() > 0 else 0.0

    valid_years = (years >= 2005) & (years <= 2026) & has_any
    temp_df = pd.DataFrame(
        {
            "year": years[valid_years].astype(int),
            "is_multi": (obj_counts[valid_years] >= 2),
        }
    )
    if not temp_df.empty:
        grouped = temp_df.groupby(["year", "is_multi"]).size().unstack(fill_value=0)
        temporal_multiobj = pd.DataFrame(
            {
                "year": grouped.index,
                "mono_objective": grouped.get(False, 0),
                "multi_objective": grouped.get(True, 0),
            }
        ).reset_index(drop=True)
        total = temporal_multiobj["mono_objective"] + temporal_multiobj["multi_objective"]
        temporal_multiobj["pct_multi"] = (
            temporal_multiobj["multi_objective"] / total.clip(lower=1) * 100
        ).round(1)
    else:
        temporal_multiobj = pd.DataFrame()

    return {
        "summary_df": summary_df,
        "co_matrix": co_matrix,
        "multi_obj_ratio": round(multi_ratio, 1),
        "temporal_multiobj": temporal_multiobj,
    }


def uncertainty_paradigms_analysis(df: pd.DataFrame) -> dict:
    """Analyze mathematical paradigms for handling uncertainty in distribution systems.

    - Programação Estocástica (Monte Carlo / Cenários)
    - Otimização Robusta (Min-Max / Uncertainty Sets)
    - Lógica Nebulosa (Fuzzy)
    - Restrições Probabilísticas (Chance-Constrained)
    - Otimização Distribucionalmente Robusta (DRO / Wasserstein)
    - Abordagem Determinística
    """
    if df.empty:
        return {
            "paradigms_df": pd.DataFrame(),
            "cross_resources": pd.DataFrame(),
            "temporal_paradigms": pd.DataFrame(),
        }

    paradigms = {
        "Estocástico (Cenários / Monte Carlo)": r"stochastic|scenario-based|monte carlo|sample average",
        "Otimização Robusta (Min-Max)": r"robust optimization|robust approach|uncertainty set|worst-case",
        "Lógica Nebulosa (Fuzzy)": r"fuzzy",
        "Distribucionalmente Robusta (DRO)": r"distributionally robust|wasserstein|ambiguity set",
        "Restrição de Chance (Probabilística)": r"chance-constrained|chance constraint|probabilistic constraint",
        "Determinístico (Caso Fixo)": r"deterministic",
    }

    resources = {
        "Geração Solar (PV)": r"photovoltaic|\bpv\b|solar",
        "Baterias / Armazenamento (BESS)": r"energy storage|battery|bess",
        "Veículos Elétricos (EV)": r"electric vehicle|\bev\b|v2g|charging",
        "Incerteza de Demanda / Carga": r"load uncertainty|demand uncertainty|forecast error|load variation",
    }

    titles_abs = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    p_matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in paradigms.items()}
    r_matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in resources.items()}

    summary_rows = []
    for k, m in p_matches.items():
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        summary_rows.append(
            {
                "paradigm": k,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    paradigms_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)

    cross_data = {r_label: [] for r_label in resources}
    for _p_label, p_m in p_matches.items():
        for r_label, r_m in r_matches.items():
            cross_data[r_label].append(int((p_m & r_m).sum()))

    cross_resources = pd.DataFrame(cross_data, index=list(paradigms.keys()))

    temporal_dict = {}
    for p_label, p_m in p_matches.items():
        filt = p_m & (years >= 2005) & (years <= 2026)
        if filt.any():
            yearly = df[filt].groupby(years[filt].astype(int))["id"].count()
            temporal_dict[p_label] = yearly

    temporal_paradigms = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    if not temporal_paradigms.empty:
        temporal_paradigms.index.name = "year"
        temporal_paradigms = temporal_paradigms.reset_index()

    return {
        "paradigms_df": paradigms_df,
        "cross_resources": cross_resources,
        "temporal_paradigms": temporal_paradigms,
    }


def planning_time_horizons_analysis(df: pd.DataFrame) -> dict:
    """Analyze static vs. multi-stage dynamic planning and operation co-optimization."""
    if df.empty:
        return {"horizons_df": pd.DataFrame()}

    horizons = {
        "Expansão Dinâmica Multi-Estágio": r"multi-stage|multistage|multi-year|sequential expansion|expansion planning|dynamic planning",
        "Co-Otimização Planejamento + Operação": r"co-optimi|planning and operation|representative days|representative periods|operational constraints|chronological",
        "Planejamento Estático (Ano-Alvo)": r"static planning|single-stage|target year|snapshot",
    }

    titles_abs = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    for h_label, pat in horizons.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "horizon": h_label,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    horizons_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    return {"horizons_df": horizons_df}


def computational_solvers_analysis(df: pd.DataFrame) -> dict:
    """Analyze mathematical solvers and simulation platforms utilized in the literature."""
    if df.empty:
        return {"solvers_df": pd.DataFrame(), "ecosystem_df": pd.DataFrame()}

    solvers = {
        "GAMS / AMPL (Modeladores Algébricos)": (r"gams|ampl", "Modelador Algébrico"),
        "MATLAB / Simulink": (r"matlab|simulink", "Scripting & Simulação"),
        "CPLEX (IBM)": (r"cplex", "Solver Exato Comercial"),
        "DIgSILENT PowerFactory": (r"digsilent|powerfactory", "Simulador Elétrico Especializado"),
        "Gurobi Optimizer": (r"gurobi", "Solver Exato Comercial"),
        "OpenDSS (EPRI)": (r"opendss|open dss", "Simulador de Distribuição"),
        "Python (Pyomo / Pandapower)": (r"python|pyomo|pandapower", "Scripting & Open-Source"),
        "PSCAD / EMTP": (r"pscad|emtp", "Simulador de Transitórios"),
    }

    titles_abs = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    for s_label, (pat, cat) in solvers.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "tool": s_label,
                "category": cat,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    solvers_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    eco_df = (
        solvers_df.groupby("category")["articles"]
        .sum()
        .reset_index()
        .sort_values(by="articles", ascending=False)
    )

    return {
        "solvers_df": solvers_df,
        "ecosystem_df": eco_df,
    }


def mathematical_complexity_spectrum(df: pd.DataFrame) -> dict:
    """Classify papers into mathematical complexity classes."""
    if df.empty:
        return {"spectrum_df": pd.DataFrame(), "temporal_spectrum": pd.DataFrame()}

    classes = {
        "Prog. Linear / MILP (Exato)": r"milp|mixed-integer linear|\blp\b|linear programming",
        "Relaxação Convexa / Cônica (SOCP/SDP)": r"second-order cone|socp|semidefinite|convex relaxation|conic",
        "Prog. Não-Linear (NLP / MINLP)": r"minlp|mixed-integer nonlinear|nonlinear programming|\bnlp\b|non-convex",
        "Meta-heurísticas (GA, PSO, DE, ACO)": r"genetic algorithm|particle swarm|\bpso\b|\bga\b|differential evolution|ant colony|harmony search|simulated annealing",
        "IA & Aprendizado por Reforço": r"machine learning|deep learning|reinforcement learning|neural network|q-learning",
    }

    titles_abs = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    temporal_dict = {}

    for c_label, pat in classes.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "complexity_class": c_label,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

        filt = m & (years >= 2005) & (years <= 2026)
        if filt.any():
            yearly = df[filt].groupby(years[filt].astype(int))["id"].count()
            temporal_dict[c_label] = yearly

    spectrum_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    temporal_spectrum = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    if not temporal_spectrum.empty:
        temporal_spectrum.index.name = "year"
        temporal_spectrum = temporal_spectrum.reset_index()

    return {
        "spectrum_df": spectrum_df,
        "temporal_spectrum": temporal_spectrum,
    }


def author_m_quotient_analysis(df: pd.DataFrame, min_papers: int = 2) -> pd.DataFrame:
    """Compute career-length normalized impact (Hirsch's m-quotient = h / career_years).

    Distinguishes rapidly emerging high-velocity researchers from established veterans.
    """
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame()

    current_year = 2026
    author_records: dict[str, dict] = {}

    for _, row in df.iterrows():
        raw_c = row.get("citation_count")
        cites = int(raw_c) if pd.notna(raw_c) else 0
        raw_y = row.get("year")
        y = int(raw_y) if pd.notna(raw_y) else 2015
        authors = row.get("authors")
        if isinstance(authors, list):
            for a in authors:
                a_clean = str(a).strip()
                if a_clean:
                    rec = author_records.setdefault(a_clean, {"cites": [], "years": []})
                    rec["cites"].append(cites)
                    rec["years"].append(y)

    rows = []
    for author, data in author_records.items():
        c_list = data["cites"]
        if len(c_list) < min_papers:
            continue
        c_sorted = sorted(c_list, reverse=True)
        h = 0
        for i, c in enumerate(c_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break

        first_year = min(data["years"])
        career_span = max(1, current_year - first_year + 1)
        m_quotient = round(h / career_span, 2)

        rows.append(
            {
                "author": author,
                "papers": len(c_list),
                "total_citations": sum(c_list),
                "h_index": h,
                "first_year": first_year,
                "career_span_years": career_span,
                "m_quotient": m_quotient,
            }
        )

    m_df = pd.DataFrame(rows).sort_values(by="m_quotient", ascending=False)
    return m_df


# ---------------------------------------------------------------------------
# Section: Knowledge Frontiers, Scientometric Disruption & Open Access Dynamics
# ---------------------------------------------------------------------------


def price_index_analysis(df: pd.DataFrame) -> dict:
    """Evaluate theoretical recency via Derek de Solla Price's (1965) Index.

    The Price Index measures the proportion of cited references published in the
    preceding 5 years. In fields with high technological dynamism (e.g. AI, Storage),
    the Price Index typically exceeds 35-40%, whereas canonical fields stay below 25%.
    """
    return {
        "available": False,
        "reason": "Price's index requires publication years for cited references.",
        "global_price_index": 0.0,
        "theme_price_df": pd.DataFrame(),
        "yearly_price_df": pd.DataFrame(),
    }
    if df.empty or "year" not in df.columns:
        return {
            "global_price_index": 0.0,
            "theme_price_df": pd.DataFrame(),
            "yearly_price_df": pd.DataFrame(),
        }

    res = df.copy()
    res["pub_year"] = valid_years(res, lo=1990, hi=2026)
    res = res.dropna(subset=["pub_year"])
    if res.empty:
        return {
            "global_price_index": 0.0,
            "theme_price_df": pd.DataFrame(),
            "yearly_price_df": pd.DataFrame(),
        }

    res["refs"] = pd.to_numeric(res.get("reference_count"), errors="coerce").fillna(20.0)
    res["cites"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0.0)

    # Calibrate Price's Index model:
    # Based on Price (1965) and Glänzel & Schoepflin (1995),
    # P_i relates to contemporary reference expansion and publication vintage.
    year_factor = ((res["pub_year"] - 1990) / 35.0).clip(0, 1) * 15.0
    ref_factor = np.minimum(res["refs"] / 40.0, 1.5) * 8.0
    margin_factor = 0.0
    if "relevance_margin" in res.columns:
        margin_factor = (
            pd.to_numeric(res["relevance_margin"], errors="coerce").fillna(0.0) * 5.0
        ).clip(-5, 10)

    res["price_index"] = (22.0 + year_factor + ref_factor + margin_factor).clip(10.0, 65.0)

    # Global index
    global_price = float(res["price_index"].mean())

    # By Theme
    if "theme_label" in res.columns:
        theme_grp = (
            res.groupby("theme_label")
            .agg(
                price_index=("price_index", "mean"),
                mean_refs=("refs", "mean"),
                articles=("title", "count"),
                mean_cites=("cites", "mean"),
            )
            .reset_index()
        )
        theme_grp["price_index"] = theme_grp["price_index"].round(1)
        theme_grp["mean_refs"] = theme_grp["mean_refs"].round(1)
        theme_grp["mean_cites"] = theme_grp["mean_cites"].round(1)
        theme_grp = theme_grp.sort_values(by="price_index", ascending=False)
    else:
        theme_grp = pd.DataFrame()

    # Yearly evolution
    yearly_grp = (
        res.groupby("pub_year")
        .agg(
            price_index=("price_index", "mean"),
            mean_refs=("refs", "mean"),
            articles=("title", "count"),
        )
        .reset_index()
    )
    yearly_grp["pub_year"] = yearly_grp["pub_year"].astype(int)
    yearly_grp["price_index"] = yearly_grp["price_index"].round(1)
    yearly_grp["mean_refs"] = yearly_grp["mean_refs"].round(1)
    yearly_grp = yearly_grp.sort_values(by="pub_year")

    return {
        "global_price_index": round(global_price, 1),
        "theme_price_df": theme_grp,
        "yearly_price_df": yearly_grp,
    }


def sleeping_beauties_detection(df: pd.DataFrame, min_age: int = 7) -> dict:
    """Detect delayed recognition articles (Sleeping Beauties in Science).

    Formalized by van Raan (2004) and Ke et al. (2015, PNAS). Identifies papers
    published at least `min_age` years ago with significant cumulative citations
    whose impact experienced a prolonged dormancy period before an awakening surge.
    """
    return {
        "available": False,
        "reason": "Sleeping Beauty detection requires annual citation histories.",
        "sleeping_beauties": pd.DataFrame(),
        "top_trajectories": pd.DataFrame(),
        "count": 0,
    }
    if df.empty or "year" not in df.columns:
        return {"sleeping_beauties": pd.DataFrame(), "top_trajectories": pd.DataFrame(), "count": 0}

    res = df.copy()
    current_year = 2026
    res["pub_year"] = valid_years(res, lo=1980, hi=current_year)
    res = res.dropna(subset=["pub_year"])
    res["age"] = (current_year - res["pub_year"]).clip(lower=0)
    res["cites"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0)

    # Candidate filter: age >= min_age and citations >= 15
    candidates = res[(res["age"] >= min_age) & (res["cites"] >= 15)].copy()
    if candidates.empty:
        return {"sleeping_beauties": pd.DataFrame(), "top_trajectories": pd.DataFrame(), "count": 0}

    # Beauty Coefficient B calculation (Ke et al., 2015 approximation):
    lags = []
    b_scores = []

    for _, row in candidates.iterrows():
        age = row["age"]
        cites = row["cites"]
        lag = max(3, int(np.round(age * 0.55)))
        b = (cites * lag) / (age + 1.0)
        lags.append(lag)
        b_scores.append(round(b, 2))

    candidates["awakening_lag"] = lags
    candidates["beauty_coefficient"] = b_scores
    candidates = candidates.sort_values(by="beauty_coefficient", ascending=False)

    keep_cols = [
        "id",
        "doi",
        "title",
        "year",
        "venue",
        "citation_count",
        "age",
        "awakening_lag",
        "beauty_coefficient",
        "theme_label",
    ]
    available_cols = [c for c in keep_cols if c in candidates.columns]
    sb_table = candidates[available_cols].head(30)

    # Historical trajectories for top 5 Sleeping Beauties
    top5 = candidates.head(5)
    traj_records = []
    for _, r in top5.iterrows():
        pub = int(r["pub_year"])
        tot_c = float(r["cites"])
        lag = int(r["awakening_lag"])
        title_val = str(r.get("title") or "Artigo")
        short_title = (title_val[:38] + "...") if len(title_val) > 40 else title_val

        for yr in range(pub, current_year + 1):
            elapsed = yr - pub
            if elapsed <= lag:
                cum = (tot_c * 0.12) * (elapsed / max(1, lag))
            else:
                surge_ratio = (elapsed - lag) / max(1, (r["age"] - lag))
                cum = (tot_c * 0.12) + (tot_c * 0.88) * surge_ratio

            traj_records.append(
                {
                    "paper": short_title,
                    "year": yr,
                    "cum_citations": round(cum, 1),
                    "phase": "Dormência" if elapsed <= lag else "Despertar",
                }
            )

    traj_df = pd.DataFrame(traj_records)

    return {
        "sleeping_beauties": sb_table,
        "top_trajectories": traj_df,
        "count": len(candidates),
    }


def disruption_index_estimation(df: pd.DataFrame) -> dict:
    """Estimate the CD Disruption Index (Wu, Wang & Evans, Nature 2019).

    CD in [-1, +1] quantifies whether a paper destabilizes the existing paradigm
    (CD > 0, introducing novel concepts that eclipse older references) or
    consolidates it (CD < 0, developing established approaches).
    Also tests the hypothesis that small teams are more disruptive than large teams.
    """
    return {
        "available": False,
        "reason": "The CD index requires a forward and backward citation graph.",
        "disruption_df": pd.DataFrame(),
        "team_size_analysis": pd.DataFrame(),
        "theme_disruption": pd.DataFrame(),
        "disruptive_ratio": 0.0,
    }
    if df.empty:
        return {
            "disruption_df": pd.DataFrame(),
            "team_size_analysis": pd.DataFrame(),
            "theme_disruption": pd.DataFrame(),
            "disruptive_ratio": 0.0,
        }

    res = df.copy()
    res["cites"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0.0)
    res["refs"] = pd.to_numeric(res.get("reference_count"), errors="coerce").fillna(18.0)

    def get_team_size(authors_val) -> int:
        if isinstance(authors_val, list):
            return len(authors_val) if authors_val else 1
        if isinstance(authors_val, str) and authors_val.strip():
            parts = [p.strip() for p in re.split(r"[,;]", authors_val) if p.strip()]
            return len(parts) if parts else 1
        return 1

    res["team_size"] = res["authors"].apply(get_team_size) if "authors" in res.columns else 1
    res["team_bucket"] = res["team_size"].apply(lambda n: str(n) if n <= 5 else "6+")

    # CD Index model:
    # High citations relative to prior references, amplified by semantic margin / originality
    # produces positive disruption.
    log_c = np.log1p(res["cites"])
    log_r = np.log1p(res["refs"])
    raw_cd = (log_c - 0.85 * log_r) / 3.0

    if "relevance_margin" in res.columns:
        margin = pd.to_numeric(res["relevance_margin"], errors="coerce").fillna(0.5)
        raw_cd += (margin - 0.5) * 0.4

    team_penalty = (res["team_size"] - 2.5) * 0.035
    res["cd_index"] = np.clip(np.tanh(raw_cd - team_penalty), -1.0, 1.0).round(3)
    res["is_disruptive"] = res["cd_index"] > 0

    disruptive_ratio = float((res["is_disruptive"].sum() / len(res)) * 100) if len(res) > 0 else 0.0

    # Team size aggregation
    order = ["1", "2", "3", "4", "5", "6+"]
    team_agg = (
        res.groupby("team_bucket")
        .agg(
            mean_cd=("cd_index", "mean"),
            median_cd=("cd_index", "median"),
            pct_disruptive=("is_disruptive", lambda s: float((s.sum() / len(s)) * 100)),
            articles=("cites", "count"),
        )
        .reindex(order)
        .dropna(subset=["articles"])
        .reset_index()
    )
    team_agg["mean_cd"] = team_agg["mean_cd"].round(3)
    team_agg["pct_disruptive"] = team_agg["pct_disruptive"].round(1)

    # Theme disruption aggregation
    if "theme_label" in res.columns:
        theme_agg = (
            res.groupby("theme_label")
            .agg(
                mean_cd=("cd_index", "mean"),
                pct_disruptive=("is_disruptive", lambda s: float((s.sum() / len(s)) * 100)),
                articles=("cites", "count"),
            )
            .reset_index()
            .sort_values(by="mean_cd", ascending=False)
        )
        theme_agg["mean_cd"] = theme_agg["mean_cd"].round(3)
        theme_agg["pct_disruptive"] = theme_agg["pct_disruptive"].round(1)
    else:
        theme_agg = pd.DataFrame()

    return {
        "disruption_df": res,
        "team_size_analysis": team_agg,
        "theme_disruption": theme_agg,
        "disruptive_ratio": round(disruptive_ratio, 1),
    }


def open_access_impact_analysis(df: pd.DataFrame) -> dict:
    """Analyze the Open Access Citation Advantage (OACA) and licensing dynamics.

    Compares citation accrual between Open Access articles (e.g. Creative Commons licenses)
    and proprietary closed-access publications across time.
    """
    return {
        "available": False,
        "reason": "OACA requires verified access status and confounder controls.",
        "oa_share_pct": 0.0,
        "oaca_ratio": 1.0,
        "yearly_oa": pd.DataFrame(),
        "comparison_table": pd.DataFrame(),
        "license_dist": pd.DataFrame(),
    }
    if df.empty:
        return {
            "oa_share_pct": 0.0,
            "oaca_ratio": 1.0,
            "yearly_oa": pd.DataFrame(),
            "comparison_table": pd.DataFrame(),
            "license_dist": pd.DataFrame(),
        }

    res = df.copy()
    res["pub_year"] = valid_years(res, lo=1995, hi=2026)
    res = res.dropna(subset=["pub_year"])
    res["cites"] = pd.to_numeric(res.get("citation_count"), errors="coerce").fillna(0)

    def is_open_access(row) -> bool:
        lic = str(row.get("license") or "").lower()
        doc = str(row.get("document_type") or "").lower()
        if any(w in lic for w in ("cc", "creative", "open", "gold", "by", "public")):
            return True
        if "open access" in doc or "open" in lic:
            return True
        return False

    res["is_oa"] = res.apply(is_open_access, axis=1)

    oa_count = res["is_oa"].sum()
    if oa_count < 10 and "doi" in res.columns:
        res["is_oa"] = res["doi"].apply(lambda d: hash(str(d)) % 5 == 0)

    res["access_type"] = res["is_oa"].map(
        {True: "Acesso Aberto (OA)", False: "Acesso Fechado / Assinatura"}
    )

    oa_share = float((res["is_oa"].sum() / len(res)) * 100) if len(res) > 0 else 0.0

    oa_stats = res[res["is_oa"]]["cites"]
    closed_stats = res[~res["is_oa"]]["cites"]

    mean_oa = float(oa_stats.mean()) if len(oa_stats) > 0 else 0.0
    mean_closed = float(closed_stats.mean()) if len(closed_stats) > 0 else 1.0
    oaca_ratio = round(mean_oa / max(0.1, mean_closed), 2)

    comp_rows = [
        {
            "Modalidade de Acesso": "Acesso Aberto (OA)",
            "Artigos": int(len(oa_stats)),
            "Citações Médias": round(mean_oa, 1),
            "Mediana Citações": round(float(oa_stats.median()) if len(oa_stats) > 0 else 0.0, 1),
            "Percentil 75": round(float(oa_stats.quantile(0.75)) if len(oa_stats) > 0 else 0.0, 1),
        },
        {
            "Modalidade de Acesso": "Acesso Fechado (Paywall)",
            "Artigos": int(len(closed_stats)),
            "Citações Médias": round(mean_closed, 1),
            "Mediana Citações": round(
                float(closed_stats.median()) if len(closed_stats) > 0 else 0.0, 1
            ),
            "Percentil 75": round(
                float(closed_stats.quantile(0.75)) if len(closed_stats) > 0 else 0.0, 1
            ),
        },
    ]
    comp_df = pd.DataFrame(comp_rows)

    # Yearly OA penetration
    yearly = (
        res.groupby("pub_year")
        .agg(
            total=("cites", "count"),
            oa_count=("is_oa", "sum"),
        )
        .reset_index()
    )
    yearly["pub_year"] = yearly["pub_year"].astype(int)
    yearly["oa_pct"] = ((yearly["oa_count"] / yearly["total"]) * 100).round(1)

    # Compute mean cites per year for OA and closed
    mean_oa_yr = []
    mean_closed_yr = []
    for yr in yearly["pub_year"]:
        sub_yr = res[res["pub_year"] == yr]
        c_oa = sub_yr[sub_yr["is_oa"]]["cites"]
        c_cl = sub_yr[~sub_yr["is_oa"]]["cites"]
        mean_oa_yr.append(round(float(c_oa.mean()), 1) if len(c_oa) > 0 else 0.0)
        mean_closed_yr.append(round(float(c_cl.mean()), 1) if len(c_cl) > 0 else 0.0)

    yearly["mean_cites_oa"] = mean_oa_yr
    yearly["mean_cites_closed"] = mean_closed_yr
    yearly = yearly.sort_values(by="pub_year")

    # License distribution
    if "license" in res.columns:
        lic_s = (
            res["license"].fillna("Não especificado / Fechado").value_counts().head(8).reset_index()
        )
        lic_s.columns = ["Licença", "Quantidade"]
    else:
        lic_s = pd.DataFrame(
            {"Licença": ["Fechado", "Aberto"], "Quantidade": [len(closed_stats), len(oa_stats)]}
        )

    return {
        "oa_share_pct": round(oa_share, 1),
        "oaca_ratio": oaca_ratio,
        "yearly_oa": yearly,
        "comparison_table": comp_df,
        "license_dist": lic_s,
    }


def technological_burst_detection(
    df: pd.DataFrame,
    top_n: int = 15,
    *,
    state_multiplier: float = 2.0,
    transition_penalty: float = 1.0,
) -> dict:
    """Detect two-state Kleinberg bursts in yearly concept frequencies.

    The observation at each year is the number of matching documents out of
    all documents published that year. Dynamic programming chooses between a
    baseline binomial state and a burst state with ``state_multiplier`` times
    the baseline probability. Moving into the burst state incurs Kleinberg's
    transition penalty; contiguous burst-state years become intervals.
    """
    if df.empty or "year" not in df.columns:
        return {
            "burst_timeline": pd.DataFrame(),
            "active_frontiers": pd.DataFrame(),
            "total_bursts": 0,
        }

    res = df.copy()
    res["pub_year"] = valid_years(res, lo=2000, hi=2026)
    res = res.dropna(subset=["pub_year"])
    if res.empty:
        return {
            "burst_timeline": pd.DataFrame(),
            "active_frontiers": pd.DataFrame(),
            "total_bursts": 0,
        }

    text_corpus = (
        res["title"].fillna("")
        + " "
        + res.get("abstract", pd.Series("", index=res.index)).fillna("")
    ).str.lower()

    frontiers_map = {
        "Vehicle-to-Grid & EVs": r"\b(?:v2g|vehicle[- ]to[- ]grid|electric vehicles?|ev charging)\b",
        "Deep Learning & RL": r"\b(?:deep learning|reinforcement learning|neural networks?|transformer)\b",
        "Battery Storage (BESS)": r"\b(?:battery storage|bess|lithium[- ]ion|energy storage systems?)\b",
        "Hosting Capacity": r"\b(?:hosting capacity|solar pv integration|der integration)\b",
        "Resilience & Extreme Events": r"\b(?:resilience|extreme weather|blackout|natural disasters?)\b",
        "Second-Order Cone (SOCP)": r"\b(?:socp|second[- ]order cone|convex relaxation)\b",
        "Microgrids & Islanding": r"\b(?:microgrids?|islanding|autonomous operation)\b",
        "Active Distribution (ADN)": r"\b(?:active distribution|adn|distribution management system)\b",
        "Peer-to-Peer Energy (P2P)": r"\b(?:p2p|peer[- ]to[- ]peer|transactive energy|blockchain)\b",
        "Digital Twin & IoT": r"\b(?:digital twin|iot|smart meters?|pmu|edge computing)\b",
        "Robust Optimization": r"\b(?:robust optimization|two[- ]stage robust|box uncertainty)\b",
        "Soft Open Points (SOP)": r"\b(?:soft open points?|sop|power electronic devices?)\b",
    }

    all_years = np.arange(int(res["pub_year"].min()), int(res["pub_year"].max()) + 1)
    totals = res["pub_year"].astype(int).value_counts().reindex(all_years, fill_value=0).to_numpy()
    burst_rows = []

    def _binomial_cost(successes: int, trials: int, probability: float) -> float:
        if trials <= 0:
            return 0.0
        probability = float(np.clip(probability, 1e-9, 1 - 1e-9))
        return -(successes * np.log(probability) + (trials - successes) * np.log1p(-probability))

    def _states(counts: np.ndarray) -> tuple[np.ndarray, float, float]:
        total_trials = int(totals.sum())
        baseline = float(counts.sum() / total_trials) if total_trials else 0.0
        elevated = min(max(baseline * state_multiplier, baseline + 1e-9), 1 - 1e-9)
        costs = np.full((len(all_years), 2), np.inf)
        previous = np.zeros((len(all_years), 2), dtype=int)
        entry_cost = transition_penalty * np.log(max(total_trials, 2))
        costs[0, 0] = _binomial_cost(int(counts[0]), int(totals[0]), baseline)
        costs[0, 1] = entry_cost + _binomial_cost(int(counts[0]), int(totals[0]), elevated)
        for index in range(1, len(all_years)):
            for state in (0, 1):
                candidates = costs[index - 1].copy()
                if state == 1:
                    candidates[0] += entry_cost
                source_state = int(np.argmin(candidates))
                previous[index, state] = source_state
                probability = elevated if state else baseline
                costs[index, state] = candidates[source_state] + _binomial_cost(
                    int(counts[index]), int(totals[index]), probability
                )
        states = np.zeros(len(all_years), dtype=int)
        states[-1] = int(np.argmin(costs[-1]))
        for index in range(len(all_years) - 1, 0, -1):
            states[index - 1] = previous[index, states[index]]
        return states, baseline, elevated

    for label, pattern in frontiers_map.items():
        matched = text_corpus.str.contains(pattern, regex=True)
        if matched.sum() < 6:
            continue

        matched_years = res.loc[matched, "pub_year"].astype(int)
        counts = matched_years.value_counts().reindex(all_years, fill_value=0).to_numpy()
        states, baseline, elevated = _states(counts)
        index = 0
        while index < len(states):
            if states[index] == 0:
                index += 1
                continue
            end_index = index
            while end_index + 1 < len(states) and states[end_index + 1] == 1:
                end_index += 1
            segment = slice(index, end_index + 1)
            interval_counts = counts[segment]
            interval_totals = totals[segment]
            strength = sum(
                _binomial_cost(int(k), int(n), baseline) - _binomial_cost(int(k), int(n), elevated)
                for k, n in zip(interval_counts, interval_totals, strict=True)
            )
            peak_index = index + int(np.argmax(interval_counts))
            is_active = end_index == len(states) - 1
            burst_rows.append(
                {
                    "Tecnologia / Conceito": label,
                    "Início do Burst": int(all_years[index]),
                    "Ano de Pico": int(all_years[peak_index]),
                    "Fim do Burst": int(all_years[end_index]),
                    "Duração (Anos)": int(end_index - index + 1),
                    "Intensidade": round(float(max(strength, 0.0)), 2),
                    "Status": "Ativo" if is_active else "Encerrado",
                    "Artigos": int(interval_counts.sum()),
                }
            )
            index = end_index + 1

    burst_df = pd.DataFrame(burst_rows)
    if not burst_df.empty:
        burst_df = burst_df.sort_values("Intensidade", ascending=False).head(top_n)
        active_df = burst_df[burst_df["Status"] == "Ativo"].copy()
    else:
        active_df = pd.DataFrame()

    return {
        "burst_timeline": burst_df,
        "active_frontiers": active_df,
        "total_bursts": len(burst_df),
    }


def detect_structural_breaks(series: pd.Series | np.ndarray) -> dict:
    """Detect structural breaks / changepoints in time series using Chow test and SSE minimization.

    Identifies historical regime shifts (inflection points) where the underlying mean
    or momentum fundamentally changed.
    """
    from scipy import stats

    if isinstance(series, pd.Series):
        years = series.index.to_numpy()
        values = series.to_numpy(dtype=float)
    else:
        values = np.asarray(series, dtype=float)
        years = np.arange(len(values))

    valid_mask = np.isfinite(values)
    values = values[valid_mask]
    years = years[valid_mask]
    T = len(values)

    if T < 6:
        return {
            "has_break": False,
            "break_index": None,
            "break_year": None,
            "f_stat": 0.0,
            "p_value": 1.0,
            "pre_mean": float(np.mean(values)) if T > 0 else 0.0,
            "post_mean": float(np.mean(values)) if T > 0 else 0.0,
            "relative_jump_pct": 0.0,
        }

    tss = float(np.sum((values - np.mean(values)) ** 2))
    best_rss = float("inf")
    best_k = None

    # Search for breakpoint k in [2, T-3]
    for k in range(2, T - 2):
        pre = values[:k]
        post = values[k:]
        rss = float(np.sum((pre - np.mean(pre)) ** 2) + np.sum((post - np.mean(post)) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_k = k

    if best_k is None or best_rss <= 0 or tss <= best_rss:
        return {
            "has_break": False,
            "break_index": None,
            "break_year": None,
            "f_stat": 0.0,
            "p_value": 1.0,
            "pre_mean": float(np.mean(values)),
            "post_mean": float(np.mean(values)),
            "relative_jump_pct": 0.0,
        }

    p = 1  # 1 degree of freedom for mean shift
    df1 = p
    df2 = T - 2 * p
    f_stat = float(((tss - best_rss) / df1) / (best_rss / df2))
    p_value = float(stats.f.sf(f_stat, df1, df2))

    pre_mean = float(np.mean(values[:best_k]))
    post_mean = float(np.mean(values[best_k:]))
    jump = float(((post_mean - pre_mean) / max(0.1, pre_mean)) * 100)

    break_year = int(years[best_k]) if len(years) > best_k else best_k

    return {
        "has_break": bool(p_value < 0.05),
        "break_index": best_k,
        "break_year": break_year,
        "f_stat": round(f_stat, 2),
        "p_value": round(p_value, 4),
        "pre_mean": round(pre_mean, 2),
        "post_mean": round(post_mean, 2),
        "relative_jump_pct": round(jump, 1),
    }


def conceptual_atypicality_analysis(df: pd.DataFrame, top_n_keywords: int = 50) -> dict:
    """Describe uncommon keyword combinations and their citation association.

    This independence-based co-occurrence diagnostic is exploratory. It is
    not the journal-pair randomized null model introduced by Uzzi et al.
    """
    import itertools
    from collections import Counter

    if df.empty or "keywords" not in df.columns:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Extract keywords per article
    doc_kws = []
    kw_counts = Counter()
    for kws in df["keywords"]:
        if isinstance(kws, list):
            cleaned = [str(k).strip().lower() for k in kws if str(k).strip()]
            unique_k = sorted(set(cleaned))
            doc_kws.append(unique_k)
            kw_counts.update(unique_k)
        else:
            doc_kws.append([])

    top_vocab = {kw for kw, _ in kw_counts.most_common(top_n_keywords)}
    if len(top_vocab) < 3:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Count pairs across all articles
    pair_counts = Counter()
    total_pairs = 0
    for k_list in doc_kws:
        filtered = [k for k in k_list if k in top_vocab]
        for pair in itertools.combinations(filtered, 2):
            sorted_pair = tuple(sorted(pair))
            pair_counts[sorted_pair] += 1
            total_pairs += 1

    if total_pairs == 0:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Compute expected frequencies and z-scores under random null model
    pair_zscores = {}
    pair_rows = []
    for pair, count in pair_counts.items():
        k1, k2 = pair
        n1 = kw_counts[k1]
        n2 = kw_counts[k2]
        expected = (n1 * n2) / max(1, len(doc_kws))
        var = max(0.1, expected * (1.0 - n1 / len(doc_kws)) * (1.0 - n2 / len(doc_kws)))
        z = (count - expected) / np.sqrt(var)
        pair_zscores[pair] = float(z)
        if z < 0:
            pair_rows.append(
                {
                    "Termo 1": k1,
                    "Termo 2": k2,
                    "Coocorrência Real": count,
                    "Esperada": round(expected, 1),
                    "Z-Score (Atipicidade)": round(z, 2),
                }
            )

    pair_rows.sort(key=lambda x: x["Z-Score (Atipicidade)"])
    atypical_pairs_df = pd.DataFrame(pair_rows).head(20)

    # Score each article
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0).to_numpy()
    cite_p95 = float(np.percentile(cites, 95)) if len(cites) > 0 else 50.0

    art_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        k_list = doc_kws[idx] if idx < len(doc_kws) else []
        filtered = [k for k in k_list if k in top_vocab]
        pairs = [tuple(sorted(p)) for p in itertools.combinations(filtered, 2)]
        zs = [pair_zscores[p] for p in pairs if p in pair_zscores]

        if zs:
            med_z = float(np.median(zs))
            min_z = float(np.min(zs))
        else:
            med_z = 0.0
            min_z = 0.0

        c = float(cites[idx])
        is_hit = c >= cite_p95

        art_rows.append(
            {
                "doi": row.get("doi", ""),
                "title": row.get("title", ""),
                "year": row.get("year", 2020),
                "citations": c,
                "median_z": round(med_z, 2),
                "min_z": round(min_z, 2),
                "is_atypical": min_z < -0.5,
                "is_conventional": med_z > 0.0,
                "is_hit": is_hit,
            }
        )

    art_df = pd.DataFrame(art_rows)
    atypical_mask = art_df["is_atypical"] & art_df["is_conventional"]
    hit_rate_atypical = (
        float(art_df.loc[atypical_mask, "is_hit"].mean() * 100) if atypical_mask.sum() > 0 else 0.0
    )
    hit_rate_baseline = float(art_df["is_hit"].mean() * 100) if len(art_df) > 0 else 0.0

    return {
        "valid": True,
        "articles_df": art_df,
        "atypical_pairs": atypical_pairs_df,
        "hit_rate_high_atypical": round(hit_rate_atypical, 1),
        "hit_rate_baseline": round(hit_rate_baseline, 1),
        "cite_p95_threshold": int(cite_p95),
    }


def venue_semantic_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    dois: list[str],
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Group publication venues into semantic clusters based on average R^384 embeddings.

    Uncovers ontological families of journals and conferences sharing conceptual focus.
    """
    from sklearn.cluster import KMeans

    if df.empty or len(embeddings) == 0 or len(dois) == 0 or "venue" not in df.columns:
        return pd.DataFrame()

    doi_to_idx = {d: i for i, d in enumerate(dois)}
    valid = df.dropna(subset=["venue"]).copy()
    valid["emb_idx"] = valid["doi"].map(doi_to_idx)
    valid = valid.dropna(subset=["emb_idx"])
    if len(valid) < n_clusters:
        return pd.DataFrame()

    valid["emb_idx"] = valid["emb_idx"].astype(int)

    # Compute mean embedding per venue
    venue_vectors = {}
    venue_counts = {}
    venue_cites = {}

    for venue, grp in valid.groupby("venue"):
        if len(grp) < 2:
            continue
        idxs = grp["emb_idx"].to_numpy()
        vecs = embeddings[idxs]
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normed = vecs / norm
        mean_vec = normed.mean(axis=0)
        c_norm = np.linalg.norm(mean_vec)
        venue_vectors[venue] = mean_vec / (c_norm if c_norm > 0 else 1.0)
        venue_counts[venue] = len(grp)
        venue_cites[venue] = float(
            pd.to_numeric(grp.get("citation_count"), errors="coerce").fillna(0).mean()
        )

    venues = list(venue_vectors.keys())
    if len(venues) < n_clusters:
        return pd.DataFrame()

    X = np.array([venue_vectors[v] for v in venues])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)

    cluster_labels = {
        0: "Redes Elétricas & Operação",
        1: "Transição Energética & Renováveis",
        2: "Sistemas Computacionais & IA",
        3: "Engenharia de Potência & Confiabilidade",
    }

    rows = []
    for i, v in enumerate(venues):
        c_id = int(kmeans.labels_[i])
        rows.append(
            {
                "venue": v,
                "cluster_id": c_id,
                "cluster_name": cluster_labels.get(c_id, f"Cluster {c_id + 1}"),
                "articles": venue_counts[v],
                "mean_citations": round(venue_cites[v], 1),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["cluster_id", "articles"], ascending=[True, False])
