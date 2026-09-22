"""Pure pandas aggregations shared by every dashboard page.

No `streamlit` import here on purpose -- these functions are plain data
transforms and can be exercised/cached independently of the UI layer. Pages
call these instead of open-coding a `groupby`/`explode`, so the same year
window, the same "count by source" shape, and the same author-name
normalization are used everywhere.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

from lake_research_map.transform.publication_categories import PUBLICATION_CATEGORIES

logger = logging.getLogger(__name__)

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
    "growing" / "stable" / "falling" against a small fixed threshold (0.15
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
                    "trend": "insufficient data",
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
            trend = "growing"
        elif slope < -0.15:
            trend = "falling"
        else:
            trend = "stable"
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


# --- Advanced Statistics, Bibliometrics & Complex Networks -----------------


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


def network_null_model_diagnostics(graph, *, n_simulations: int = 100, seed: int = 42) -> dict:
    """Compare clustering with degree-preserving rewired null networks."""
    import networkx as nx

    if graph.number_of_nodes() < 4 or graph.number_of_edges() < 3:
        return {"valid": False, "reason": "insufficient_network"}
    observed = float(nx.average_clustering(graph))
    null_values: list[float] = []
    null_assortativity: list[float] = []
    rng = np.random.default_rng(seed)
    swaps = max(graph.number_of_edges() * 5, 1)
    for _ in range(max(1, n_simulations)):
        candidate = nx.Graph(graph)
        try:
            nx.double_edge_swap(
                candidate,
                nswap=swaps,
                max_tries=max(swaps * 20, 100),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            continue
        null_values.append(float(nx.average_clustering(candidate)))
        null_assortativity.append(float(nx.degree_assortativity_coefficient(candidate)))
    if not null_values:
        return {"valid": False, "reason": "rewiring_failed"}
    null_array = np.asarray(null_values)
    null_std = float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0

    # Assortativity rides along on the same rewired ensemble: the null is
    # already built, and a bare assortativity coefficient says nothing without
    # one -- degree sequence alone forces some of it.
    observed_assortativity = float(nx.degree_assortativity_coefficient(graph))
    assort_array = np.asarray([value for value in null_assortativity if np.isfinite(value)])
    assort_std = float(assort_array.std(ddof=1)) if len(assort_array) > 1 else 0.0
    assort_mean = float(assort_array.mean()) if len(assort_array) else float("nan")

    return {
        "valid": True,
        "observed_clustering": observed,
        "null_mean": float(null_array.mean()),
        "null_std": null_std,
        "z_score": (observed - float(null_array.mean())) / null_std if null_std > 0 else None,
        "empirical_p_value": float((1 + np.sum(null_array >= observed)) / (len(null_array) + 1)),
        "simulations": len(null_array),
        "observed_assortativity": observed_assortativity
        if np.isfinite(observed_assortativity)
        else None,
        "assortativity_null_mean": assort_mean if np.isfinite(assort_mean) else None,
        "assortativity_z_score": (
            (observed_assortativity - assort_mean) / assort_std
            if assort_std > 0 and np.isfinite(observed_assortativity) and np.isfinite(assort_mean)
            else None
        ),
        **_robustness_under_removal(graph),
    }


def _robustness_under_removal(graph, *, fraction: float = 0.1, seed: int = 42) -> dict:
    """How much of the network survives losing its hubs versus random nodes.

    A collaboration network held together by a few hub authors fragments under
    targeted removal while barely noticing random loss. The gap between the two
    is the structural claim; either number alone is just a graph size.
    """
    import networkx as nx

    total = graph.number_of_nodes()
    remove = max(1, int(round(total * fraction)))
    if total - remove < 2:
        return {"robustness_targeted": None, "robustness_random": None, "robustness_removed": 0}

    def giant_share(candidate) -> float:
        components = list(nx.connected_components(candidate))
        return float(max(len(c) for c in components) / total) if components else 0.0

    hubs = [node for node, _ in sorted(graph.degree, key=lambda kv: -kv[1])[:remove]]
    targeted = graph.copy()
    targeted.remove_nodes_from(hubs)

    rng = np.random.default_rng(seed)
    random_shares = []
    for _ in range(20):
        victim = graph.copy()
        victim.remove_nodes_from(rng.choice(list(graph.nodes), size=remove, replace=False).tolist())
        random_shares.append(giant_share(victim))

    return {
        "robustness_targeted": giant_share(targeted),
        "robustness_random": float(np.mean(random_shares)),
        "robustness_removed": remove,
    }


def periodized_collaboration_ties(author_rows: pd.DataFrame, *, n_periods: int = 3) -> pd.DataFrame:
    """Split coauthor ties into periods and count new versus repeated ones.

    The existing recurrent-edge count is static: it says how many pairs ever
    published twice, which cannot distinguish a field that keeps recruiting new
    collaborators from one that has closed into fixed teams. Splitting by the
    year a tie first appears is what makes that visible.
    """
    required = {"doi", "year"}
    author_column = "author_display" if "author_display" in author_rows.columns else "author"
    if author_rows.empty or not required.issubset(author_rows.columns):
        return pd.DataFrame()

    rows = author_rows.dropna(subset=["doi", "year", author_column]).copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce")
    rows = rows.dropna(subset=["year"])
    if rows.empty:
        return pd.DataFrame()

    # Every unordered author pair on each paper, with that paper's year.
    ties: dict[tuple[str, str], list[int]] = {}
    for (_doi, year), group in rows.groupby(["doi", "year"]):
        names = sorted({str(name) for name in group[author_column]})
        if len(names) < 2:
            continue
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                ties.setdefault((left, right), []).append(int(year))
    if not ties:
        return pd.DataFrame()

    years = np.sort(rows["year"].unique())
    if len(years) < n_periods:
        n_periods = max(1, len(years))
    edges = np.array_split(years, n_periods)

    records = []
    for block in edges:
        if not len(block):
            continue
        low, high = int(block.min()), int(block.max())
        new_ties = 0
        repeated = 0
        for appearances in ties.values():
            within = [y for y in appearances if low <= y <= high]
            if not within:
                continue
            # "New" means the pair's first-ever collaboration falls in this
            # period; anything earlier makes it a returning partnership.
            if min(appearances) >= low:
                new_ties += 1
            else:
                repeated += 1
        total = new_ties + repeated
        records.append(
            {
                "period": f"{low}–{high}",
                "new_ties": new_ties,
                "repeated_ties": repeated,
                "total_ties": total,
                "new_share": float(new_ties / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(records)


def _knn_overlap(matrix: np.ndarray, projection: np.ndarray, k: int) -> float:
    """Share of each point's k nearest neighbours that survive the projection.

    The most legible of the three neighbourhood measures: 0.7 means seven of
    every ten neighbours a point has in the clustering space are still its
    neighbours on screen.
    """
    from sklearn.neighbors import NearestNeighbors

    if k < 1 or len(matrix) <= k:
        return float("nan")
    high = NearestNeighbors(n_neighbors=k + 1).fit(matrix).kneighbors(return_distance=False)
    low = NearestNeighbors(n_neighbors=k + 1).fit(projection).kneighbors(return_distance=False)
    shared = [len(set(a).intersection(b)) for a, b in zip(high, low, strict=True)]
    return float(np.mean(shared) / k)


def semantic_stability_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    projection: np.ndarray,
    *,
    n_bootstrap: int = 30,
    seed: int = 42,
) -> dict:
    """Measure cluster seed/subsample stability and 2D neighborhood preservation."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import adjusted_rand_score

    matrix = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    projection = np.asarray(projection, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 10 or len(np.unique(labels)) < 2:
        return {"valid": False, "reason": "insufficient_semantic_population"}
    if projection.shape != (len(matrix), 2):
        return {"valid": False, "reason": "invalid_projection"}
    clusters = len(np.unique(labels))
    sample_size = max(clusters * 2, int(np.ceil(len(matrix) * 0.8)))
    sample_size = min(sample_size, len(matrix))
    rng = np.random.default_rng(seed)
    ari_values: list[float] = []
    for run in range(max(1, n_bootstrap)):
        indices = np.sort(rng.choice(len(matrix), size=sample_size, replace=False))
        predicted = KMeans(n_clusters=clusters, random_state=seed + run, n_init=10).fit_predict(
            matrix[indices]
        )
        ari_values.append(float(adjusted_rand_score(labels[indices], predicted)))
    neighbors = min(10, max(1, (len(matrix) - 1) // 2))
    return {
        "valid": True,
        "bootstrap_ari_mean": float(np.mean(ari_values)),
        "bootstrap_ari_min": float(np.min(ari_values)),
        "projection_trustworthiness": float(
            trustworthiness(matrix, projection, n_neighbors=neighbors)
        ),
        # Trustworthiness only punishes neighbours the map *invents*. A
        # projection that tears a real cluster in two scores well on it while
        # being badly wrong, so continuity (the same measure with the spaces
        # swapped, which punishes neighbours the map *loses*) and the plain
        # kNN overlap are reported beside it.
        "projection_continuity": float(trustworthiness(projection, matrix, n_neighbors=neighbors)),
        "knn_overlap": _knn_overlap(matrix, projection, neighbors),
        "neighbors": neighbors,
        "n_bootstrap": len(ari_values),
        "clusters": clusters,
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


def citation_determinants_glm(df: pd.DataFrame, *, observation_year: int = 2026) -> dict:
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
            "Rank (r)": ranks[:30].astype(int),
            "Term": words[:30],
            "Real Frequency (f)": freqs[:30].astype(int),
            "Ideal Zipf forecast": expected_ideal[:30].round(1),
            "Empirical Adjustment": (10 ** pred[:30]).round(1),
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
                        "Time": epoch_name,
                        "Theme": th,
                        "Characteristic Terms (c-TF-IDF)": ", ".join(top_kws),
                        "Articles": int((sub["theme_label"] == th).sum()),
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
                reasons.append("Typical / Consistent Pattern")
                continue
            r_list = []
            if cites[i] >= cite_p95 and years[i] >= 2018:
                r_list.append(f"Recently hyper-cited ({int(cites[i])} citations)")
            elif cites[i] >= cite_p95:
                r_list.append(f"Extreme citation count ({int(cites[i])} citations)")
            if author_counts[i] >= author_p98 and author_counts[i] >= 10:
                r_list.append(f"Mega-team ({int(author_counts[i])} authors)")
            if years[i] < 1990:
                r_list.append(f"Historical article ({int(years[i])})")
            if rel[i] < 0.25:
                r_list.append(f"Divergent semantic margin ({rel[i]:.2f})")
            if not r_list:
                r_list.append("Joint multidimensional atypicality")
            reasons.append("; ".join(r_list))

        res["anomaly_reason"] = reasons
    except Exception:
        res["anomaly_score"] = 0.0
        res["is_anomaly"] = False
        res["anomaly_reason"] = "Erro no ajuste"

    return res


# The regexes `optimization_methods_taxonomy` classifies with, hoisted to module
# scope so a sampler drawing strata for human review matches on exactly the same
# rule the dashboard displays. Re-deriving the match from the label text (which
# `evidence taxonomy` did at first) silently produced zero classes and asked
# reviewers to label a stratum the classifier does not recognise.
OPTIMIZATION_METHOD_PATTERNS: dict[str, re.Pattern[str]] = {
    "Multi-objective Optimization": re.compile(r"multi-objective|pareto"),
    "Genetic Algorithms (GA)": re.compile(r"genetic algorithm|\bga\b"),
    "Particle Swarm Optimization (PSO)": re.compile(r"particle swarm|\bpso\b"),
    "Mixed-Integer Linear Programming (MILP)": re.compile(r"milp|mixed-integer linear"),
    "Machine Learning & AI": re.compile(
        r"machine learning|deep learning|reinforcement learning|neural network"
    ),
    "Other Metaheuristics": re.compile(
        r"differential evolution|harmony search|simulated annealing|ant colony"
    ),
    "Stochastic Programming": re.compile(r"stochastic programming|scenario-based"),
    "Robust Optimization": re.compile(r"robust optimization|robust approach"),
    "Conical / Convex Relaxation (SOCP)": re.compile(
        r"second-order cone|conic|convex relaxation|socp"
    ),
}


def taxonomy_haystack(frame: pd.DataFrame) -> pd.Series:
    """Lowercased title + abstract, the exact text the taxonomy matches against.

    Keywords are deliberately excluded because the classifier excludes them;
    a sampler that searched a wider field would build strata the dashboard
    disagrees with.
    """
    return (
        frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("abstract", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.lower()


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

    opt_patterns = OPTIMIZATION_METHOD_PATTERNS

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
        # Counting rows needs no particular column: keying this on the `id`
        # surrogate coupled the taxonomy to a primary key it never uses,
        # and broke on any frame assembled without one.
        yearly = df[matched & (years >= 2005)].groupby("year").size()
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
        "IEEE 33-Bus (standard radial)": re.compile(r"33-bus|ieee 33|33 bus|33-node"),
        "IEEE 69-Bus": re.compile(r"69-bus|ieee 69|69 bus|69-node"),
        "Real Utility Networks": re.compile(
            r"real distribution|real-world|practical distribution|actual distribution|utility network"
        ),
        "Regional Systems (Brazil / Europe)": re.compile(
            r"brazilian|european|california|uk distribution|nordic"
        ),
        "IEEE 123-Bus / 119-Bus": re.compile(r"123-bus|ieee 123|119-bus|ieee 119"),
    }

    resource_patterns = {
        "Solar Generation (PV)": re.compile(r"photovoltaic|\bpv\b|solar"),
        "Storage / Batteries": re.compile(r"energy storage|battery|bess"),
        "Electric Vehicles (EV)": re.compile(r"electric vehicle|\bev\b|v2g|charging"),
        "Network reconfiguration": re.compile(r"reconfiguration|switching|switch"),
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


def objective_functions_taxonomy(df: pd.DataFrame) -> dict:
    """Analyze optimized objectives and multi-criteria formulations."""
    if df.empty:
        return {
            "summary_df": pd.DataFrame(),
            "co_matrix": pd.DataFrame(),
            "multi_obj_ratio": 0.0,
            "temporal_multiobj": pd.DataFrame(),
        }

    objs = {
        "Economic Costs": r"cost|capex|opex|investment|economic|capital expenditure",
        "Reliability (SAIDI/SAIFI/ENS)": r"reliability|saidi|saifi|ens|energy not supplied|interruption|outage|unserved",
        "Technical losses": r"power loss|energy loss|technical loss|transmission loss|loss reduction",
        "Voltage Profile": r"voltage profile|voltage deviation|voltage stability|power quality|voltage drop|voltage regulation",
        "Decarbonization / Emissions": r"emission|carbon|decarboniz|greenhouse|environmental|co2",
        "Resilience": r"resilience|extreme weather|disaster|blackout|hardening|restoration",
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
    """Analyze mathematical paradigms for handling uncertainty in distribution systems."""
    if df.empty:
        return {
            "paradigms_df": pd.DataFrame(),
            "cross_resources": pd.DataFrame(),
            "temporal_paradigms": pd.DataFrame(),
        }

    paradigms = {
        "Stochastic (Scenarios / Monte Carlo)": r"stochastic|scenario-based|monte carlo|sample average",
        "Robust Optimization (Min-Max)": r"robust optimization|robust approach|uncertainty set|worst-case",
        "Fuzzy Logic": r"fuzzy",
        "Distributionally Robust Optimization (DRO)": r"distributionally robust|wasserstein|ambiguity set",
        "Chance Constraint (Probabilistic)": r"chance-constrained|chance constraint|probabilistic constraint",
        "Deterministic (Fixed Case)": r"deterministic",
    }

    resources = {
        "Solar Generation (PV)": r"photovoltaic|\bpv\b|solar",
        "Batteries / Storage (BESS)": r"energy storage|battery|bess",
        "Electric Vehicles (EV)": r"electric vehicle|\bev\b|v2g|charging",
        "Demand / Load Uncertainty": r"load uncertainty|demand uncertainty|forecast error|load variation",
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
            yearly = df[filt].groupby(years[filt].astype(int)).size()
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
        "Multistage Dynamic Expansion": r"multi-stage|multistage|multi-year|sequential expansion|expansion planning|dynamic planning",
        "Co-Optimization Planning + Operation": r"co-optimi|planning and operation|representative days|representative periods|operational constraints|chronological",
        "Static Planning (Target Year)": r"static planning|single-stage|target year|snapshot",
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
        "GAMS / AMPL (Algebraic Modelers)": (r"gams|ampl", "Algebraic Modeler"),
        "MATLAB / Simulink": (r"matlab|simulink", "Scripting & Simulation"),
        "CPLEX (IBM)": (r"cplex", "Commercial Exact Solver"),
        "DIgSILENT PowerFactory": (r"digsilent|powerfactory", "Specialized Electric Simulator"),
        "Gurobi Optimizer": (r"gurobi", "Commercial Exact Solver"),
        "OpenDSS (EPRI)": (r"opendss|open dss", "Distribution Simulator"),
        "Python (Pyomo / Pandapower)": (r"python|pyomo|pandapower", "Scripting & Open Source"),
        "PSCAD / EMTP": (r"pscad|emtp", "Transient Simulator"),
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
        "Linear Programming / MILP (Exact)": r"milp|mixed-integer linear|\blp\b|linear programming",
        "Convex / Conical Relaxation (SOCP/SDP)": r"second-order cone|socp|semidefinite|convex relaxation|conic",
        "Non-Linear Prog (NLP / MINLP)": r"minlp|mixed-integer nonlinear|nonlinear programming|\bnlp\b|non-convex",
        "Metaheuristics (GA, PSO, DE, ACO)": r"genetic algorithm|particle swarm|\bpso\b|\bga\b|differential evolution|ant colony|harmony search|simulated annealing",
        "AI & Reinforcement Learning": r"machine learning|deep learning|reinforcement learning|neural network|q-learning",
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
            yearly = df[filt].groupby(years[filt].astype(int)).size()
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
                    "Technology / Concept": label,
                    "Burst's Beginning": int(all_years[index]),
                    "Peak year": int(all_years[peak_index]),
                    "Burst end": int(all_years[end_index]),
                    "Duration (Years)": int(end_index - index + 1),
                    "Intensity": round(float(max(strength, 0.0)), 2),
                    "Status": "Active" if is_active else "Ended",
                    "Articles": int(interval_counts.sum()),
                }
            )
            index = end_index + 1

    burst_df = pd.DataFrame(burst_rows)
    if not burst_df.empty:
        burst_df = burst_df.sort_values("Intensity", ascending=False).head(top_n)
        active_df = burst_df[burst_df["Status"] == "Active"].copy()
    else:
        active_df = pd.DataFrame()

    return {
        "burst_timeline": burst_df,
        "active_frontiers": active_df,
        "total_bursts": len(burst_df),
    }


def _max_f_over_breakpoints(values: np.ndarray) -> tuple[float, int | None]:
    """Largest mean-shift F statistic over every admissible breakpoint.

    Returns ``(f_stat, k)``; ``k`` is None when no split improves on the
    pooled mean.
    """
    total = len(values)
    tss = float(np.sum((values - np.mean(values)) ** 2))
    best_rss = float("inf")
    best_k = None
    for k in range(2, total - 2):
        pre, post = values[:k], values[k:]
        rss = float(np.sum((pre - np.mean(pre)) ** 2) + np.sum((post - np.mean(post)) ** 2))
        if rss < best_rss:
            best_rss, best_k = rss, k
    if best_k is None or best_rss <= 0 or tss <= best_rss:
        return 0.0, None
    df2 = total - 2
    return float(((tss - best_rss) / 1) / (best_rss / df2)), best_k


def detect_structural_breaks(
    series: pd.Series | np.ndarray, *, n_bootstrap: int = 200, seed: int = 42
) -> dict:
    """Detect structural breaks / changepoints in time series using Chow test and SSE minimization.

    Identifies historical regime shifts (inflection points) where the underlying mean
    or momentum fundamentally changed.

    The breakpoint is *searched*, so the F statistic is a maximum over every
    admissible split and its null distribution is not F: read against the F
    table, a pure-noise series looks significant far too often. The reported
    `p_value` is therefore an empirical one from permuting the series under a
    no-break null; `p_value_naive` keeps the old F-table value so the
    difference stays visible rather than silently changing.
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
            "p_value_naive": 1.0,
            "p_value_resolution": 1.0,
            "bootstrap_samples": 0,
            "pre_mean": float(np.mean(values)) if T > 0 else 0.0,
            "post_mean": float(np.mean(values)) if T > 0 else 0.0,
            "relative_jump_pct": 0.0,
        }

    f_stat, best_k = _max_f_over_breakpoints(values)

    if best_k is None:
        return {
            "has_break": False,
            "break_index": None,
            "break_year": None,
            "f_stat": 0.0,
            "p_value": 1.0,
            "p_value_naive": 1.0,
            "p_value_resolution": 1.0,
            "bootstrap_samples": 0,
            "pre_mean": float(np.mean(values)),
            "post_mean": float(np.mean(values)),
            "relative_jump_pct": 0.0,
        }

    # The F-table value, kept only for comparison: it assumes the breakpoint
    # was fixed in advance, which it was not.
    p_value_naive = float(stats.f.sf(f_stat, 1, T - 2))

    # Permuting the series destroys any ordering, so each replicate is drawn
    # from a no-break null while keeping the observed values and their spread.
    # The share of replicates whose own searched maximum reaches the observed
    # one is the p-value the search actually earns.
    rng = np.random.default_rng(seed)
    exceedances = 0
    draws = 0
    for _ in range(max(1, n_bootstrap)):
        null_f, null_k = _max_f_over_breakpoints(rng.permutation(values))
        if null_k is None:
            null_f = 0.0
        draws += 1
        if null_f >= f_stat:
            exceedances += 1
    p_value = float((1 + exceedances) / (draws + 1))
    # An empirical p-value cannot resolve below 1/(draws+1); reporting the
    # floor stops a reader treating "0.005" as a precise small number rather
    # than "as small as this many replicates can show".
    p_value_resolution = float(1.0 / (draws + 1))

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
        "p_value_naive": round(p_value_naive, 4),
        "p_value_resolution": round(p_value_resolution, 5),
        "bootstrap_samples": draws,
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
                    "Term 1": k1,
                    "Term 2": k2,
                    "Real Co-occurrence": count,
                    "Expected": round(expected, 1),
                    "Z-score (atypicality)": round(z, 2),
                }
            )

    pair_rows.sort(key=lambda x: x["Z-score (atypicality)"])
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
