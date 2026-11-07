"""CAPES/Qualis journal-classification lookup for the corpus's venues.

The reference data (`config.CAPES_QUALIS_XLSX`) is the official CAPES export for the
2017–2020 evaluation cycle — the last journal-level Qualis grading (CAPES moves to an
article-level evaluation for 2025-2028). A journal's stratum is evaluation-area
specific; this corpus is electric-power/distribution-planning, so it's matched
against `ENGENHARIAS IV` only — a different evaluation area can grade the same journal
differently.

Venue strings in the corpus do not exactly match the reference titles (e.g. this
corpus has "Renewable and Sustainable Energy Reviews", the reference has "RENEWABLE
& SUSTAINABLE ENERGY REVIEWS"; ScienceDirect/IEEE Xplore titles also carry "(Print)"/
"(Online)" suffixes the reference sometimes does and sometimes doesn't), so venues are
matched by fuzzy title similarity -- the same approach `transform.silver_articles`
already uses for PDF-to-title matching, reusing its `normalize_title`.
"""

from __future__ import annotations

import logging
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from apps.dashboard.analytics.counts import source_counts_by, valid_years
from config import CAPES_QUALIS_XLSX
from transform.silver_articles import normalize_title

logger = logging.getLogger(__name__)

QUALIS_AREA = "ENGENHARIAS IV"
MATCH_THRESHOLD = 85.0

NOT_CLASSIFIED = "Not classified"

# Best to worst; unclassified always last. Shared by every chart/table that
# ranks or orders by classification, so "A1 first" only needs to be defined once.
# Shown by every page whose Qualis view has nothing to match against -- the
# dashboard container ships without `data/`, so this is an expected state.
SOURCE_UNAVAILABLE_MESSAGE = (
    "CAPES/Qualis source unavailable: the official classification workbook "
    "(`config.CAPES_QUALIS_XLSX`) was not found, so no venue can be classified."
)

ESTRATO_ORDER = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C", NOT_CLASSIFIED)


def _normalise_column_name(value: object) -> str:
    """Return a comparison key for Portuguese/English workbook headers."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "".join(char for char in text.casefold() if char.isalnum())


def _empty_reference() -> pd.DataFrame:
    return pd.DataFrame(columns=["issn", "titulo", "estrato"])


def load_qualis_reference(path=None) -> pd.DataFrame:
    """Read the official CAPES export, filtered to `QUALIS_AREA`.

    Returns columns `["issn", "titulo", "estrato"]`, or an empty frame with
    those columns when the workbook is absent. Pure and read-only: no
    Streamlit import and no file written anywhere. `data/` is read-only input
    and is not shipped in the dashboard image (`.dockerignore`), so a missing
    workbook is an expected state the pages report, never an exception.

    The xlsx holds every evaluation area (~171k rows) and openpyxl takes
    seconds to parse it. Caching that parse is `loaders._qualis_reference`'s
    job (`@st.cache_data`, the dashboard's only cache layer); an earlier
    parquet cache written next to the source was removed because the
    dashboard must never write into the data directory.
    """
    xlsx_path = Path(path or CAPES_QUALIS_XLSX)
    if not xlsx_path.is_file():
        logger.info("load_qualis_reference: CAPES/Qualis workbook not found at %s", xlsx_path)
        return _empty_reference()

    with warnings.catch_warnings():
        # The source workbook has no explicit default cell style; openpyxl
        # substitutes its own and warns about it, but this never affects the
        # data actually read -- narrowly silenced so any other, genuinely
        # actionable warning from this call still surfaces.
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style, apply openpyxl's default",
            category=UserWarning,
        )
        try:
            df = pd.read_excel(xlsx_path, sheet_name="RelatorioQualis")
        except ValueError:
            # CAPES has published the same report with both `RelatorioQualis`
            # and `Relatorio Qualis` sheet names. The first sheet is the safest
            # fallback for future exports with another harmless spelling.
            df = pd.read_excel(xlsx_path, sheet_name=0)

    # The checked-in/reference download is already scoped to the relevant area
    # and therefore has no `Área de Avaliação` column. Newer full exports do
    # include it, so filter only when that optional column is present.
    column_map = {_normalise_column_name(column): column for column in df.columns}
    area_column = column_map.get("areadeavaliacao")
    if area_column is not None:
        df = df[df[area_column].astype(str).str.strip() == QUALIS_AREA]

    required = {
        "issn": column_map.get("issn"),
        "titulo": column_map.get("titulo"),
        "estrato": column_map.get("estrato"),
    }
    if any(column is None for column in required.values()):
        logger.warning(
            "load_qualis_reference: workbook %s is missing required columns; found %s",
            xlsx_path,
            list(df.columns),
        )
        return _empty_reference()
    return df.rename(columns={source: target for target, source in required.items()})[
        ["issn", "titulo", "estrato"]
    ].reset_index(drop=True)


def match_venues_to_qualis(
    venues: list[str], qualis_df: pd.DataFrame, threshold: float = MATCH_THRESHOLD
) -> pd.DataFrame:
    """Fuzzy-match each distinct venue to a Qualis title, or leave it unclassified.

    Pure function (no file I/O) -- `qualis_df` must have a `titulo` column (as
    returned by `load_qualis_reference`) and, when matched, an `estrato` column.
    Returns one row per input venue: `["venue", "matched_title", "estrato", "score"]`.
    Below `threshold`, `matched_title`/`estrato` are `None` rather than a guessed
    grade -- an unmatched venue must read as "not classified," never as a wrong one.
    """
    choices = {idx: normalize_title(title) for idx, title in qualis_df["titulo"].items() if title}
    rows = []
    for venue in venues:
        normalized = normalize_title(venue)
        match = (
            process.extractOne(
                normalized, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
            )
            if normalized and choices
            else None
        )
        if match is None:
            rows.append({"venue": venue, "matched_title": None, "estrato": None, "score": None})
            continue
        _, score, ref_idx = match
        rows.append(
            {
                "venue": venue,
                "matched_title": qualis_df.loc[ref_idx, "titulo"],
                "estrato": qualis_df.loc[ref_idx, "estrato"],
                "score": score,
            }
        )
    return pd.DataFrame(rows, columns=["venue", "matched_title", "estrato", "score"])


# The tiers the "A1–A3 subset" view groups together.
TOP_TIERS = ("A1", "A2", "A3")


@dataclass(frozen=True)
class QualisStrata:
    """Every Qualis view's inputs, derived once from one venue match table.

    `match_table`: one row per venue in scope (`venue`, `matched_title`,
    `estrato`, `score` as display text, `articles`), best tier first and the
    most-published venue first within a tier. `with_estrato`: the articles with
    their venue's `estrato` (NaN for a venue the match table lacks).
    `totals`: articles per stratum present, in `ESTRATO_ORDER`. `order`: those
    strata, best first.
    """

    match_table: pd.DataFrame
    with_estrato: pd.DataFrame
    totals: pd.Series
    order: list[str]


def qualis_strata(articles_df: pd.DataFrame, venue_matches: pd.DataFrame) -> QualisStrata:
    """Join the articles in scope to the full-corpus venue match table.

    `venue_matches` is the table matched once over the *unfiltered* corpus
    (`loaders.all_venue_qualis_map`) and narrowed here to the venues in scope,
    rather than re-matching a filtered venue tuple -- which would re-run the
    rapidfuzz pass (and, transitively, the ~9 s workbook parse) on every global
    filter change, since the venue tuple is the cache key. Missing optional
    columns (`matched_title`, `score`) are tolerated.
    """
    venues = set(articles_df["venue"].dropna().unique()) if "venue" in articles_df else set()
    match_df = venue_matches.copy()
    for column in ("venue", "matched_title", "estrato", "score"):
        if column not in match_df.columns:
            match_df[column] = pd.Series(dtype=object)
    match_df = match_df[match_df["venue"].isin(venues)].reset_index(drop=True)
    match_df["estrato"] = match_df["estrato"].fillna(NOT_CLASSIFIED)
    # A raw None in a string column renders as literal "undefined" in Streamlit's
    # dataframe grid (pandas' Arrow-backed string dtype stores it as a genuine
    # null, not NaN) -- always give it an explicit placeholder instead.
    match_df["matched_title"] = match_df["matched_title"].fillna("—")
    match_df["score"] = match_df["score"].apply(lambda s: f"{s:.0f}" if pd.notna(s) else "—")

    counts = articles_df["venue"].value_counts() if "venue" in articles_df else pd.Series(dtype=int)
    match_df["articles"] = match_df["venue"].map(counts).fillna(0).astype(int)
    tier_rank = {estrato: rank for rank, estrato in enumerate(ESTRATO_ORDER)}
    match_df = (
        match_df.assign(_tier_rank=match_df["estrato"].map(tier_rank))
        .sort_values(["_tier_rank", "articles"], ascending=[True, False])
        .drop(columns=["_tier_rank"])
        .reset_index(drop=True)
    )[["venue", "matched_title", "estrato", "score", "articles"]]

    venue_to_estrato = dict(zip(match_df["venue"], match_df["estrato"], strict=True))
    with_estrato = articles_df.copy()
    with_estrato["estrato"] = (
        with_estrato["venue"].map(venue_to_estrato) if "venue" in with_estrato else pd.NA
    )
    present = set(with_estrato["estrato"].dropna().unique())
    totals = with_estrato["estrato"].value_counts().reindex(list(ESTRATO_ORDER), fill_value=0)
    totals = totals[totals.index.isin(present)]
    return QualisStrata(match_df, with_estrato, totals, list(totals.index))


def strata_volume_by_year(
    with_estrato: pd.DataFrame, *, through: str = "B2"
) -> tuple[pd.DataFrame, list[str]]:
    """Articles per (year, stratum) for the strata from A1 down to `through`.

    `ESTRATO_ORDER` is best-to-worst with the unclassified bucket last, so
    "through B2" is everything from A1 to B2 in that ranking. Returns the
    `year`/`estrato`/`count` frame and the strata present, best first.
    """
    allowed = list(ESTRATO_ORDER[: ESTRATO_ORDER.index(through) + 1])
    years_df = with_estrato.copy()
    years_df["year"] = valid_years(years_df)
    years_df = years_df.dropna(subset=["year"]).astype({"year": int})
    years_df = years_df[years_df["estrato"].isin(allowed)]
    if years_df.empty:
        return pd.DataFrame(columns=["year", "estrato", "count"]), []
    by_year = (
        years_df.groupby(["year", "estrato"]).size().reset_index(name="count").sort_values("year")
    )
    present = set(years_df["estrato"].unique())
    return by_year.reset_index(drop=True), [e for e in allowed if e in present]


def top_tier_summary(
    with_estrato: pd.DataFrame, *, tiers: tuple[str, ...] = TOP_TIERS, top_n: int = 15
) -> dict[str, pd.DataFrame]:
    """The "A1–A3 subset" view's three tables, or all-empty frames.

    `by_year`: `source_counts_by` over the valid years. `top_venues`: the
    `top_n` most-published venues with `venue`, `articles`, `estrato`, `ieee`,
    `elsevier` (the source split goes in the hover of the tier-coloured
    ranking). `by_source_tier`: articles per (`source`, `estrato`), empty when
    the frame has no `source`.
    """
    subset = with_estrato[with_estrato["estrato"].isin(tiers)]
    years_df = subset.copy()
    years_df["year"] = valid_years(years_df)
    years_df = years_df.dropna(subset=["year"]).astype({"year": int})
    by_year = (
        source_counts_by(years_df, "year").sort_values("year").reset_index(drop=True)
        if not years_df.empty
        else pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])
    )

    split = source_counts_by(subset, "venue") if not subset.empty else None
    top = subset["venue"].value_counts().head(top_n)
    top_venues = pd.DataFrame({"venue": top.index, "articles": top.to_numpy()})
    tier_of = subset.drop_duplicates("venue").set_index("venue")["estrato"]
    top_venues["estrato"] = top_venues["venue"].map(tier_of)
    for source in ("ieee", "elsevier"):
        top_venues[source] = (
            top_venues["venue"].map(split.set_index("venue")[source]).fillna(0).astype(int)
            if split is not None and source in split.columns
            else 0
        )

    by_source_tier = (
        subset.groupby(["source", "estrato"]).size().reset_index(name="count")
        if "source" in subset.columns and not subset.empty
        else pd.DataFrame(columns=["source", "estrato", "count"])
    )
    return {"by_year": by_year, "top_venues": top_venues, "by_source_tier": by_source_tier}
