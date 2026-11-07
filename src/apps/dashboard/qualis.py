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
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

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
