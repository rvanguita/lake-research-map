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

import hashlib
import logging
import warnings
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from lake_research_map.config import CAPES_QUALIS_XLSX
from lake_research_map.transform.silver_articles import normalize_title

logger = logging.getLogger(__name__)

QUALIS_AREA = "ENGENHARIAS IV"
MATCH_THRESHOLD = 85.0

NOT_CLASSIFIED = "Not classified"

# Best to worst; unclassified always last. Shared by every chart/table that
# ranks or orders by classification, so "A1 first" only needs to be defined once.
ESTRATO_ORDER = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C", NOT_CLASSIFIED)


def _cache_path(xlsx_path: Path) -> Path:
    """Parquet cache path for `xlsx_path`, keyed on its mtime, size and area.

    The key is in the filename, so replacing the CAPES export simply produces
    a different path -- a stale cache can never be read as if it were current.
    """
    stat = xlsx_path.stat()
    key = hashlib.sha256(f"{stat.st_mtime_ns}:{stat.st_size}:{QUALIS_AREA}".encode()).hexdigest()[
        :16
    ]
    return xlsx_path.with_name(f".qualis_{key}.parquet")


def load_qualis_reference(path=None) -> pd.DataFrame:
    """Read the official CAPES export, filtered to `QUALIS_AREA`.

    Returns columns `["issn", "titulo", "estrato"]`. Thin I/O wrapper over the
    xlsx file -- not unit tested, same as this project's other raw-file loaders.

    The xlsx holds every evaluation area (~171k rows) and openpyxl takes ~3.4s
    to parse it, all to keep the few thousand ENGENHARIAS IV rows. That subset
    is cached next to the source as parquet, which reloads in milliseconds on
    later dashboard starts.
    """
    xlsx_path = Path(path or CAPES_QUALIS_XLSX)
    cache = _cache_path(xlsx_path)
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            # A truncated/unreadable cache must never be fatal -- fall back to
            # the xlsx, which then overwrites it.
            logger.debug("load_qualis_reference: unusable cache %s", cache, exc_info=True)

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
        df = pd.read_excel(xlsx_path, sheet_name="RelatorioQualis")
    df.columns = [c.strip() for c in df.columns]
    df["Área de Avaliação"] = df["Área de Avaliação"].astype(str).str.strip()
    df = df[df["Área de Avaliação"] == QUALIS_AREA]
    reference = df.rename(columns={"ISSN": "issn", "Título": "titulo", "Estrato": "estrato"})[
        ["issn", "titulo", "estrato"]
    ].reset_index(drop=True)

    try:
        reference.to_parquet(cache, index=False)
        for stale in cache.parent.glob(".qualis_*.parquet"):
            if stale != cache:
                stale.unlink()
    except Exception:
        # A read-only data dir just means no cache -- never fail the load.
        logger.debug("load_qualis_reference: could not write cache %s", cache, exc_info=True)
    return reference


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
