"""Citation/reference-count backfill for records the pipeline can't derive
these numbers for on its own (Elsevier bib entries carry neither field).

`data/enrichment_cache.json` is a hand-maintained `{doi: {citation_count,
reference_count}}` cache (not produced by any code in this repo -- it was
built out-of-band). Without this module, re-running `--stage bronze` wipes
every citation/reference count the cache supplied, because
`_build_elsevier_records` writes `citation_count=None`/`reference_count=None`
literally and `_upsert` overwrites field by field. This module makes that
backfill a first-class, idempotent step of the bronze build instead of a
one-off manual patch.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.config import DATA_DIR
from lake_research_map.db.bronze_models import EnrichmentObservation

ENRICHMENT_CACHE_PATH = DATA_DIR / "enrichment_cache.json"


def _normalize_doi(doi: str | None) -> str | None:
    """Same normalization as transform.bronze_articles.normalize_doi.

    Duplicated (rather than imported) to avoid a circular import --
    bronze_articles imports this module to run the enrichment step.
    """
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.strip().casefold() or None


def load_enrichment_cache(path: Path | None = None) -> dict[str, dict[str, int | None]]:
    """Load the DOI -> {citation_count, reference_count} cache.

    Tolerates a missing file -- `data/` is gitignored and won't exist on a
    fresh checkout -- by returning an empty dict rather than raising.
    """
    cache_path = path or ENRICHMENT_CACHE_PATH
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, int | None]] = {}
    for doi, values in raw.items():
        key = _normalize_doi(doi)
        if not key or not isinstance(values, dict):
            continue
        normalized[key] = {
            "citation_count": values.get("citation_count"),
            "reference_count": values.get("reference_count"),
        }
    return normalized


def load_enrichment_observations(
    session: Session,
    *,
    as_of: datetime | None = None,
) -> dict[str, dict[str, int | None]]:
    """Select the latest successful provider observation per DOI as of a cutoff."""
    query = select(EnrichmentObservation).where(EnrichmentObservation.status == "success")
    if as_of is not None:
        query = query.where(EnrichmentObservation.observed_at <= as_of)
    rows = session.scalars(
        query.order_by(
            EnrichmentObservation.doi,
            EnrichmentObservation.observed_at.desc(),
            EnrichmentObservation.id.desc(),
        )
    ).all()
    selected: dict[str, dict[str, int | None]] = {}
    for row in rows:
        selected.setdefault(
            row.doi,
            {
                "citation_count": row.citation_count,
                "reference_count": row.reference_count,
            },
        )
    return selected


def merge_enrichment(
    cache: dict[str, dict[str, int | None]],
    observations: dict[str, dict[str, int | None]],
) -> dict[str, dict[str, int | None]]:
    """Overlay provider observations on the hand-built cache, field by field.

    A plain ``dict.update`` replaced a DOI's whole cache entry, so an OpenAlex
    work with an empty `referenced_works` turned a curated `reference_count`
    of 42 into 0. That zero is often OpenAlex not holding the list rather than
    the work citing nothing: of 324 works it reported at zero, 81 have
    references deposited in Crossref. An observation therefore replaces a
    field only when it carries a value, and never lowers a positive count to
    zero.
    """
    merged = {doi: dict(values) for doi, values in cache.items()}
    for doi, observed in observations.items():
        target = merged.setdefault(doi, {"citation_count": None, "reference_count": None})
        for field in ("citation_count", "reference_count"):
            value = observed.get(field)
            if value is None:
                continue
            if value == 0 and (target.get(field) or 0) > 0:
                continue
            target[field] = value
    return merged
