"""Citation/reference-count backfill for records the pipeline can't derive
these numbers for on its own (Elsevier bib entries carry neither field).

`data/enrichment_cache.json` is a hand-maintained `{doi: {citation_count,
reference_count}}` cache (not produced by any code in this repo -- it was
built out-of-band). Bronze is a delete-and-rebuild from Raw
(`transform/bronze_articles.py::build_bronze_articles`), and
`_build_elsevier_records` writes `citation_count=None`/`reference_count=None`
literally, so every bronze run starts with no Elsevier counts at all. This
module makes the backfill a first-class, idempotent step of that build
(`_enrich_citation_counts`, run after the rebuild, filling only nulls) instead
of a one-off manual patch that the next run would erase.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import DATA_DIR
from db.bronze_models import EnrichmentObservation
from ingest.doi import normalize_doi

ENRICHMENT_CACHE_PATH = DATA_DIR / "enrichment_cache.json"


class EnrichmentCacheError(ValueError):
    """`enrichment_cache.json` exists but cannot be read as the cache it claims to be."""


def load_enrichment_cache(path: Path | None = None) -> dict[str, dict[str, int | None]]:
    """Load the DOI -> {citation_count, reference_count} cache.

    Tolerates a *missing* file -- `data/` is gitignored and won't exist on a
    fresh checkout -- by returning an empty dict rather than raising. A file
    that exists but cannot be read or parsed is a different situation:
    returning `{}` for it too made a corrupt cache indistinguishable from no
    cache, and bronze was rebuilt without a single Elsevier count and no
    message. That raises `EnrichmentCacheError`, which the bronze stage does
    not catch, so the run fails and says why.
    """
    cache_path = path or ENRICHMENT_CACHE_PATH
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except (OSError, ValueError) as exc:
        # `json.JSONDecodeError` and `UnicodeDecodeError` are both ValueErrors.
        raise EnrichmentCacheError(
            f"enrichment cache {cache_path} exists but could not be read: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise EnrichmentCacheError(
            f"enrichment cache {cache_path} must be a JSON object keyed by DOI, "
            f"not {type(raw).__name__}"
        )

    normalized: dict[str, dict[str, int | None]] = {}
    for doi, values in raw.items():
        key = normalize_doi(doi)
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
