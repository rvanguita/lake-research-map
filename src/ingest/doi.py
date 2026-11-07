"""The one DOI normalizer: every cross-source join, dedup and lookup goes through it.

DOI is the only reliable key across the two publishers (see CLAUDE.md "The two
sources are not interchangeable"), but they spell it differently: IEEE exports
a bare DOI (`10.1109/TPWRS.2024.3418651`), ScienceDirect a resolver URL
(`https://doi.org/10.1016/j.ijepes.2020.106042`), and older records or other
providers use `http://dx.doi.org/`. DOIs are case-insensitive by specification,
and both spellings occur for the same work. Comparing anything but the
normalized form lets the same paper survive deduplication twice, or miss its
enrichment row.

This used to be two hand-synchronised copies (bronze building and the
enrichment lookups), duplicated to dodge a circular import; a module with no
project imports removes the reason for the copy. Keep it that way: it must stay
importable from `ingest/`, `transform/`, `quality.py` and the dashboard alike.
"""

from __future__ import annotations

import re

_RESOLVER_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", flags=re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    """Canonical comparison form of a DOI, or None when there is none.

    Strips surrounding whitespace and a `http(s)://(dx.)doi.org/` resolver
    prefix (any case), then casefolds. Empty or whitespace-only input -- and a
    bare prefix with nothing after it -- returns None, so callers can treat
    "no DOI" as one value (silver counts those as `skipped_no_doi`).
    """
    if not doi:
        return None
    doi = _RESOLVER_PREFIX.sub("", doi.strip())
    return doi.strip().casefold() or None
