"""Parse every configured publisher search-provenance report.

These files are not tabular CSVs. Each is free text recording the query,
filters, year range, and full search URL that produced one export batch. The
loader extracts those fields with regexes but retains the complete text. Rows
are keyed by source file, not publisher, because one publisher can have many
separately reproducible searches.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.config import SEARCH_CONFIG_PATHS, relative_path
from lake_research_map.db.raw_models import Config
from lake_research_map.ingest.hashing import record_source_file

_YEAR_RANGE_RE = re.compile(r"year\s+(\d{4})\s*-\s*(\d{4})", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_QUOTED_QUERY_RE = re.compile(r'(".*?"(?:\s*OR\s*".*?")*)', re.IGNORECASE)


def _parse_config_text(raw_text: str) -> dict:
    query_match = _QUOTED_QUERY_RE.search(raw_text)
    query_string = query_match.group(1).strip() if query_match else None

    year_match = _YEAR_RANGE_RE.search(raw_text)
    year_range = f"{year_match.group(1)}-{year_match.group(2)}" if year_match else None

    url_match = _URL_RE.search(raw_text)
    search_url = url_match.group(0).strip() if url_match else None

    # Everything that isn't the query line, the year line, or the URL line is
    # treated as filter/provenance description.
    filter_lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
        and (query_string is None or query_string not in line)
        and not _YEAR_RANGE_RE.search(line)
        and not _URL_RE.search(line)
    ]
    filters = "\n".join(filter_lines) or None

    return {
        "query_string": query_string,
        "filters": filters,
        "year_range": year_range,
        "search_url": search_url,
    }


def load_configs(session: Session) -> int:
    """Parse configured reports into ``lit_config`` and return rows written."""
    written = 0
    for source, config_path in SEARCH_CONFIG_PATHS:
        if not config_path.exists():
            continue

        record_source_file(session, config_path, source=source, kind="config")

        raw_text = config_path.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_config_text(raw_text)

        stored_path = relative_path(config_path)
        existing = session.scalar(select(Config).where(Config.source_file == stored_path))
        if existing is None:
            existing = Config(source=source, source_file=stored_path)
            session.add(existing)

        existing.source = source
        existing.raw_text = raw_text
        existing.source_file = stored_path
        existing.query_string = parsed["query_string"]
        existing.filters = parsed["filters"]
        existing.year_range = parsed["year_range"]
        existing.search_url = parsed["search_url"]
        written += 1

    session.flush()
    return written
