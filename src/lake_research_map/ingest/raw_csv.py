"""Load configured IEEE Xplore metadata CSV exports (28 columns)
into lit_raw.ieee_csv_rows -- one row per CSV row, values kept as JSON so no
typing/normalization happens yet (that's bronze's job).
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from lake_research_map.config import IEEE_DIR, relative_path
from lake_research_map.db.raw_models import IeeeCsvRow
from lake_research_map.ingest.hashing import record_source_file


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_ieee_csv(session: Session) -> int:
    """Load every configured IEEE metadata CSV file and return rows written."""
    written = 0
    for csv_path in sorted(IEEE_DIR.glob("export*.csv")):
        _, changed = record_source_file(session, csv_path, source="ieee", kind="csv")
        stored_path = relative_path(csv_path)

        already_loaded = session.scalar(
            select(IeeeCsvRow.id).where(IeeeCsvRow.source_file == stored_path)
        )
        if not changed and already_loaded is not None:
            continue

        df = pd.read_csv(csv_path)
        session.execute(delete(IeeeCsvRow).where(IeeeCsvRow.source_file == stored_path))

        for idx, row in df.iterrows():
            fields = {col: _clean_value(row[col]) for col in df.columns}
            doi = fields.get("DOI")
            doi = doi.strip() if isinstance(doi, str) else None
            session.add(
                IeeeCsvRow(
                    row_index=int(idx),
                    source_file=stored_path,
                    fields=fields,
                    doi=doi,
                )
            )
            written += 1

    session.flush()
    return written
