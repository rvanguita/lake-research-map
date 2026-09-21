"""Parse every supported IEEE and ScienceDirect .bib export into
lit_raw.bib_entries.

Uses bibtexparser (a real BibTeX parser) rather than naive line/`@`
splitting -- the IEEE files close one entry and open the next on the same
line (`month={Feb},}@ARTICLE{...`), which a naive splitter would merge or
truncate. bibtexparser 2.x handles this correctly out of the box.
"""

from __future__ import annotations

import bibtexparser
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from lake_research_map.config import ELSEVIER_BIB_DIRS, IEEE_BIB_DIRS, relative_path
from lake_research_map.db.raw_models import BibEntry
from lake_research_map.ingest.hashing import record_source_file


def _load_bib_dir(session: Session, source: str, directory) -> int:
    written = 0
    for bib_path in sorted(directory.glob("*.bib")):
        _, changed = record_source_file(session, bib_path, source=source, kind="bib")
        stored_path = relative_path(bib_path)

        already_loaded = session.scalar(
            select(BibEntry.id).where(BibEntry.source_file == stored_path)
        )
        if not changed and already_loaded is not None:
            continue

        library = bibtexparser.parse_file(str(bib_path))
        session.execute(delete(BibEntry).where(BibEntry.source_file == stored_path))

        for entry in library.entries:
            fields = {f.key: f.value for f in entry.fields}
            doi = fields.get("doi")
            doi = doi.strip() if isinstance(doi, str) else None
            session.add(
                BibEntry(
                    source=source,
                    bib_key=entry.key,
                    entry_type=entry.entry_type,
                    source_file=stored_path,
                    fields=fields,
                    doi=doi,
                )
            )
            written += 1

    return written


def load_bib_entries(session: Session) -> int:
    """Parse both sources' .bib files. Returns rows written."""
    written = 0
    for directory in IEEE_BIB_DIRS:
        written += _load_bib_dir(session, "ieee", directory)
    for directory in ELSEVIER_BIB_DIRS:
        written += _load_bib_dir(session, "elsevier", directory)
    session.flush()
    return written
