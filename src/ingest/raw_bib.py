"""Parse every supported IEEE and ScienceDirect .bib export into
`raw.lit_bib_entries`.

Uses bibtexparser (a real BibTeX parser) rather than naive line/`@`
splitting -- the IEEE files close one entry and open the next on the same
line (`month={Feb},}@ARTICLE{...}`), which a naive splitter would merge or
truncate. bibtexparser 2.x handles this correctly out of the box.

What it does not do is raise. A block it cannot parse goes to
`library.failed_blocks` and parsing continues, so a file with one broken
entry used to load silently minus that entry -- and an unbalanced brace in a
glued IEEE block swallows every entry after it, with no error anywhere
(measured on 2026-11-07: the entries after the brace simply vanish). Every
failed block is therefore logged with its file and line, stored in
`raw.lit_bib_parse_failures` beside the file's entries, counted into the run
stats (`bib_failed_blocks`) and reported by the raw contract
(`raw.bib_parse_failures`).
"""

from __future__ import annotations

import logging

import bibtexparser
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from config import ELSEVIER_BIB_DIRS, IEEE_BIB_DIRS, relative_path
from db.raw_models import BibEntry, BibParseFailure
from ingest.hashing import record_source_file

logger = logging.getLogger(__name__)

# How much of a failed block is kept as evidence. An unbalanced brace makes the
# block run to the end of the file, and a whole page of an export is not worth
# storing twice on a shared server.
FAILED_BLOCK_EXCERPT_CHARS = 1000


def _failure_reason(block) -> str:
    """What bibtexparser knew about the failure, in one line.

    A `BlockAbortedException` keeps its message in `abort_reason` and has an
    empty `str()`; a duplicate-key block carries a plain `Exception` with a
    message. The class name is the fallback, never an empty reason.
    """
    error = getattr(block, "error", None)
    reason = getattr(error, "abort_reason", None) or (str(error) if error is not None else "")
    return str(reason or type(block).__name__).strip()


def _record_failed_blocks(session: Session, source: str, stored_path: str, library) -> int:
    failed = 0
    for block in library.failed_blocks:
        # bibtexparser counts lines from zero; editors and people count from one.
        line = block.start_line + 1 if block.start_line is not None else None
        reason = _failure_reason(block)
        logger.warning(
            "%s: .bib block at line %s could not be parsed and was dropped (%s: %s)",
            stored_path,
            "?" if line is None else line,
            type(block).__name__,
            reason,
        )
        session.add(
            BibParseFailure(
                source=source,
                source_file=stored_path,
                start_line=line,
                block_type=type(block).__name__,
                error=reason,
                raw_text=(block.raw or "")[:FAILED_BLOCK_EXCERPT_CHARS] or None,
            )
        )
        failed += 1
    return failed


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
        session.execute(delete(BibParseFailure).where(BibParseFailure.source_file == stored_path))

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
        _record_failed_blocks(session, source, stored_path, library)

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


def count_bib_parse_failures(session: Session) -> int:
    """Failed blocks on record: what `run_raw` reports as `bib_failed_blocks`.

    Read back from the table rather than from this run's parsing, so an
    unchanged run -- which skips every file -- still reports the failures its
    stored entries were parsed with: the same number the raw contract sees.
    """
    return int(session.scalar(select(func.count()).select_from(BibParseFailure)) or 0)
