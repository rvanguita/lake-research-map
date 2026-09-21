"""Shared helpers for idempotent raw ingestion: hash a file and check/record
it against lit_raw.source_files so re-running a stage doesn't reprocess
unchanged inputs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.config import relative_path
from lake_research_map.db.raw_models import SourceFile
from lake_research_map.db.time import naive_utc_now


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_source_file(
    session: Session, path: Path, source: str, kind: str
) -> tuple[SourceFile, bool]:
    """Insert or update the manifest row for `path`.

    Returns (row, changed) where `changed` is True if the file is new or its
    hash differs from what was recorded before -- callers use this to decide
    whether to reprocess the file's contents.
    """
    sha = sha256_file(path)
    stat = path.stat()
    stored_path = relative_path(path)
    existing = session.scalar(select(SourceFile).where(SourceFile.path == stored_path))
    if existing is None:
        row = SourceFile(
            path=stored_path,
            source=source,
            kind=kind,
            sha256=sha,
            size_bytes=stat.st_size,
            mtime=dt.datetime.fromtimestamp(stat.st_mtime),
        )
        session.add(row)
        session.flush()
        return row, True

    changed = existing.sha256 != sha
    existing.source = source
    existing.kind = kind
    if changed:
        existing.sha256 = sha
        existing.size_bytes = stat.st_size
        existing.mtime = dt.datetime.fromtimestamp(stat.st_mtime)
        existing.ingested_at = naive_utc_now()
        session.flush()
    return existing, changed
