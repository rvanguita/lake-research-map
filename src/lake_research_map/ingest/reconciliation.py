"""Reconcile the active Raw projection with an immutable source snapshot."""

from __future__ import annotations

from collections import Counter

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from lake_research_map.db.raw_models import (
    BibEntry,
    Config,
    DatasetSourceFile,
    IeeeCsvRow,
    PdfFile,
    SourceBlob,
    SourceChange,
    SourceFile,
    SourceRevision,
)
from lake_research_map.ingest.snapshots import ScannedSource, diff_manifests


def active_manifest(session: Session) -> dict[str, tuple[str | None, str]]:
    return {
        row.path: (row.source_revision_id, row.sha256)
        for row in session.scalars(select(SourceFile)).all()
    }


def remove_absent_sources(session: Session, present_paths: set[str]) -> int:
    stale = [
        row.path
        for row in session.scalars(select(SourceFile)).all()
        if row.path not in present_paths
    ]
    if not stale:
        return 0
    session.execute(delete(Config).where(Config.source_file.in_(stale)))
    session.execute(delete(IeeeCsvRow).where(IeeeCsvRow.source_file.in_(stale)))
    session.execute(delete(BibEntry).where(BibEntry.source_file.in_(stale)))
    session.execute(delete(PdfFile).where(PdfFile.path.in_(stale)))
    session.execute(delete(SourceFile).where(SourceFile.path.in_(stale)))
    session.flush()
    return len(stale)


def persist_snapshot(
    session: Session,
    sources: list[ScannedSource],
    *,
    version_id: str,
    execution_id: str,
    previous: dict[str, tuple[str | None, str]],
) -> dict[str, int]:
    """Persist immutable revisions and attach lineage to active Raw rows."""
    current = {source.path: source for source in sources}
    session.execute(
        delete(DatasetSourceFile).where(DatasetSourceFile.dataset_version_id == version_id)
    )
    changes = diff_manifests(previous, current)

    for source in sources:
        blob = session.get(SourceBlob, source.sha256)
        if blob is None:
            session.add(
                SourceBlob(
                    sha256=source.sha256,
                    archive_path=source.archive_path,
                    size_bytes=source.size_bytes,
                )
            )
        revision = session.get(SourceRevision, source.revision_id)
        if revision is None:
            session.add(
                SourceRevision(
                    revision_id=source.revision_id,
                    path=source.path,
                    source=source.source,
                    kind=source.kind,
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                    mtime=source.mtime,
                )
            )
        session.add(
            DatasetSourceFile(
                dataset_version_id=version_id,
                path=source.path,
                source_revision_id=source.revision_id,
            )
        )

        manifest_row = session.scalar(select(SourceFile).where(SourceFile.path == source.path))
        if manifest_row is None:
            manifest_row = SourceFile(
                path=source.path,
                source=source.source,
                kind=source.kind,
                sha256=source.sha256,
                size_bytes=source.size_bytes,
                mtime=source.mtime,
            )
            session.add(manifest_row)
        manifest_row.source = source.source
        manifest_row.kind = source.kind
        manifest_row.sha256 = source.sha256
        manifest_row.size_bytes = source.size_bytes
        manifest_row.mtime = source.mtime
        manifest_row.dataset_version_id = version_id
        manifest_row.source_revision_id = source.revision_id

        lineage = {
            "dataset_version_id": version_id,
            "source_revision_id": source.revision_id,
        }
        if source.kind == "config":
            session.execute(
                update(Config).where(Config.source_file == source.path).values(**lineage)
            )
        elif source.kind == "csv":
            session.execute(
                update(IeeeCsvRow).where(IeeeCsvRow.source_file == source.path).values(**lineage)
            )
        elif source.kind == "bib":
            session.execute(
                update(BibEntry).where(BibEntry.source_file == source.path).values(**lineage)
            )
        elif source.kind == "pdf":
            session.execute(
                update(PdfFile)
                .where(PdfFile.path == source.path)
                .values(archive_path=source.archive_path, **lineage)
            )

    session.execute(delete(SourceChange).where(SourceChange.execution_id == execution_id))
    session.add_all(
        [
            SourceChange(
                execution_id=execution_id,
                dataset_version_id=version_id,
                path=str(change["path"]),
                change_type=str(change["change_type"]),
                previous_revision_id=change["previous_revision_id"],
                current_revision_id=change["current_revision_id"],
            )
            for change in changes
        ]
    )
    session.flush()
    counts = Counter(str(change["change_type"]) for change in changes)
    return {
        key: counts.get(key, 0) for key in ("added", "modified", "renamed", "removed", "unchanged")
    }
