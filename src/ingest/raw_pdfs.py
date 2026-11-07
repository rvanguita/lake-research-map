"""Inventory data/articles/*.pdf into `raw.lit_pdf_files`.

Only records file identity (name, path, hash, size) here -- title matching
against articles happens in silver, and text extraction happens in gold.
"""

from __future__ import annotations

from sqlalchemy import select

from config import ARTICLES_DIR, relative_path
from db.raw_models import PdfFile
from ingest.hashing import record_source_file


def load_pdf_inventory(session) -> int:
    written = 0
    for pdf_path in sorted(ARTICLES_DIR.glob("*.pdf")):
        # The manifest row already carries the file's hash: reading a PDF a
        # second time just to hash it again doubled the stage's I/O. Its
        # `changed` flag is deliberately not the skip test, though: it says
        # whether the manifest moved, not whether this inventory row exists and
        # agrees (a manifest row can predate the inventory row), so the row's
        # own hash is compared instead.
        manifest, _changed = record_source_file(session, pdf_path, source="articles", kind="pdf")
        sha = manifest.sha256

        existing = session.scalar(select(PdfFile).where(PdfFile.filename == pdf_path.name))
        stat = pdf_path.stat()
        if existing is None:
            session.add(
                PdfFile(
                    filename=pdf_path.name,
                    path=relative_path(pdf_path),
                    sha256=sha,
                    size_bytes=stat.st_size,
                )
            )
            written += 1
        elif existing.sha256 != sha:
            existing.path = relative_path(pdf_path)
            existing.sha256 = sha
            existing.size_bytes = stat.st_size
            written += 1

    session.flush()
    return written
