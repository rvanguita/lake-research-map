"""Inventory data/articles/*.pdf into lit_raw.pdf_files.

Only records file identity (name, path, hash, size) here -- title matching
against articles happens in silver, and text extraction happens in gold.
"""

from __future__ import annotations

from sqlalchemy import select

from lake_research_map.config import ARTICLES_DIR, relative_path
from lake_research_map.db.raw_models import PdfFile
from lake_research_map.ingest.hashing import record_source_file, sha256_file


def load_pdf_inventory(session) -> int:
    written = 0
    for pdf_path in sorted(ARTICLES_DIR.glob("*.pdf")):
        record_source_file(session, pdf_path, source="articles", kind="pdf")

        existing = session.scalar(select(PdfFile).where(PdfFile.filename == pdf_path.name))
        stat = pdf_path.stat()
        sha = sha256_file(pdf_path)
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
