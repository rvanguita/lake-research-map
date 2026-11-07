"""`ingest/raw_pdfs.py`: the PDF inventory hashes each file once per run."""

from __future__ import annotations

from db.raw_models import PdfFile, SourceFile
from ingest import hashing, raw_pdfs


def _inventory(monkeypatch, tmp_path) -> list[str]:
    """Run the inventory over `tmp_path`, returning every path that was hashed."""
    hashed: list[str] = []
    real = hashing.sha256_file

    def counting(path, *args, **kwargs):
        hashed.append(path.name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(hashing, "sha256_file", counting)
    # Also catch a loader that hashes through its own import of the helper.
    monkeypatch.setattr(raw_pdfs, "sha256_file", counting, raising=False)
    monkeypatch.setattr(raw_pdfs, "ARTICLES_DIR", tmp_path)
    for module in (hashing, raw_pdfs):
        monkeypatch.setattr(module, "relative_path", lambda path: path.name)
    return hashed


def test_each_pdf_is_hashed_once_and_recorded_in_both_tables(monkeypatch, tmp_path, raw_session):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-b")
    hashed = _inventory(monkeypatch, tmp_path)

    written = raw_pdfs.load_pdf_inventory(raw_session)

    assert written == 2
    assert sorted(hashed) == ["a.pdf", "b.pdf"]
    manifest = {row.path: row.sha256 for row in raw_session.query(SourceFile)}
    inventory = {row.filename: row.sha256 for row in raw_session.query(PdfFile)}
    assert inventory == manifest


def test_rerun_writes_only_the_changed_pdf(monkeypatch, tmp_path, raw_session):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-b")
    _inventory(monkeypatch, tmp_path)
    raw_pdfs.load_pdf_inventory(raw_session)

    (tmp_path / "b.pdf").write_bytes(b"%PDF-b, revised")

    assert raw_pdfs.load_pdf_inventory(raw_session) == 1
    assert raw_pdfs.load_pdf_inventory(raw_session) == 0
    revised = raw_session.query(PdfFile).filter_by(filename="b.pdf").one()
    assert revised.sha256 == hashing.sha256_file(tmp_path / "b.pdf")
    assert revised.size_bytes == len(b"%PDF-b, revised")


def test_a_manifest_row_that_predates_the_inventory_row_is_still_inventoried(
    monkeypatch, tmp_path, raw_session
):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    _inventory(monkeypatch, tmp_path)
    # The manifest already knows the file (a run killed between the two writes
    # leaves exactly this state), so the loader's own record_source_file call
    # reports it unchanged. The `changed` flag must not be the skip test: the
    # inventory row has to be written regardless.
    _row, changed = hashing.record_source_file(
        raw_session, tmp_path / "a.pdf", source="articles", kind="pdf"
    )
    raw_session.flush()
    assert changed is True

    assert raw_pdfs.load_pdf_inventory(raw_session) == 1
    inventoried = raw_session.query(PdfFile).filter_by(filename="a.pdf").one()
    assert inventoried.sha256 == hashing.sha256_file(tmp_path / "a.pdf")
