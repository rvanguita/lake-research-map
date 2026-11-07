"""Tests for the .bib ingestion loader (ingest/raw_bib.py).

Covers the historical gotcha CLAUDE.md flags: IEEE's .bib export closes one
entry and opens the next on the same line (`month={Feb},}@ARTICLE{...}`),
which a naive line/`@` splitter would merge or truncate. bibtexparser 2.x (a
real BibTeX parser, not string splitting) is used specifically to handle
this correctly -- these tests exercise that assumption directly rather than
re-testing bibtexparser itself.
"""

from __future__ import annotations

from pathlib import Path

from db.raw_models import BibEntry
from ingest.raw_bib import _load_bib_dir


def _write_bib(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_bib_dir_parses_single_well_formed_entry(raw_session, tmp_path):
    _write_bib(
        tmp_path,
        "single.bib",
        """@ARTICLE{Liu2020,
  author={Liu, J.},
  title={A Study of Distribution Networks},
  year={2020},
  doi={10.1109/example.2020.0001},
}
""",
    )

    written = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()

    assert written == 1
    entries = raw_session.query(BibEntry).all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.source == "ieee"
    assert entry.bib_key == "Liu2020"
    assert entry.entry_type.lower() == "article"
    assert entry.doi == "10.1109/example.2020.0001"
    assert entry.fields["title"] == "A Study of Distribution Networks"


def test_load_bib_dir_parses_entries_with_no_separator_between_them(raw_session, tmp_path):
    # Mirrors the real IEEE export gotcha: one entry's closing brace is
    # immediately followed by the next entry's `@` on the same line, with no
    # blank line or newline between them.
    _write_bib(
        tmp_path,
        "no_separator.bib",
        "@ARTICLE{First2019,\n"
        "  author={A. Author},\n"
        "  title={First Paper},\n"
        "  year={2019},\n"
        "  month={Feb},}@ARTICLE{Second2019,\n"
        "  author={B. Author},\n"
        "  title={Second Paper},\n"
        "  year={2019},\n"
        "}\n",
    )

    written = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()

    assert written == 2
    entries = {e.bib_key: e for e in raw_session.query(BibEntry).all()}
    assert set(entries) == {"First2019", "Second2019"}
    assert entries["First2019"].fields["title"] == "First Paper"
    assert entries["Second2019"].fields["title"] == "Second Paper"


def test_load_bib_dir_normalizes_missing_doi_to_none(raw_session, tmp_path):
    _write_bib(
        tmp_path,
        "no_doi.bib",
        """@ARTICLE{NoDoi2021,
  author={C. Author},
  title={No DOI Paper},
  year={2021},
}
""",
    )

    _load_bib_dir(raw_session, "elsevier", tmp_path)
    raw_session.commit()

    entry = raw_session.query(BibEntry).one()
    assert entry.source == "elsevier"
    assert entry.doi is None


def test_load_bib_dir_skips_unchanged_files_and_reprocesses_on_change(raw_session, tmp_path):
    bib_path = _write_bib(
        tmp_path,
        "repeat.bib",
        """@ARTICLE{Repeat2022,
  author={D. Author},
  title={Repeat Paper},
  year={2022},
}
""",
    )

    first = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()
    assert first == 1
    assert raw_session.query(BibEntry).count() == 1

    # Second pass over the same, unchanged file must not reprocess/duplicate it.
    second = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()
    assert second == 0
    assert raw_session.query(BibEntry).count() == 1

    # Editing the file's content should trigger a reprocess, replacing (not
    # duplicating) the row for that source_file.
    bib_path.write_text(
        bib_path.read_text(encoding="utf-8").replace("Repeat Paper", "Repeat Paper Revised"),
        encoding="utf-8",
    )
    third = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()
    assert third == 1
    assert raw_session.query(BibEntry).count() == 1
    assert raw_session.query(BibEntry).one().fields["title"] == "Repeat Paper Revised"
