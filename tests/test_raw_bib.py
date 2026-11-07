"""Tests for the .bib ingestion loader (ingest/raw_bib.py).

Covers the historical gotcha CLAUDE.md flags: IEEE's .bib export closes one
entry and opens the next on the same line (`month={Feb},}@ARTICLE{...}`),
which a naive line/`@` splitter would merge or truncate. bibtexparser 2.x (a
real BibTeX parser, not string splitting) is used specifically to handle
this correctly -- these tests exercise that assumption directly rather than
re-testing bibtexparser itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import relative_path
from db.raw_models import BibEntry, BibParseFailure
from ingest.raw_bib import _load_bib_dir, count_bib_parse_failures
from quality import assert_contract, raw_contract


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


# ---------------------------------------------------------------------------
# No silent input loss (Wave P0-5a): bibtexparser does not raise on a block it
# cannot parse -- it drops it into `library.failed_blocks` and parses on, so a
# file with one broken entry used to load silently minus that entry.
# ---------------------------------------------------------------------------


def test_load_bib_dir_counts_and_records_a_malformed_block_among_valid_entries(
    raw_session, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="ingest.raw_bib")
    # IEEE's glued shape, with the middle block missing the `=` of a field.
    path = _write_bib(
        tmp_path,
        "glued_with_one_broken.bib",
        "@ARTICLE{First2019,\n"
        "  author={A. Author},\n"
        "  title={First Paper},\n"
        "  year={2019},\n"
        "  month={Feb},}@ARTICLE{Broken2019,\n"
        "  author={B. Author},\n"
        "  title {Broken Paper},\n"
        "  year={2019},}@ARTICLE{Third2019,\n"
        "  author={C. Author},\n"
        "  title={Third Paper},\n"
        "  year={2019},\n"
        "}\n",
    )
    stored_path = relative_path(path)

    written = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()

    # The valid neighbours load; the broken block is counted, not lost.
    assert written == 2
    assert {e.bib_key for e in raw_session.query(BibEntry).all()} == {"First2019", "Third2019"}
    assert count_bib_parse_failures(raw_session) == 1
    failure = raw_session.query(BibParseFailure).one()
    assert (failure.source, failure.source_file) == ("ieee", stored_path)
    # 1-based, the glued line that opens the broken block.
    assert failure.start_line == 5
    assert failure.block_type == "ParsingFailedBlock"
    assert "Expected a `=`" in failure.error
    assert failure.raw_text.startswith("@ARTICLE{Broken2019")
    assert f"{stored_path}: .bib block at line 5 could not be parsed" in caplog.text

    # The raw contract reports it as a warning: visible, not yet blocking.
    results = raw_contract(raw_session, "v1")
    check = next(result for result in results if result.check_id == "raw.bib_parse_failures")
    assert (check.severity, check.passed, check.observed, check.expected) == (
        "warning",
        False,
        1,
        0,
    )
    assert check.details["failed_blocks_by_file"] == {stored_path: 1}
    assert check.details["blocks"] == [
        {"source_file": stored_path, "line": 5, "error": failure.error}
    ]
    assert_contract("raw", [check])
    # And the failure keeps referencing a manifest path, like every child row.
    references = next(r for r in results if r.check_id == "raw.child_source_references")
    assert references.passed

    # An unchanged file is skipped on the next run, and its failure stays on record.
    assert _load_bib_dir(raw_session, "ieee", tmp_path) == 0
    assert count_bib_parse_failures(raw_session) == 1


def test_an_unbalanced_brace_swallows_the_rest_of_the_file_and_is_counted(raw_session, tmp_path):
    """The worst case: everything after the broken block vanishes with it."""
    _write_bib(
        tmp_path,
        "unbalanced.bib",
        "@ARTICLE{First2019,\n"
        "  title={First Paper},\n"
        "}@ARTICLE{Broken2019,\n"
        "  title={Broken {Paper},\n"
        "}@ARTICLE{Third2019,\n"
        "  title={Third Paper},\n"
        "}\n",
    )

    written = _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()

    assert written == 1
    assert [e.bib_key for e in raw_session.query(BibEntry).all()] == ["First2019"]
    failure = raw_session.query(BibParseFailure).one()
    assert failure.start_line == 3
    # The excerpt is the evidence of what the brace took with it.
    assert "Third2019" in failure.raw_text


def test_a_clean_file_records_no_failure(raw_session, tmp_path):
    _write_bib(tmp_path, "clean.bib", "@ARTICLE{Clean2020,\n  title={Clean},\n}\n")

    _load_bib_dir(raw_session, "ieee", tmp_path)
    raw_session.commit()

    assert count_bib_parse_failures(raw_session) == 0
    check = next(
        r for r in raw_contract(raw_session, "v1") if r.check_id == "raw.bib_parse_failures"
    )
    assert check.passed
