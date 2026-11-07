"""The pure cleaners applied to pypdf output before chunking."""

from __future__ import annotations

from transform.pdf_text import (
    clean_pdf_text,
    dehyphenate,
    drop_page_number_lines,
    reflow_lines,
    strip_reference_section,
)

BODY = "\n".join(f"Sentence number {index} of the paper body." for index in range(40))


def test_strip_reference_section_cuts_at_the_last_standalone_heading_past_60_percent():
    text = BODY + "\nReferences\n[1] H. L. Willis, Spatial Load Forecasting.\n[2] Another."

    stripped, removed = strip_reference_section(text)

    assert removed is True
    assert stripped == BODY
    assert "Willis" not in stripped


def test_strip_reference_section_ignores_a_heading_in_the_first_half():
    text = "Contents\nReferences\n" + BODY

    assert strip_reference_section(text) == (text, False)


def test_strip_reference_section_recognises_small_caps_numbers_and_bibliography():
    for heading in (
        "R EFERENCES",
        "R E F E R E N C E S",
        "VIII. REFERENCES",
        "7 References:",
        "BIBLIOGRAPHY",
    ):
        stripped, removed = strip_reference_section(f"{BODY}\n{heading}\n[1] A cited work.")
        assert removed is True, heading
        assert stripped == BODY


def test_strip_reference_section_leaves_text_without_a_heading_untouched():
    text = BODY + "\nThe references cited above are discussed in Section 2."

    assert strip_reference_section(text) == (text, False)


def test_dehyphenate_joins_only_lowercase_line_break_hyphens():
    text = "distri-\nbution planning, IEEE-\n33 bus, end of line -\nThe next"

    joined, count = dehyphenate(text)

    assert joined == "distribution planning, IEEE-\n33 bus, end of line -\nThe next"
    assert count == 1


def test_reflow_joins_single_newlines_and_keeps_paragraph_breaks():
    text = "First line of a\nparagraph.\n\nSecond   paragraph\nhere.\n\n\n\nThird."

    assert reflow_lines(text) == "First line of a paragraph.\n\nSecond paragraph here.\n\nThird."


def test_drop_page_number_lines_keeps_short_words_numerals_and_blank_lines():
    text = "Body text.\n12\n--\nII.\nA.\n(a)\n\nMore text."

    cleaned, dropped = drop_page_number_lines(text)

    assert cleaned == "Body text.\nII.\nA.\n(a)\n\nMore text."
    assert dropped == 2


def test_clean_pdf_text_reports_each_change_and_yields_sentence_boundaries():
    raw = (
        BODY.replace("paper body.", "paper\nbody.", 1)
        + "\n7\nThe distri-\nbution network is planned.\nReferences\n[1] Willis."
    )

    cleaned, report = clean_pdf_text(raw)

    assert report == {
        "reference_section_removed": True,
        "short_lines_dropped": 1,
        "hyphens_joined": 1,
    }
    assert "distribution network is planned." in cleaned
    assert "\n" not in cleaned
    assert ". Sentence number 1 " in cleaned
    assert "Willis" not in cleaned


def test_clean_pdf_text_is_stable_on_text_it_already_cleaned():
    cleaned, _ = clean_pdf_text(BODY + "\n\nSecond paragraph.\nReferences\n[1] Willis.")

    again, report = clean_pdf_text(cleaned)

    assert again == cleaned
    assert report == {
        "reference_section_removed": False,
        "short_lines_dropped": 0,
        "hyphens_joined": 0,
    }
