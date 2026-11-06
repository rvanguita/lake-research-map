"""Clean pypdf output before it is chunked and embedded.

pypdf returns one `\\n` per visual line, keeps the hyphen of every word broken
at a line end, and keeps page furniture -- page numbers, column debris -- as
lines of their own. Measured on the 97 linked PDFs on 2026-11-06 (before ->
after cleaning):

- 88 carry their reference list after a standalone reference heading, at
  78-90% of the text (p10-p90). Those chunks are citation lists, not evidence,
  and they compete with the paper's own text in every search. Removing them
  takes the full-text chunks from 4,437 to 3,623.
- 6,323 words were split across a line break with a hyphen.
- 7,368 lines were page furniture (at most three characters, no letter).
- 3,439 of the 4,437 full-text chunks contained no blank line, so
  `_chunk_text` relied on `". "` -- which misses every sentence that ends at a
  line end. Reflowed text lets it see those too: non-final chunks ending a
  sentence went from 89.7% to 92.7%.

Each function here is pure and composed by `clean_pdf_text`, whose order
matters: the reference heading and the page-number lines are only recognisable
while lines are still lines, so both run before the line structure is reflowed.
"""

from __future__ import annotations

import re

# The last reference heading must sit past this fraction of the text. A table of
# contents or an early "References" mention is not where the list starts; the
# measured headings sit at 0.78-0.90.
REFERENCE_MIN_POSITION = 0.6

# A heading alone on its line: "References", "REFERENCES", small caps that
# pypdf letter-spaces ("R EFERENCES", "R E F E R E N C E S"), an optional
# section number ("7. References", "VIII. REFERENCES"), or "Bibliography".
_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(?:(?:\d+|[IVXLC]+)\.?[ \t]+)?"
    r"(?:R[ \t]?E[ \t]?F[ \t]?E[ \t]?R[ \t]?E[ \t]?N[ \t]?C[ \t]?E[ \t]?S|BIBLIOGRAPHY)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# Only a lowercase letter on both sides: "distri-\nbution" is one word, while
# "IEEE-\n33" or "-\nThe" are left alone.
_LINE_END_HYPHEN = re.compile(r"([a-z])-[ \t]*\n[ \t]*([a-z])")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SPACES = re.compile(r"[ \t]+")


def strip_reference_section(text: str) -> tuple[str, bool]:
    """Cut at the last standalone reference heading, if it is past 60% of the text.

    Everything after the heading goes, appendices included: in this corpus an
    appendix after the references is rare, and keeping the list costs far more
    (one in six full-text chunks was a citation list).
    """
    headings = list(_REFERENCE_HEADING.finditer(text))
    if not headings:
        return text, False
    start = headings[-1].start()
    if start < REFERENCE_MIN_POSITION * len(text):
        return text, False
    return text[:start].rstrip(), True


def drop_page_number_lines(text: str) -> tuple[str, int]:
    """Drop lines of at most three characters with no letter in them.

    Page numbers ("12"), rules ("--") and stray marks go; "II.", "A." and "(a)"
    stay, and blank lines are kept because they are the paragraph breaks.
    """
    kept: list[str] = []
    dropped = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if 0 < len(stripped) <= 3 and not any(char.isalpha() for char in stripped):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def dehyphenate(text: str) -> tuple[str, int]:
    """Rejoin words split by a hyphen at a line end.

    A genuine compound that happens to break at its hyphen is joined too
    ("power-\\nflow" becomes "powerflow"); there is no dictionary to tell the two
    apart, and a lost hyphen costs a tokenizer far less than a split word.
    """
    return _LINE_END_HYPHEN.subn(r"\1\2", text)


def reflow_lines(text: str) -> str:
    """Join the lines of each paragraph; keep blank-line paragraph breaks.

    After this, a sentence that ended at a line end ends in ". " too, which is
    the boundary the chunker looks for before cutting at the size cap.
    """
    paragraphs = []
    for paragraph in _PARAGRAPH_BREAK.split(text.replace("\r\n", "\n")):
        joined = _SPACES.sub(" ", paragraph.replace("\n", " ")).strip()
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def clean_pdf_text(text: str) -> tuple[str, dict[str, int | bool]]:
    """Apply the cleaners in their required order and report what each changed."""
    text, references_removed = strip_reference_section(text)
    text, short_lines = drop_page_number_lines(text)
    text, hyphens = dehyphenate(text)
    return reflow_lines(text), {
        "reference_section_removed": references_removed,
        "short_lines_dropped": short_lines,
        "hyphens_joined": hyphens,
    }
