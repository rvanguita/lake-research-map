# Review protocol — PDF to article pairing (v1)

**Workflow:** `pdf` · **Labels:** `match` · `mismatch` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

## 1. What you are judging

Each subject is `<DOI>::<path to PDF>`. You are judging **the pairing**, not either
side alone: the article can be correct and its PDF still be the wrong file.

## 2. Why this sample looks the way it does

PDFs are linked to articles by fuzzy title match at threshold 85. The sample is
banded to straddle that threshold, because a sample of confident matches cannot tell
you where the threshold should sit.

On the current corpus the bands are badly uneven — **96 of the 97 linked PDFs score
95 or above and exactly one falls in 85–90**. A sample of 16 therefore cannot
calibrate the threshold; it can only detect whether high-confidence matches are
trustworthy. Record that limitation rather than reporting a threshold as validated.

## 3. Decision rules

- `match` — the PDF is the full text of that DOI.
- `mismatch` — it is a different article. A preprint or an extended conference
  version of the same work is **`mismatch`**: the DOI identifies one specific
  record, and analyses that read page counts or full text must not silently use
  another version.
- `uncertain` — the file does not open, is truncated, is scanned without usable
  text, or the title is genuinely ambiguous between two records.

Do **not** look at `pdf_match_score` before deciding. It is the thing under test.

Compare the PDF's first page — title, authors, venue, year — against the record.
Record a `rationale` for every `mismatch` and `uncertain`.
