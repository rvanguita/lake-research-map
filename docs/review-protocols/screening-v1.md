# Review protocol — no-DOI rejection audit (v1)

**Workflow:** `screening` · **Labels:** `include` · `exclude` · `uncertain`

> **Draft.** The operational rules are mine. **Section 3 is a genuine research
> decision and the criteria below are a placeholder — replace them with the
> review's own inclusion criteria before labelling anything**, then bump to `v2`
> (a protocol version is immutable once used).

## 1. What you are judging

Each subject is `reject::<source>::<source id>`. These are records dropped at the
silver stage for having no DOI, and they are invisible to every downstream analysis.

The question is: **should this record have been in the corpus?**

## 2. Why this sample looks the way it does

48 records were rejected, all for `no_doi`, and 30 are sampled — 62% coverage, high
enough to bound the bias directly.

The point is not to recover the records. It is to answer whether dropping them was
unbiased. If the rejected set turns out to be systematically in-scope work, "records
without a DOI were excluded" stops being a footnote and becomes a limitation on
every count in the review.

## 3. Inclusion criteria — PLACEHOLDER, REPLACE BEFORE USE

The review's topic is planning of electric power distribution systems. As a
first approximation only:

- `include` — the record is a research article on distribution-system planning and
  would have belonged in the corpus.
- `exclude` — front matter (preface, index, table of contents), a duplicate of a
  record already present, or work on a different topic — notably the
  logistics/supply-chain reading of "distribution".
- `uncertain` — the title alone is insufficient and no abstract is available.

Record a `rationale` for every `include`: those are the records whose loss the
review has to disclose.
