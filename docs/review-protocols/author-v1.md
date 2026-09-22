# Review protocol — author identity (v1)

**Workflow:** `author` · **Labels:** `same` · `different` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

## 1. What you are judging

Each subject is `<name A>||<name B>`. The question is: **are these two names the
same person?**

## 2. Why this sample looks the way it does

Author names are canonicalized heuristically, which fails in two opposite
directions:

- **wrong merge** — two people collapsed into one name, inflating a ranking and
  inventing collaborations that never happened;
- **wrong split** — one person spread across several names, deflating their output.

The sample interleaves both, because measuring only one bounds half the error.

The population statistic is itself a finding worth recording: **19,009 merge-risk
pairs across 9,688 distinct names** — roughly two suspicious pairs per name. That is
the scale of the problem this package exists to bound; 40 labelled pairs give an
error rate, not a fix.

## 3. Decision rules

- `same` — the same researcher. Use the corpus as evidence: shared co-authors,
  venue, subject area, and adjacent publication years.
- `different` — different people. Common surname plus compatible initials is **not**
  sufficient evidence of `same`; without corroboration it is `uncertain`.
- `uncertain` — no corroborating evidence either way. Expect to use this a lot, and
  do not force a decision: an honest `uncertain` rate is itself a result.

Judge from within this corpus only. Do not consult ORCID, Scopus or a personal page:
the canonicalizer has only these names and these papers, so external evidence would
measure a different system than the one under test.

Record a one-line `rationale` for every `same` and every `different`.
