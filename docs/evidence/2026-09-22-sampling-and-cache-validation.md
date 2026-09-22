# Evidence — a sampling bias in the crawl, and a cache that turned out to be sound

**Date:** 2026-09-22 · **Branch:** `feat/roadmap-closure`
**Affects:** `WP-07`, `WP-23`, `WP-24`, `ADR-07`, `RISK-09`

Two analyses over data the project already held. One invalidates figures
reported earlier the same day; the other confirms an artifact that had never
been checked.

## 1. The observed subset was one publisher

`refresh_openalex_observations` walked `sorted(dois)`. A DOI's registrant prefix
*is* its publisher, so the sorted key is correlated with source, and `10.1016`
sorts entirely before `10.1109`. The crawl hit its request quota part-way
through the Elsevier block:

| Registrant | Observed | Corpus |
|---|---:|---:|
| `10.1016` Elsevier | 989 (98.9%) | 1,612 (51.9%) |
| `10.1109` IEEE | **0 (0.0%)** | 1,340 (43.1%) |
| `10.1049` IET | 0 (0.0%) | 84 (2.7%) |

```sql
SELECT SUBSTRING_INDEX(doi,'/',1) p, COUNT(*)
FROM bronze.lit_enrichment_observations WHERE status='success' GROUP BY p;
SELECT SUBSTRING_INDEX(doi,'/',1) p, COUNT(*) FROM gold.lit_articles GROUP BY p;
```

**This corrects `2026-09-22-openalex-crawl.md`.** That note reported 81.9%
annual-count coverage and 91.7% backward coverage as measured results. Both are
accurate as shares of the *observed* population and neither describes the
corpus: the observed population was 98.9% Elsevier. Against the corpus the same
figures are 26.3% and 29.4%, and against IEEE specifically they are zero. The
note's numbers were not wrong; their denominator was unstated, which made them
read as something they were not.

### The fix

`interleave_by_registrant` positions each DOI at `(index + 0.5) / group_size`
within its registrant and sorts by that fraction, so *any* prefix of the result
is proportional. Measured against the real corpus:

| Cut | Elsevier | IEEE | IET |
|---|---|---|---|
| first 500 | 51.8% (corpus 51.7%) | 43.0% (43.0%) | 2.8% (2.7%) |
| first 1,000 | 51.8% | 43.0% | 2.7% |
| first 2,000 | 51.7% | 43.0% | 2.7% |

Deterministic and seedless, so the same corpus always yields the same order —
the reproducibility `NFR-01` requires of anything a published figure rests on.

Both coverage audits now print a per-registrant table against the corpus
denominator. The same run that reports `81.9%` immediately reports:

```
Annual-count coverage by publisher (denominator is the corpus, not the crawl):
  Elsevier     989 /  1612 (61.4%)
  IEEE           0 /  1340 ( 0.0%)   <- not reached
```

An aggregate over a partial collection inherits whatever ordered that
collection. That is now a documented rule (`PRD` §5.5), a recorded decision
(`ADR-07`), and a registered risk (`RISK-09`).

## 2. The hand-built enrichment cache validates

`CLAUDE.md` flags `data/enrichment_cache.json` as "hand-built … not produced by
any code here" — an artifact with no provenance feeding citation counts into the
corpus. The OpenAlex observations make it checkable for the first time.

```sql
SELECT a.doi, a.citation_count, o.citation_count, a.reference_count, o.reference_count
FROM bronze.lit_articles a
JOIN bronze.lit_enrichment_observations o ON o.doi = LOWER(TRIM(a.doi))
WHERE o.status='success' AND a.citation_count IS NOT NULL;
```

Over 1,875 comparable rows:

| Field | Identical | Median error | Mean absolute error |
|---|---:|---:|---:|
| `reference_count` | **99.7%** | 0 | 0.1 |
| `citation_count` | 90.0% | 0 | 0.2 |

Reference counts are the stronger test — a reference list does not change after
publication, so disagreement there would be plain error — and they agree almost
perfectly.

The citation divergences are **one-directional**: the local value is never
higher than OpenAlex's, only lower, in 188 of 1,875 rows. That is the signature
of a snapshot taken earlier in time, not of inaccuracy; citations only
accumulate. The cache is sound, and its drift has a temporal explanation.

Caveat on the denominator: the join multiplies against Bronze's file-qualified
rows, so 1,875 pairs cover fewer distinct DOIs, and every pair comes from the
Elsevier-skewed observed subset described above. The agreement rate is a
property of that subset until the crawl completes.

## Verification

Suite **369 passed, 2 skipped** (361 before); `ruff check` and
`ruff format --check` clean. Eight new tests: the sorted order reproducing the
bias (asserted so the regression stays legible), proportionality at four cut
points, no publisher starved at a small cut, determinism, conservation of every
DOI, single-publisher and empty inputs, and label fallback.

The crawl driver invokes the CLI afresh each round, so the reordering takes
effect on the next quota window without restarting it.

## Residual

The corpus-level figures stay partial until the crawl finishes: 1,000 of 3,115
DOIs observed, forward citation coverage still 0%. What changed is that a
partial crawl is now a representative sample rather than a publisher-complete
block, and that its partiality is stated per source wherever coverage is
reported.
