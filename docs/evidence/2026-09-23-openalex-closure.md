# Evidence — closing WP-07, and measuring WP-23 and WP-24 for real

**Date:** 2026-09-23 · **Branch:** `feat/roadmap-closure`
**Packages:** `WP-07` (complete), `WP-23`, `WP-24`

The first quota window after the fixes of 2026-09-22 ran the batched DOI pass
over the rest of the corpus. That produced the first full-corpus coverage
figures, and three of them turned out to be wrong for reasons that had nothing
to do with the data.

## WP-07 — the refresh is complete

| | |
|---|---|
| Corpus DOIs | 3,115 |
| Resolved by OpenAlex | **3,099 (99.5%)** |
| `not_found` | 16 — OpenAlex holds no record |
| Throttled in window 1, succeeded on retry | 75 |

The batched filter (50 DOIs per request) had only ever run against fixtures;
this was its first live use, and it resolved 2,099 DOIs in one window. The
skip rule honours only `success`, which is why the 75 throttled DOIs were
retried rather than abandoned.

## Three audit figures that described the audit, not the corpus

**1. Backward coverage read 89.5%. It is 100%.** All 324 works without a
reference edge report `reference_count = 0` in their own observation. Their
reference list is known and empty; counting edge presence scored them as
missing. The forward direction had already been fixed for the identical
mistake; this was its mirror image.

**2. Trajectory coverage read 76.7%, failing an 80% gate. It is 98.6%.**
OpenAlex's `counts_by_year` lists only years with at least one citation. Of the
721 works with no series, 677 have never been cited — a trajectory of zeros,
fully known — 44 were cited but published before the series begins, and **none**
are unexplained. The gate passes.

**3. The per-publisher table labelled *observed* works as *covered*.** It was
added the day before specifically so that an aggregate could not mislead, and it
showed Elsevier at 99.4% under "Annual-count coverage" while the aggregate was
76.7%, because it was passed the set of observed DOIs instead of the covered
ones. It now takes named columns, one per measurement, so a heading and its
numbers cannot drift apart again.

The common cause of 1 and 2: a share of zero rows is not a share of zero
evidence. Both audits now count an upstream zero as an answer.

## WP-23 — what is measured, and what is not yet collected

```
Trajectory known:      3055 (98.6%) -- 2378 with a series, 677 never cited
Unexplained gaps:      44
Left-censored:         584 works published before 2012
Complete from publication: 2515 (81.2%)

Reference pairs:       89700
Dated:                 5780 (6.4%); distinct cited works dated 1124/57454

Verdict: trajectories PASS, reference years FAIL (gate 80% each)
```

Two findings the old audit could not have produced:

- **Left-censoring is the real limit on longevity analyses, not coverage.** 584
  works predate the provider's 2012 series start. Their trajectory is present
  but begins mid-life, so Price's index can use them and longevity or Sleeping
  Beauty cannot. The usable population for those two is the 2,515 works whose
  history is complete from publication.
- **Half of WP-23's deliverable had never been collected.** Price's index needs
  the publication year of every cited reference. The corpus cites 57,454
  distinct works and only 1,124 of them are corpus works with a known year;
  56,330 were never resolved. `enrichment resolve-references` now resolves them
  into `lit_reference_works` at 50 per request (~1,130 requests), resumably and
  in hash order — sorted OpenAlex ids track ingestion era, which tracks
  publication year, the very property being measured.

### Validating the years, not just counting them

Coverage says a year exists; it does not say the year is right, and Price's
index is only as good as the years it averages. Two checks need no further
collection and pass today:

| Check | Result | Gate |
|---|---|---|
| Provider year vs corpus metadata, within one year | **98.9%** of 3,099 works | 95% |
| Reference not newer than its citer (+1 year for in-press) | **99.97%** of 5,780 dated pairs — 2 inconsistent | 99% |

Exact agreement is 84.4%; almost all of the gap is the one-year difference
between online-first and issue year, which is why the check tolerates it. 34
works differ by more than a year. Both checks are now part of
`audit citation-years`, printed as a third verdict, and report "n/a" rather
than "PASS" when nothing is comparable.

### Fitting the rest into one window

At 50 ids per request the 56,330 unresolved references cost ~1,130 requests —
more than a window on their own, and two windows once the ~233 forward-crawl
requests go first. OpenAlex documents OR filters of up to 100 values; at 100
the resolution costs ~564, and the two passes together ~797, inside one
window. The 100-value width has not been exercised live, so the resolver
falls back to the proven 50 on the first rejected request and retries the same
chunk rather than stopping. Either way the result is the same data; only the
number of windows differs.

## WP-24 — both gates pass

```
Backward known:    3099 (100.0%) -- 2775 with reference edges, the rest report zero references
Forward crawled:   875 (28.2%), 0 truncated
Graph integrity    Agreement: 100.0% (3314/3314 each way), Verdict: PASS
Access validation  CC/OAPA-licensed and open in OpenAlex: 50/50; same CC licence 95.7% of 46; PASS
Disruption index:  unavailable -- usable coverage 28.2% against an 80% gate
```

- **Graph integrity** cross-checks OpenAlex against itself. Every citation is
  exposed twice — in the citing work's `referenced_works` and in a `cites:`
  query on the cited work — and for a pair whose two ends are both corpus works
  and whose forward crawl is complete, the two must agree. They agree on all
  3,314 checkable pairs, in both directions. That is a trust measure independent
  of any coverage figure.
- **Access validation** uses a genuinely independent source. A CC or OAPA
  licence in the IEEE export implies open access, so each of those 50 articles
  must be open in OpenAlex; all 50 are. An `IEEE`-copyright article that
  OpenAlex finds open is a green repository copy, not a contradiction, and is
  excluded from the test.

The completion rule allows the disruption index to remain unavailable when
coverage is inadequate, so the package's gates are met today. It is left open
until the forward crawl completes, so that integrity is re-checked over the
whole graph rather than 28% of it. OACA stays unavailable regardless: no
confounder-controlled model exists, and the rule says it must not be computed
without one.

## Two collection defects fixed on the way

- **A throttled work was stamped as crawled.** The per-work loop set
  `citing_crawled_at` even when the response was a 429, so the resume skipped
  those works permanently and recorded them as truncated. Five works from
  window 1 were affected; they and four works the old `while … else` had
  over-reported as truncated were reset for recrawl. A throttled or rejected
  batch is now never marked.
- **The forward crawl asked once per work.** 2,254 requests for the 2,215
  works left — more than two windows. OR-joining 50 works into one `cites:`
  query and attributing each citer back through its own `referenced_works`
  makes it 233, and truncation is now decided by OpenAlex's `meta.count`.

## Verification

Suite **382 passed, 2 skipped** (369 before); `ruff check` and
`ruff format --check` clean. Thirteen new tests cover batched citer
attribution, truncation by `meta.count`, a rejected filter failing in one
request, reference resolution skipping corpus works and not storing throttles,
hash ordering, the known-zero semantics in both directions, left-censoring,
reference-year coverage, the cross-index integrity check, and the access
validation's green-OA exclusion. The five forward-crawl tests were moved onto
the batched entry point, and the throttling test now asserts that no throttled
work is marked crawled.

## Residual

A driver (`crawl-driver-v2.sh`) sleeps until the next window — the server's
own `Retry-After` plus ten minutes, no longer a fixed six hours — then runs the
forward crawl (≈233 requests) and the reference resolver (≈1,130), and repeats
until both report nothing left. Two windows are expected. WP-24 closes after
the first, WP-23 after the second; both close on the verdict lines of their
audits, not on a date.
