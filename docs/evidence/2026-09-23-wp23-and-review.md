# Evidence — WP-23 closed through Crossref, and a full-branch review

**Date:** 2026-09-23 · **Branch:** `feat/roadmap-closure`
**Packages:** `WP-23` (complete), `WP-24`, `WP-07`, `WP-08`

## WP-23 — cited-reference years without the OpenAlex quota

The last open gate was cited-reference years: 6.4% dated against 80%, with
56,330 cited works unresolved and the only planned source behind a quota of
1,000 requests per window. Crossref has no per-window quota and a publisher's
reference deposit carries most years inline, so it became the second source.

| | |
|---|---|
| Reference lists requested | 3,115 (all corpus DOIs) |
| Deposited | 2,815 |
| References stored | 101,130 |
| Reference DOIs needing a year lookup | 11,434 — 11,057 found |
| Requests | 385, no throttling, no early stop |

`audit citation-years`, after collection:

```
Trajectory known:      3053 (98.5%)   Left-censored: 584   Complete from publication: 2514
Reference lists:       2815 from Crossref deposits, 49 from OpenAlex
Known empty:           235; unenumerated: 16
Dated references:      93858 / 101716 (92.3%)
Provider year vs corpus metadata: 98.9% within one year
Reference not newer than its citer: 99.99% (6 inconsistent)
Verdict: trajectories PASS, reference years PASS, validation PASS
```

**One list per work, never spliced.** Price's index is a property of a citing
work's own reference list, so each work uses its Crossref deposit when one
exists and its OpenAlex `referenced_works` otherwise. The two do not enumerate
the same thing: they agree on a work's reference count within 10% for only
50.3% of 2,726 works, Crossref larger in 1,694 (median ratio 1.08, p90 2.25),
because a deposit keeps the books, reports and standards that OpenAlex never
resolves to a work. Splicing the lists would count shared references twice;
choosing the fuller one per work is what the ratio argues for.

**Caveats that the aggregate would hide.** IET deposits almost nothing — 16
references across 84 works — so IET falls back to OpenAlex lists whose
references the overnight resolution is still dating; 1 of 84 IET works has
80% of its references dated today. The per-publisher table in the audit
shows it.

## The full-branch review

`code-review` (high) ran over the whole of PR #24 — 142 files, +24k/−8k
lines — and returned ten findings, explicitly unverified. Each was checked
against the code and, where possible, against the live data.

| # | Finding | Verdict | Action |
|---|---|---|---|
| 1 | Version id omits OpenAlex observations, so a refresh can never be published | **Confirmed** | Observations enter the fingerprint (only when present, so ids without enrichment are unchanged) |
| 2 | Airflow's per-stage processes: the start-up sweep marks the live execution crashed | **Confirmed** (my WP-08 change) | Sweep excludes the execution being continued; resuming clears `finished_at` |
| 3 | Legacy first build replaces candidate vectors with live rows | **Confirmed** | Live rows fill gaps only |
| 4 | Legacy re-activation bypass is dead code | **Confirmed** — all 6,235 legacy chunks lack revision and hash | Refused up front with the reason |
| 5 | `cache.update(observations)` lets OpenAlex's 0 erase curated reference counts | **Confirmed, and wider** | Per-field merge; a zero never lowers a positive count |
| 6 | Mutating CLI commands bypass the pipeline lock | **Confirmed** | Duplicate decisions, `versions activate` and `refresh-openalex` take it |
| 7 | Batched breaker counts DOIs, so one failed request stops the crawl | **Confirmed** | Counts failed requests; an exhausted quota still stops at once |
| 8 | `materialize_version` stores JSON `null` in the retired `embedding` column | Plausible, no live reader | Not changed |
| 9 | Code hash includes dashboard code, so a caption edit re-publishes | **Confirmed, but the proposed fix was wrong** | Presentation modules excluded; `analytics.py` kept, because the semantic stage persists its diagnostics |
| 10 | The source archive writes under `data/` | By design (content-addressed retention, `SDD` §3) | Not changed |

**Finding 5 invalidated a figure published the day before.** OpenAlex reports
`reference_count = len(referenced_works)`, and an empty list is sometimes
OpenAlex lacking the list rather than the work citing nothing. Of the 324
works it reported at zero, **81 have references deposited in Crossref** and
18 a positive count in the corpus metadata. The "backward coverage 100%" in
`2026-09-23-openalex-closure.md` treated all 324 as known-empty; the
corroborated figure is **97.1%**. A zero now counts as an answer only when no
source disputes it — which also moved never-cited trajectories from 677 to
675 and the trajectory coverage from 98.6% to 98.5%.

## Verification

Suite **406 passed, 2 skipped**; `ruff check` and `ruff format --check`
clean. New tests pin each confirmed finding: the recovery exclusion, the
per-request breaker (and the immediate stop on an exhausted quota), the
per-field merge, the corroborated zero, the enrichment fingerprint, the
presentation-only code hash, and which CLI commands take the lock.
