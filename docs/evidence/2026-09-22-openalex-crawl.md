# Evidence — the OpenAlex crawl, what it measured, and the quota that stopped it

**Date:** 2026-09-22 · **Branch:** `feat/roadmap-closure`
**Packages:** `WP-07`, `WP-23`, `WP-24`

The first live OpenAlex crawl this project has ever run. It produced the first
measured coverage numbers for `WP-23` and `WP-24`, and it was stopped by a rate
limit that turned out to be nothing like what the delay tuning assumed.

## Measured baseline

| | |
|---|---|
| Corpus DOIs in the active version | 3,115 |
| Observations stored | 1,075 — **1,000 success**, 73 `rate_limited`, 2 `not_found` |
| External works resolved | 1,000 (32.1% of the corpus) |
| Backward citation edges | 40,766, all `referenced_works` |
| Annual citation-count rows | 4,554 across 819 works |

`audit citation-years` (`WP-23`'s gate):

```
Observed works:      1000
With annual counts:  819 (81.9%)
Years per work:      min 1, median 4
Year range:          2012-2026
```

`audit citation-graph` (`WP-24`'s gate):

```
Population:          1000
Backward (refs):     917 (91.7%)
Forward crawled:     0 (0.0%), of which 0 truncated
Usable for CD:       0 (0.0%) -- both directions, not truncated
```

> **Correction (same day).** The figures below are shares of the *observed*
> population, which later analysis showed to be 98.9% Elsevier and 0% IEEE --
> the crawl walked DOIs in sorted order and never reached `10.1109`. They are
> accurate as stated and they do **not** describe the corpus. See
> `2026-09-22-sampling-and-cache-validation.md`.

**Both denominators matter.** 81.9% and 91.7% are shares of the *observed*
population, which is itself 32.1% of the corpus. Against the corpus the figures
are 26.3% and 29.4%. `WP-23`'s 80% gate passes on observed works and fails on the
corpus; the audits report the observed population deliberately, because a work
OpenAlex never resolved cannot have a trajectory and mixing the denominators
would make an unrun crawl look like missing data at the provider.

## What actually stopped the crawl

The run halted with `success` frozen at exactly 1,000 while `rate_limited` kept
climbing. The working theory was a burst limit, and the response was to raise
`--delay` from 0.1 to 0.15 and plan a resume at 0.4.

That theory was wrong, and the diagnostic added in this cycle said so on its
first request:

```
Retry-After: 19587          (5.4 hours)
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
```

**It is a quota of 1,000 requests per window, not a rate.** The 1,000 successes
match the limit exactly. `--delay` was never the variable — pacing cannot buy
requests that the quota does not grant, and the resume planned for 0.4s would
have been refused just as fast as 0.1s was. Finishing 3,115 DOIs needs roughly
four windows, and the forward citation crawl needs a further ~1,100 requests on
top.

This also raises a question worth answering before the next window: the polite
pool is documented at 100,000 requests/day, and a `mailto` is being sent both as
a query parameter and in the `User-Agent`. Getting 1,000 suggests this client is
not in the polite pool at all. The header log now makes that checkable rather
than inferable.

## The three client defects this exposed

1. **No circuit breaker.** 73 consecutive 429s and the loop kept going, spending
   four requests and seven seconds of backoff per DOI. Left running it would have
   issued roughly 8,000 futile requests over several hours against an API that had
   already refused. Both passes now abort after five consecutive failures,
   commit what they have, and print how to resume. Five and not one: an isolated
   429 is noise, five in a row is a policy.
2. **`Retry-After` was discarded.** The backoff guessed `min(2**n, 8)` while the
   server was stating the answer exactly — and guessing short is what turns a
   brief throttle into a sustained one. Now honoured, capped at 60s so a header
   cannot park the process.
3. **The forward crawl had no 429 handling at all** — a bare
   `raise_for_status()`, so the first throttled page would have killed a
   thousand-work crawl. It had never been exercised against the live API, which
   is exactly why that went unnoticed. It now retries on the same budget, and a
   work whose crawl was cut short is reported as `throttled`/`truncated` rather
   than as a work with no citations.

4. **Retrying under a multi-hour `Retry-After` was itself futile.** Honouring the
   header exposed a second problem: capped at 60s, each throttled DOI would sleep
   three minutes to be refused three more times, and the circuit breaker would
   take fifteen minutes to fire instead of five requests. When the server names a
   wait beyond the cap, the refusal is now surfaced immediately and the breaker
   ends the batch. Measured live: **5 failures, 2 seconds**, against 73 failures
   and hours before.

`--delay` default raised 0.1 → 0.25 anyway: headroom costs minutes, and the old
value sat exactly on the documented 10 req/s ceiling with none. It is not,
however, the fix for this limit, and the CLI now says so rather than suggesting a
larger delay against a quota.

## Live validation of the fix

Re-running against the still-throttled API, with the quota at zero:

```
openalex refresh: 3115 DOIs, 1000 already observed, 2115 pending, fetching 50
OpenAlex throttled this client (HTTP 429); headers: {'Retry-After': '19411',
  'X-RateLimit-Limit': '1000', 'X-RateLimit-Remaining': '0', ...}
openalex refresh: stopping after 5 consecutive rate_limited responses at 5/50
{'fetched': 5, 'success': 0, 'stopped_early': 'rate_limited', 'remaining': 2115}
```

Three properties confirmed in production rather than only in fixtures: the
resume skipped the 1,000 already observed, the throttle headers were reported
once, and the batch ended after five refusals in about two seconds.

## Verification

Suite **361 passed, 2 skipped** (343 before); `ruff check` and
`ruff format --check` clean. Eighteen new tests, six of them on batching: `Retry-After` preferred, capped and
falling back; the breaker firing at five and *not* at one; committed progress
surviving the break; `remaining` reflecting what still needs fetching; the
forward crawl surviving a throttled page and reporting a persistent throttle as
truncated; a wait beyond the cap skipping the sleep entirely while a short one is still slept through.

The additive migration for `citing_crawled_at` / `citing_truncated` on
`lit_external_works` was applied with the crawl stopped, and re-running bootstrap
is a no-op.

## Batching: the fix the quota actually called for

The quota is counted in *requests*, and the client was spending one per DOI.
Once that is the binding constraint, the answer is not to pace the requests but
to stop making so many: OpenAlex OR-joins up to 50 values in a single filter,
so a DOI population costs `ceil(n/50)` requests instead of `n`.

| | per-DOI | batched |
|---|---:|---:|
| Remaining 2,115 DOIs | 2,115 requests | **43** |
| Plus the citing crawl (~1,100) | ~3,215 → 4 windows, ~22 h | ~1,143 → ~1 window |

`fetch_openalex_batch` returns the same per-DOI result shape as the single
lookup, so persistence, the circuit breaker, the commit cadence and the stats
all keep counting subjects rather than requests, and `batch_size=1` still
routes through the original path. Two details matter for correctness:

- **A DOI the response omits is `not_found`, not an error.** OpenAlex simply
  leaves unknown works out of a filtered result, and calling that an error
  would retry it on every later run forever.
- **A throttled batch fails every DOI in it with one verdict**, which is what
  lets the breaker recognise a throttle as a throttle rather than as fifty
  unrelated failures. It now costs one request to discover a block, not five.

This is also the politer way to ask: the same data for a sixtieth of the load.

## Residual — what the next window has to do

- ~2,115 DOIs unobserved, plus the 73 throttled ones, which the resume retries
  because the skip rule only honours success.
- The forward citation crawl has not run at all, so `WP-24` stays blocked on
  measurement rather than on code: 0% forward coverage is a fact about the
  crawl, not about the corpus.
- Four windows at 1,000 requests each, roughly 24 hours, unless the polite-pool
  question above turns out to have an answer that raises the quota.
