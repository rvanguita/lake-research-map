# Evidence — WP-21's remaining increments, and which were actually blocked

**Date:** 2026-09-22 · **Branch:** `feat/roadmap-closure` · **Package:** `WP-21`

`WP-21` was recorded as blocked on three evidence packages. Two of its seven
deliverable bullets were not blocked on anything.

## 1. Pipeline and provenance — never blocked

`WP-21`'s status listed the increments that had landed and the three waiting on
labels. It never mentioned the **Pipeline and provenance** bullet ("active
version, freshness, quality gates, and removal reconciliation"), two halves of
which had no dependency on any evidence package at all.

`article_population_status()` had returned `dataset_version_id` since `WP-04`
and **no page ever read it** — only `is_canonical` was consumed, for the
degraded-mode banner. Nine of ten pages therefore reported figures that could
not be tied to a snapshot, against `PRD.md` §10's first success criterion:
*every published analytical result identifies a dataset version and population
coverage*.

`loaders.render_population_provenance` now prints, on every page:

```
Dataset `2974743a4b26` · published 2026-09-21 23:19 · 1,204 of 3,115 articles after filters
```

Rendered once inside `require_articles` rather than per page, so a page added
later cannot forget it, and living in `loaders` rather than `components`
because `components` imports `loaders`. The two pages that bypass
`require_articles` were handled explicitly: Trends calls the helper directly,
and Pipeline and provenance is itself the provenance view.

One overlap is deliberate and worth naming rather than claiming away: the
sidebar already showed `1,204/3,115 articles · **gold** layer`. That is live
feedback sitting beside the filter controls and updating as they move; the
page line is the provenance record that makes a screenshot self-describing,
and only it carries the version and publication time. The counts appear in
both. `WP-21`'s "no equivalent view elsewhere" is about analytical charts,
not about chrome restating a total, so the sidebar was left alone.

Where no version is published the caption says so — `Layer \`silver\` (no
published version)` — rather than omitting the line and leaving the reader to
assume one exists. A missing publication timestamp is omitted, never invented.

## 2. Retrieval benchmark — needed a surface, not labels

`dashboard/retrieval_eval.py` (Recall@k, MRR, nDCG@k, latency) and the pooled
judgement sampler both existed, were unit-tested, and were **reachable only
from `tests/`**. The project could compute retrieval metrics and had nowhere to
display them, so the bullet read as blocked on `WP-13` when what was missing
was the page.

The Quality and RAG tab now scores Dense, BM25 and Hybrid RRF over the
versioned query set at depth 10 — the depth the pool was drawn at, since
scoring deeper would count unjudged results as irrelevant and understate every
mode.

**What it refuses to claim matters more than what it computes.** All 96 stored
judgements come from the reviewer id `codex-assisted`: 92 relevant, 3 not
relevant, 1 uncertain across 10 queries. With no independent reviewer present,
the panel prints the table under an explicit **not approved evidence** warning
that names the reviewer and `WP-13`'s requirement for two independent reviews,
and suppresses the approving caption entirely. A test asserts that an AI-only
judgement set produces the warning and no caption, and that independent
reviewers produce the caption and no warning.

## 3. A copy sweep that the automated test cannot do

The language test catches a dropped clause when the fragment before it ends in
`". "` and the next opens lowercase. The migration also dropped clauses
*without* leaving the period, so a fragment ends mid-sentence on a bare word
and the next opens with a capital. Sweeping for that found six real defects:

| Where | What it said |
|---|---|
| `quality.py` | the PDF caption ended mid-phrase on ``the card '`` |
| `quality.py` | the countries caption had shuffled clauses and a missing space (`barsexceeds`) |
| `semantics.py` | the active-learning caption had its clauses transposed into nonsense |
| `pipeline_layers.py` | "the diagnosis below is just one Legacy verification" |
| `forecasting.py` | "traced : continuation predicted by the same chosen model Shows…" |
| `highlights.py` | "high counts are typical of Researches" |
| `topics.py` | ended on the Portuguese "pertence" |

The heuristic is **not** automated, because a fragment legitimately continues
into a proper noun — `"consolidated through the "` + `"Raw → Bronze → Silver →
Gold layers."` — and a test that cries wolf gets disabled. The rule is recorded
in the test's docstring to be re-run by hand after any bulk copy edit.

## Verification

Suite **343 passed, 2 skipped** (334 before); `ruff check` and
`ruff format --check` clean over 104 files. Nine new tests: four on the
provenance caption (version naming, filtered counts, degraded layer, missing
timestamp) and five on the benchmark panel's disclosure.

Judgement parsing checked against the live database: 96 rows, 10 scorable
queries, 92 relevant, single reviewer `codex-assisted` — so the warning path is
the one that renders today.

## Residual

Taxonomy validation and identity audit remain blocked on `WP-12` and `WP-11`,
which are blocked on human labels, which are now exportable in a readable form
(see `2026-09-22-review-queue-context.md`). The retrieval benchmark's numbers
stay unapproved until two independent reviewers replace the AI-assisted pass.
