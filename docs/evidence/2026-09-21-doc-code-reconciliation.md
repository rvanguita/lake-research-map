# Documentation-to-code reconciliation — 2026-09-21

Audit of every `ROADMAP.md` work-package status against the code and the live MySQL databases,
followed by the corrections it justified.

## Environment and commands

- `uv run pytest -q`, `uv run ruff check`, `uv run ruff format --check`.
- An AST sweep over every `dashboard/` module for adjacent string literals with no separating space, and
  for Portuguese function words and accented characters.
- Read-only inventory of `lit_*` tables and row counts across `raw`, `bronze`, `silver`, `gold`.
- Headless render of all 10 pages via `streamlit.testing.v1.AppTest` against the live database.
- `uv run lake-research-map maintenance recover-stale --older-than-minutes 30`.
- `uv run lake-research-map --stage all`.

## Measured baseline, before

| Check | Result |
|---|---|
| `uv run pytest -q` | **2 failed**, 238 passed, 1 skipped, 7.78 s |
| `uv run ruff check` | **1 error** — F821 undefined `res`, `tests/test_advanced_statistics.py:90` |
| Active Gold version | `31a2270b…` — 3,115 articles, 7,552 chunks, 0 blocking gate failures |
| Chunk embedding metadata | **0 of 7,552** rows carried `text_sha256`, `embed_revision`, `embedding_dim`, `embedding_dtype`, `embedding_normalized`, `embedded_at` |
| JSON vector mirror | **7,552 of 7,552** chunks carried it |
| `gold.lit_semantic_runs` | 0 rows |
| Human-evidence tables | `lit_review_protocols`/`_assignments`/`_labels`/`_adjudications`, `lit_model_approvals` — all 0 rows |
| External-data tables | `lit_enrichment_observations`, `_external_works`, `_citation_year_counts`, `_citation_edges`, `_access_observations` — all 0 rows |
| Orphaned public functions | 20 (19 in `analytics.py`, 1 in `forecasting.py`); 17 still had passing tests |
| Stale executions | 5 `running` executions and 5 orphaned `running` embed runs, the oldest from 09:41 |

## Defects found and corrected

1. **The suite was red and the lint gate was failing.** A copy-paste leftover asserted on an undefined
   name, and a trustworthiness bound of 0.95 was unreachable for a fixture whose "projection" truncates
   4 dimensions to 2. Bound lowered to 0.9 with the reason recorded in the test.

2. **False-discovery-rate control was applied to a pre-selected family.** `pages/topics.py` ran
   Mann-Kendall over the 7 most-positive and 7 most-negative OLS slopes and then adjusted those ≤14
   p-values, while the caption claimed FDR control. Correcting a set chosen for being extreme inflates
   significance. Verified on a synthetic 40-keyword fixture: the caption now reads "across all 40
   keywords with at least 15 occurrences; the table shows the 14 terms charted above, with their
   family-adjusted values." Mann-Kendall now runs over every keyword meeting the prevalence rule, Benjamini-Hochberg
   is applied to that whole family, and the table is narrowed to the charted terms afterwards; the
   caption states the family size.

3. **Interval coverage could not fail.** `forecasting.py` set the conformal radius to the 0.9 quantile of
   the rolling-origin errors and then reported the fraction of those same errors the radius covered —
   ≈0.9 by construction. Coverage is now measured on held-out folds and returns `None` when the series
   is too short to spare any. A regression test asserts the abstention.

4. **Stale-run recovery was a no-op.** The predicate required `heartbeat_at IS NOT NULL AND heartbeat_at
   < cutoff`, but an abandoned run usually has no heartbeat at all — and `NULL < cutoff` is `NULL`. The
   command therefore skipped exactly the population it exists to clear. It now falls back to
   `started_at`; the run cleared all 5 stale executions and their 5 orphaned stage rows. Covered by a
   new SQLite test.

5. **One analytical function fabricated data.** `open_access_impact_analysis` substituted
   `hash(doi) % 5 == 0` for real access status whenever fewer than 10 articles looked open access,
   producing a meaningless 20% share that is not even stable across processes (`str` hashing is
   salted per run). `METHODOLOGY.md` already declares OACA unavailable. Deleted with its test.

6. **The English migration had been done by machine translation and was incomplete.**
   - **184 implicit string concatenations** were missing the space between fragments, so captions
     rendered run together ("…straight line.The smooth curvature…"). Detected by tokenising every
     module and comparing adjacent string literals; all 184 repaired, 0 remain.
   - **"Citações" had become "Quotations"** across 7 pages — a plausible-looking word that is simply
     the wrong term.
   - **Five labels were an emoji variation selector with no text** (`"️"`), rendering as blank tab
     and metric captions, and two more metric labels were the empty string, which Streamlit warns about
     and `NFR-07` forbids. One `st.markdown` emitted the literal `"* New vocabulary vs. foundational**"`.
   - **Portuguese remained in 11 files.** Three values were produced in `analytics.py`/`forecasting.py`
     and rendered verbatim (`binomial_negativa`, `crescimento`/`maturidade`, `… (previsto)`); those were
     fixed at the producer, with their consumers and tests updated. Two captions were half-translated
     fragments that had lost their connective text entirely (the CAPES/Qualis matching note and the
     heavy-tail introduction) and were rewritten rather than word-substituted.
   - **"Accumulated" for "cumulative"** in 17 places — another plausible-looking wrong word, on axis
     titles, tab labels and the CCDF chart. The CCDF caption additionally asserted that the log-normal
     "models the literature with greater fidelity", which contradicts the AIC selection shown directly
     above it on this corpus; it now explains how to read the shape and defers to the likelihood
     comparison rather than pre-judging it.
   - Verified by an AST sweep for Portuguese function words and accented characters across every
     dashboard module: the only remaining matches are English docstrings that trip the heuristic.

7. **Twenty public functions were unreachable from any page**, and 17 of them had passing tests, so CI
   reported green on features no user could see. Five methods that `METHODOLOGY.md` already declares
   unavailable and twelve exploratory functions with no owner page were deleted with their tests;
   `tests/test_strategic_analytics.py` had nothing left and was removed. Five orphans were wired into
   their owner pages instead.

8. **The dashboard could write vectors that publication then deleted.** The Quality page had a
   "generate embeddings now" button calling `transform/embeddings.py::build_embeddings`, which wrote the
   live `lit_chunks` table synchronously. `materialize_version` rebuilds that table from the versioned
   candidate on every publish, so the vectors were discarded at the next publication and never passed
   `embed_contract`. The button, `dashboard/actions.py`, and `build_embeddings` were all removed; the
   page now points at the `embed` stage instead. `ADR-06` no longer has an in-process exception, and
   embedding has exactly one writer.

9. **The stability panel was measuring the wrong space, slowly.** The first implementation ran the
   bootstrap in the raw 384-dimensional embedding space and took **62 s per render** with no cache, which
   would have made the Screening page unusable. Themes and the map are both built in the shared PCA(50)
   space (`transform/semantics.reduced_space`), so measuring stability anywhere else answers a different
   question. Moved into that space and cached in `loaders` keyed on `(doi, theme, x, y)` rather than on
   the matrix: **6.2 s cold, cached for 300 s**. Trustworthiness is measured against the coordinates
   actually on screen, so it follows whichever projection the user selected.

   Measured on the active corpus: trustworthiness **0.96**, bootstrap ARI mean **0.73** with a worst
   resample of **0.39** — the theme boundaries are materially less stable than the map's crispness
   suggests, which is exactly the disclosure `WP-17` existed to produce.

10. **Every page was rendered headlessly against the live database** via `AppTest`, before and after each
   change: 10 of 10 render with no exception and no error block. A label sweep over the rendered metrics
   found six blank labels (`""` or a lone variation selector) that `NFR-07` forbids and Streamlit warns
   about; all are now named. Retrieval was re-verified after the binary-only reader change: 7,552 chunks
   load in 2.27 s, hybrid RRF returns in 1.80 s and still exposes `dense_score` and `bm25_score`
   alongside the fused score.

## Work-package status corrections

The roadmap was wrong in both directions. `WP-14` claimed "dashboard integration remains open" for a
panel that already shipped. `WP-15`, `WP-17` and `WP-19` were marked "Open" or `blocked-awaiting-evidence`
for code that was complete and merely unwired — `WP-19`'s degree-preserving null in particular does not
depend on the author-identity work it was blocked behind. `SDD.md` §12 listed two debts that had already
been paid: analytical reads are bound to `lit_publication_state` through `load_active_dataset_table`,
and the legacy embedding transform is binary-first.

## Measured result, after

| Check | Result |
|---|---|
| `uv run pytest -q` | **229 collected, 228 passed, 1 opt-in MySQL test skipped** |
| `uv run ruff check` / `format --check` | clean |
| Orphaned public functions | **0** (re-audited by AST against every page controller) |
| Blank or empty UI labels | **0** across all 10 rendered pages |
| Run-together string joins | **0** (was 184) |
| Pages rendering with no exception | **10 of 10**, against the live database |
| Stale `running` executions / stage rows | **0 / 0** (was 5 / 5) |
| JSON vector mirrors written by the new run | **0** |
| Retrieval after the binary-only reader | 7,552 chunks load in 2.27 s; hybrid RRF 1.80 s, component scores intact |
| Stability panel cost | 6.2 s cold and cached, down from 62 s uncached per render |
| Published version | `2974743a…` replaced `31a2270b…`; 3,115 articles, 7,552 chunks, 3,115 semantic rows |
| Embedding metadata on the active version | **7,552 of 7,552** carry all seven fields (was 0 of 7,552) |
| JSON mirrors on the active version | **0 of 7,552** (was 7,552 of 7,552) |
| `gold.lit_semantic_runs` | **1 row** (was 0) |
| Persisted quality checks | **30, zero failures** — including `embed.text_hash`, which was absent from the previous version's 29 |

Analytical values now surfaced that were previously computed and discarded, or not computed at all:
citation tail α 2.11 over 577 tail articles selected by AIC; GLM condition number 3.0, 11 influential
observations, zero-inflation gap **+9.8%**; Lotka α 2.34 with R² 0.907 over 5,882 authors; projection
trustworthiness 0.96 with bootstrap ARI mean 0.73 and a worst resample of **0.39**; corpus forecast skill
**+71%** against persistence with held-out interval coverage measured over three folds.

## Residual gates

- `WP-05`, `WP-09`–`WP-13` need two reviewers' labels. The schema and the
  `reviews setup|export|import|adjudicate|approve` CLI exist; every table is empty, and the dashboard
  still reads screening labels only from an uploaded CSV, so `gold.lit_review_labels` has no reader.
- `WP-07`, `WP-23`, `WP-24` need `OPENALEX_API_KEY` and `OPENALEX_EMAIL`. `ingest/openalex.py` collects
  only outgoing edges and never creates an `ExternalWork` row for a cited work, so cited-reference years
  stay unreconstructible and Price's index remains unavailable even after a credentialed refresh.
- `WP-08` role provisioning: `db/roles.py` works but is reachable only as `python -m`, not a CLI
  subcommand, and needs four `MYSQL_*` passwords absent from `.env`.
- `WP-15`: no zero-inflated model is fitted; family choice is still the dispersion > 1.5 heuristic. The
  observed zero-inflation gap on the active corpus is +9.8%, so this is not a hypothetical concern.
- `WP-17`: the stability figures are computed on demand and cached for five minutes, not persisted to a
  table, so they are not part of the published version's evidence.
- `WP-18`: every backtest fold is one-step, so the two-year horizon remains unvalidated; no MASE; the
  Bass fit discards its covariance.
- `WP-22`: 7 `st.tabs` call sites remain ungated, `pages/researchers.py:161` being the expensive one.
  AppTest coverage is component-level, not application-level.
