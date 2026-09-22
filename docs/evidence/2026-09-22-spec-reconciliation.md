# Specification reconciliation — 2026-09-22

## Scope

This pass reconciled the implemented system with its dashboard copy, provenance
contract, retrieval-evaluation boundary, temporal citation semantics, semantic-run
evidence, and engineering documents. Follow-up acceptance work ran the Raw stage,
validated the schema contract on the shared MySQL server, and persisted explicitly
AI-assisted draft reviews. It did not call OpenAlex or manufacture human review
evidence.

## Measured baseline

Read-only inspection of the active version `2974743a4b…` found 3,115 Gold
articles, 7,552 chunks, 3,115 semantic rows, 206 duplicate candidates, complete
binary embedding coverage, and zero failed blocking quality checks. After the
follow-up review setup there are 820 assignments and 216 AI-assisted labels: 96
retrieval judgments and 120 taxonomy judgments. Independent human labels remain
absent. The active Bronze database contains no
`lit_enrichment_observations`, so a persisted citation observation year cannot yet
be inferred from provider history.

The current publisher input tree contains 35 BibTeX exports under
`data/references/` (17 IEEE Xplore and 18 ScienceDirect). Of the 3,115 Gold
articles, 3,105 have lineage to at least one current export and 1,584 derive only
from the current export roots. The two local `config.csv` files were initially
TODO templates. They were subsequently restored from preserved evidence: the
ScienceDirect report at `data/config.csv` and the immutable IEEE report whose
object SHA-256 is `943812ff175ff5f368320e08fc555c315a1a4b9ac20a62cc4b668a988e197b22`.

## Delivered reconciliation

- Raw search provenance is keyed by `source_file`, allowing multiple reports from
  one publisher. The quality contract warns when an export directory has no real
  report or still contains a TODO template. The MySQL bootstrap migration is
  additive in data terms: it changes only the uniqueness index and rewrites no
  provenance rows.
- Dashboard copy is English and non-empty across registered pages. A source scan
  test protects that rule and also detects accidentally split sentence literals.
- Dense, BM25, and RRF results expose component ranks and scores. A pure benchmark
  harness computes Recall@k, reciprocal rank, nDCG@k, and latency summaries over
  identical rankings. No retrieval method is declared superior without judgments.
- Citation age and m-quotient calculations require an explicit observation year.
  The loader selects a persisted provider-observation year first, then documented
  environment overrides, and finally the prior calendar year while exposing the
  basis in the UI.
- Semantic stability diagnostics and the cluster-count sweep are persisted on the
  semantic run that produced the map. The dashboard reads those run artifacts
  instead of recomputing present-day caveats for a historical projection.
- Obsolete dashboard-side Raw upload, file-cache OpenAlex enrichment, layer
  selector, serializer, and thematic-centroid paths were removed. The optional
  vector index remains as the intentional scale-triggered WP-25 path.
- PRD, SDD, ROADMAP, CLAUDE guidance, and `.env.example` now describe Gold-first
  reads, append versus snapshot behavior, provenance multiplicity, local security
  boundaries, optional OpenAlex credentials, and embedding controls consistently.

## Verification

The full isolated suite collected 313 tests: **311 passed and two opt-in MySQL
contract tests were skipped**. With `RUN_MYSQL_INTEGRATION=1`, both tests passed
against MySQL 8.4.11 in 0.96 seconds. The acceptance run verified JSON/NULL/BLOB
round trips, rollback, advisory locks, and the `lit_config` migration described
below. Ruff lint and format checks are recorded in the handoff.

The live migration started from two rows, zero duplicate `source_file` values,
and `UNIQUE(source)`. It finished with the same two rows, zero probe rows,
`UNIQUE(source_file)`, and a non-unique `ix_lit_config_source`. Calling the
migration twice was a no-op. Two temporary IEEE rows with distinct source files
were accepted, a repeated source file was rejected, and rollback restored the
original row count.

## Residual evidence and operational risks

- WP-05, WP-09, WP-11, and WP-13 remain blocked on human decisions. Screening
  protocol `v2` now defines the inclusion criteria. One AI-assisted retrieval
  review and one AI-assisted taxonomy review are persisted, but neither substitutes
  for independent human labels. The taxonomy draft contains 108 `present`, 11
  `absent`, and one `ambiguous` decision; its immutable label-set SHA-256 is
  `37f8c9e019ff40f148325021054f66c83f4b1a10c2ff3d27ba1a06b1e975e0a2`.
- The recovered search reports contain the exact query, filters, year range,
  export date, and publisher URL. A Raw append run ingested the two current files
  on 2026-09-22: 2 sources were added, 259 were unchanged, and none were removed.
  `raw.lit_config` now preserves four reports: the historical and current IEEE
  reports plus the legacy and current ScienceDirect reports.
- OpenAlex observation and citation-graph tables remain unevaluated on the active
  corpus until a contact email is configured and a live crawl is authorized.
- End-to-end injected-failure publication and concurrent-writer recovery are
  broader than the low-level transaction/lock contract validated here and remain
  governed by WP-08.
- Citation analyses using the fallback observation year identify that basis, but a
  provider timestamp is the stronger reproducibility contract and remains absent.

## Addendum — second audit pass, 2026-09-22

The reconciliation above was re-audited against the working tree. Four defects
survived it, three of them in exactly the areas it reported as closed. They are
recorded here rather than folded into the sections above, because the sequence
matters: the claims were written before these were true.

- **"Dashboard copy is English and non-empty across registered pages" was not yet
  true.** `pages/researchers.py` carried `"…not como identidade confirmada"` under
  a heading reading `"...Canonicized names, non-identities verified"`, and
  `pages/quality.py` rendered a metric with the label `""` and the delta
  `"todos processados"`. Both are repaired. The scan test missed them because its
  word list held domain nouns only, and the surviving fragment was built entirely
  from Portuguese function words; it now also covers closed-class words, and an
  AST check rejects a `metric_row` entry whose label is empty — `st.metric` takes
  that label positionally, so an empty one renders a number with no name and no
  `st.subheader("")` to find.
- **`theme.apply_dashboard_theme()` was still present and still dead** — 205 lines
  of injected CSS with no caller anywhere, including tests, while
  `.streamlit/config.toml` supplied the actual palette. Removed; `theme.py` no
  longer imports Streamlit at all. `.claude/skills/streamlit-dashboard/SKILL.md`
  documented it as the styling entry point, alongside a light/dark toggle
  (`render_theme_toggle`, `_LIGHT_TOKENS`, `st.session_state["dashboard_theme_mode"]`)
  that no longer exists either — `_active_theme_type()` returns `"dark"`
  unconditionally. That section now describes the fixed dark theme it really has.
- **`analytics.valid_years()` pinned its upper bound at 2026 and silently deleted
  the 11 articles dated 2027** from every trend, production and keyword surface.
  The function's own docstring said the corpus spans to 2027, so the bound
  contradicted the comment beside it. Publishers stamp an in-press record with its
  future issue year, so the guard belongs at next calendar year, not this one;
  `plausible_year_bound()` now decides it per call, and the two call sites that
  passed `hi=2026` explicitly were unpinned. Three regression tests cover the
  in-press year, parse garbage, and an explicit override.
- **`CLAUDE.md`'s measured facts were still the 2026-09-17 corpus** — 1,836 bronze,
  1,831 silver, 5 rejections, 18 duplicate pairs, 96 PDFs. Re-measured: 4,877
  bronze, 3,115 silver, 48 rejections, 206 pairs, 192 PDFs inventoried with 97
  linked. Two corrections beyond the arithmetic:
  - The fields called "IEEE-only" come from the IEEE **CSV** (304 rows), not from
    IEEE generally (1,468 distinct DOIs). Coverage is 301/3,115 for `online_date`
    and `document_type`, 295 for `countries`, 265 for `license` — about 9.7% of
    the corpus and a fifth of its IEEE half, so "the IEEE subset" overstated the
    denominator by 5×.
  - `countries` is JSON whose empty value is `[]`, not `NULL`, so both
    `IS NOT NULL` and `NOT IN ('', '[]')` report all 3,115 rows as populated. Only
    `JSON_LENGTH(countries) > 0` gives the true 295. The dashboard was already
    correct; the risk is to any future SQL that is not.

Bronze-to-silver attrition was also re-examined and is not cross-source: 0 DOIs
appear under both publishers, while Elsevier's 3,289 rows carry 1,647 distinct
DOIs and IEEE's 1,588 carry 1,468. The near-2× Elsevier ratio is overlapping
export runs of the same search collapsing at silver, which is the design working
— but it means the funnel's attrition describes the export overlap, not the
literature, and `CLAUDE.md` now says so.

Suite after this pass: **314 passed, 2 skipped**; `ruff check` and
`ruff format --check` clean over 100 files.
