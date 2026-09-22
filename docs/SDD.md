# System Design Document — lake-research-map

**Status:** Current architecture plus approved target design, including versioned publication and executable contracts delivered on 2026-09-21

**Evidence cutoff:** 2026-09-21

**Related documents:** [PRD.md](PRD.md), [ROADMAP.md](ROADMAP.md), [METHODOLOGY.md](METHODOLOGY.md)

This document distinguishes facts about the running implementation from approved changes that remain on the roadmap. Statements marked **Current** describe code in the repository. Statements marked **Target** are design decisions whose delivery is tracked in `ROADMAP.md`.

## 1. System context

`lake-research-map` converts manually collected publisher exports into an auditable literature corpus and a read-only analytical application.

```text
IEEE CSV/BibTeX ─┐
Elsevier BibTeX ─┼─> Raw -> Bronze -> Silver -> Gold -> Embed -> Semantic
Local PDFs ──────┘                    │         │                  │
                                     │         └─ curated chunks ┘
OpenAlex ----------------------------┘
                                                │
CLI <-------------------------------------------┤
Airflow ----------------------------------------┤
Streamlit dashboard ----------------------------┘
```

### 1.1 Architectural boundaries

- MySQL contains four databases named `raw`, `bronze`, `silver`, and `gold`.
- Those databases are shared with unrelated projects. This application may access only `lit_*` tables.
- Pipeline stages are Python functions invoked by the same CLI locally and through Airflow.
- Dashboard pages read MySQL and trigger controlled Airflow DAGs; they do not write analytical tables directly.
- Mathematical functions live outside page controllers and must not import Streamlit.
- The current corpus is small enough for single-process pandas/NumPy/scikit-learn execution. Distributed compute is not justified without measured pressure.

## 2. Current component architecture

| Component | Responsibility | Writes |
|---|---|---|
| `ingest/` | Parse publisher files, source configuration, PDF inventory, and enrichment cache | Raw tables and enrichment cache |
| `transform/bronze_articles.py` | Harmonize IEEE and Elsevier records | `bronze.lit_articles` |
| `transform/silver_articles.py` | DOI deduplication, rejection audit, non-article flags, PDF matching | Silver tables |
| `transform/gold_articles.py` | Apply approved merges; build curated articles and reconcile chunks | Gold articles/chunks |
| `transform/embeddings.py` | Model identity, revision check, batch/thread bounds | Nothing; the writer is `versioned_gold.py` |
| `transform/semantics.py` | Contrastive scoring, themes, projection, duplicate candidates | Semantic tables |
| `transform/duplicate_resolution.py` | Validate persistent merge/keep/undo decisions | Duplicate overrides |
| `pipeline.py` | Bootstrap, stage routing, CLI, and run telemetry | Pipeline-run records |
| Airflow DAGs | Schedule/trigger the same CLI commands | Airflow metadata only |
| `dashboard/data.py` | Fault-tolerant SQL reads returning DataFrames | None |
| `dashboard/loaders.py` | Cached normalization, joins, and filtered datasets | Streamlit cache only |
| `dashboard/analytics.py`, `forecasting.py` | Pure analytical computation | None |
| `dashboard/charts.py` | Plotly figure construction and chart contracts | None |
| `dashboard/pages/` | English presentation controllers | None |

## 3. Data architecture and contracts

### 3.1 Contract conventions

Every durable dataset must define:

- **Grain:** what one row represents.
- **Logical key:** the fields that uniquely identify the row.
- **Time semantics:** source event/publication time, ingestion time, and observation time where applicable.
- **Lineage:** input dataset version and producing run.
- **Quality gates:** invariants required before publication.
- **Change behavior:** insert, update, removal, backfill, and rollback rules.

JSON list fields are convenient at current scale but are not substitutes for identity resolution or relational constraints. A future normalized table is justified only when it enables a required query, override workflow, or integrity rule.

### 3.2 Raw database

| Table | Grain and key | Current contract | Known limitation |
|---|---|---|---|
| `lit_source_files` | One active path; unique `path` | SHA-256 plus dataset/revision lineage | Active projection only; immutable history is held by revisions and manifests |
| `lit_config` | One search report; unique `source_file` | Publisher plus free-text query/filter/year/URL provenance | Historical completeness still depends on the manually exported report |
| `lit_ieee_csv_rows` | One active CSV row; `(source_file, row_index)` | Original fields retained as JSON with version/revision lineage | Historical parsed rows are replayed from archived source bytes rather than retained inline |
| `lit_bib_entries` | One parsed entry; `(source, bib_key, source_file)` | Robust BibTeX parsing, including adjacent IEEE entries | Parser output is preserved, but not the exact byte range per entry |
| `lit_pdf_files` | One active file; unique `filename` | Original path, archive path, hash, size, and lineage | Filename uniqueness still assumes a flat PDF directory |

**Current:** every consumed CSV, BibTeX, configuration, enrichment cache, and PDF is retained once in `data/.lake_research_map/objects` by SHA-256. `lit_source_blobs`, `lit_source_revisions`, `lit_dataset_source_files`, and `lit_source_changes` preserve immutable manifests and add/modify/rename/remove events. Raw child rows reference their version and revision. The default append policy retains archived active paths that are absent from a partial download; explicit snapshot mode treats the supplied set as authoritative and removes missing paths and children transactionally.

### 3.3 Bronze database

`bronze.lit_articles` has one harmonized source record, keyed by `(source, source_id)`. It normalizes DOI, authors, keywords, numeric fields, and publication metadata while retaining Raw back-references. IEEE-only metadata is nullable/not applicable for Elsevier.

Current enrichment fills missing citation/reference counts from `data/enrichment_cache.json`; IEEE values take precedence when present. The append-only `lit_enrichment_observations` table and as-of selector govern live OpenAlex refreshes, but the active reference corpus currently contains no persisted provider observations, so citation timing falls back to the documented environment/calendar basis.

**Current contract:**

- Each row references its Raw source revision and output dataset version.
- The default append policy retains archived active sources that are absent from a
  newly downloaded batch; explicit snapshot mode removes absent rows transactionally.
- Citation/reference enrichment is stored as an observation with provider, `observed_at`, work identifier, response status, and optional raw response hash.
- Precedence is deterministic and separately documented for identity fields, descriptive metadata, and time-varying metrics.

### 3.4 Silver database

`silver.lit_articles` has one row per normalized DOI. `silver.lit_rejected` has one rejected Bronze record with its reason and timestamp. Silver is currently delete-and-rebuild derived state.

Required invariants:

- DOI is non-empty, normalized, and unique.
- Every active Bronze row is either represented through `bronze_ids` or present in `lit_rejected`.
- `has_abstract`, `has_pdf`, and `is_non_article` agree with their supporting fields.
- PDF links reference active Raw PDF inventory rows.
- Source-specific fields carry explicit applicability/coverage semantics.

PDF matching currently uses `rapidfuzz.fuzz.token_sort_ratio` with threshold 85. The project guideline that specifies `token_set_ratio` conflicts with the implementation; `WP-05` requires benchmarking both on a reviewed match set before one becomes the canonical contract.

### 3.5 Gold database

| Table | Grain | Current logical key | Required target constraint |
|---|---|---|---|
| `lit_articles` | One curated work after approved merges | `doi` | Unique DOI plus dataset-version lineage |
| `lit_chunks` | One passage | `(doi, chunk_type, seq)` by convention | Database uniqueness, text hash, parent integrity |
| `lit_semantics` | One semantic record per article/run | `doi` | `(dataset_version, semantic_run, doi)` or versioned replacement |
| `lit_duplicate_pairs` | One unresolved unordered pair | Pair by convention | Ordered unique pair and semantic-run lineage |
| `lit_duplicate_overrides` | One reviewed unordered pair | Unique ordered pair | Existing checks plus reviewer/version provenance |
| `lit_pipeline_runs` | One stage attempt | Auto-increment ID | Parent execution, input/output versions, sequence, attempt, metrics, and outcome |
| `lit_dataset_*` snapshots | One Gold output row per version/logical key | Composite version key | Candidate and historical Gold state used for atomic publication and rollback |
| `lit_dataset_versions` | One logical corpus state | Deterministic SHA-256 | Source/config/code/curation hashes and candidate/active/superseded/failed state |
| `lit_quality_results` | One contract result per stage attempt/check | `(stage_run_id, check_id)` | Severity, observed/expected values, details, and pass/fail |

Gold articles are rebuilt, while chunks are reconciled by comparing stored and desired text. Documentation must not call this a “text-hash comparison” until a hash column exists. Unchanged text retains its vector; changed text nulls vector fields; stale chunks are removed.

**Current:** Gold is the canonical input for corpus-level dashboard analyses when its minimum readiness contract passes: the table must be non-empty; DOI must be populated and unique; and the common analytical columns must exist. If that contract fails, the selector falls back to Silver and then Bronze, and every analytical page displays a degraded-mode warning with the reason. This protects approved Gold merges without turning incomplete initialization into a hard dashboard failure.

**Current extension:** the pipeline only materializes a candidate into the live Gold projection after persisted blocking checks pass. Gold dashboard reads resolve through the active publication pointer; the structural readiness check governs only the explicitly labeled Silver/Bronze degraded fallback.

### 3.6 Embedding contract

Current versioned embeddings use `BAAI/bge-small-en-v1.5`, float32 vectors, and binary BLOB storage. Candidate work is selected when the binary vector is absent, its text hash has changed, or its model contract differs. The JSON compatibility copy is no longer written: it roughly doubled chunk storage and dominated the dashboard's vector load (11.6 s to 2.3 s once it was excluded from the query). Readers are binary-only. The column is retained, nullable, because previously published versions carry both representations and stay reactivatable through the binary one; nothing reads the JSON.

The target embedding contract requires:

- `text_sha256`, model name, immutable model revision, vector dimension, dtype, normalization mode, and embedded timestamp.
- Pending means binary vector absent, text hash changed, or model contract incompatible.
- Binary length equals `dimension * 4`; non-finite vectors fail the stage.
- The JSON fallback is retired: it is neither written nor read. The column is retained and nullable so that dropping it stays a separately reviewed migration rather than an unattended one.

### 3.7 Semantic and screening contract

Semantic outputs derive from abstract embeddings. The current stage calculates two anchor similarities, KMeans themes in a shared PCA space, t-SNE coordinates, and near-duplicate pairs.

The current quality gate rejects publication unless all candidate chunks have compatible finite binary embeddings and every eligible abstract has a semantic row. Semantic rows and duplicate candidates are versioned and materialized atomically with articles/chunks. Persisting the full parameter/model-revision manifest remains in `WP-06`.

The current dashboard can export a deterministic four-stratum review sample and ingest one or more long-form CSV files in memory. It normalizes labels, rejects invalid/unknown decisions, preserves raw reviewers, gives explicit adjudication precedence, reports pairwise agreement when support is sufficient, and selects a candidate threshold on a 70% calibration split before evaluating it on a 30% holdout with stratified bootstrap intervals. Fewer than 40 resolved decisions or fewer than 10 observations in either class disables threshold recommendation. The result never changes filters or database state.

Screening labels remain target governed data rather than dashboard session state. A durable label records DOI, reviewer, decision, reason, protocol version, timestamps, dataset version, and assignment. Adjudicated labels and calibrated thresholds remain distinct from raw reviewer labels.

## 4. Data flow, idempotency, and recovery

### 4.1 Current stage behavior

| Stage | Current behavior | Gap |
|---|---|---|
| Raw | Content-addressed scan, immutable manifest/revisions, transactional active reconciliation | Archive garbage collection is not automated |
| Bronze | Transactional rebuild with file-qualified natural keys and dataset lineage; append-only provider observations | The active corpus has no live provider observations yet |
| Silver | Transactional DOI rebuild/rejection audit with dataset lineage | PDF matcher validation remains in `WP-05` |
| Gold | Build immutable candidate article/chunk snapshot; reuse compatible unchanged vectors | Database uniqueness for legacy live chunk keys remains application-enforced |
| Embed | Fill candidate binary/JSON vectors and block incompatible/non-finite output | Immutable model revision/text hash remain in `WP-06` |
| Semantic | Build candidate signals only at complete embedding coverage; publish all Gold outputs atomically | Full semantic-run parameter manifest remains in `WP-06` |

### 4.2 Current versioned run protocol

1. Create or resume a parent execution shared by CLI/Airflow stage attempts under the project-scoped MySQL advisory lock.
2. Create a parent run containing code revision, configuration hash, trigger, and requested stages.
3. Scan inputs into a candidate dataset version and reconcile additions, changes, and removals.
4. Execute each stage in a transaction or staging tables appropriate to its size.
5. Run schema, uniqueness, referential, coverage, range, and business-rule checks.
6. Publish the new version only if required gates pass; otherwise retain the previous active version.
7. Record stage metrics, quality results, lineage, duration, and error details against the parent run.
8. Invalidate dashboard caches after successful publication.

Retry policy is stage-specific and declared per stage in `STAGE_POLICY` (`airflow/dags/lake_research_map_dags.py`). Pure derived stages may retry safely; external enrichment uses bounded retries, timeouts, and durable response status; a failed batch does not masquerade as “not found.” Backfills create new versions rather than rewriting the evidential history.

Abandoned executions are recovered without human intervention. `run()` sweeps executions still marked `running` whenever it actually acquired the MySQL advisory lock, because holding that lock proves no other writer is live and therefore that every surviving `running` row was left by a dead process. Under SQLite the lock is bypassed, the flag is false, and no sweep occurs. The manual `maintenance recover-stale` command shares the same implementation and keeps its age cutoff for operators who want one.

## 5. Additive schema evolution

All unattended bootstrap changes remain additive. New columns are nullable during deployment, backfilled and validated, then enforced through application checks; destructive or non-null migrations require a separately reviewed migration path.

| Proposed dataset | Purpose | Minimum fields |
|---|---|---|
| `gold.lit_dataset_versions` | Identify reproducible corpus states | **Implemented:** version ID, source/config/code/curation hashes, parent, status, timestamps |
| `gold.lit_quality_results` | Persist contract outcomes | **Implemented:** version/run, check ID, severity, observed/expected, pass/fail, details |
| `gold.lit_pipeline_executions` | Correlate stage attempts | **Implemented:** execution/workflow/trigger/version/status/timestamps |
| `gold.lit_publication_state` | Separate published and working state | **Implemented:** active and working version pointers plus publishing execution |
| `raw.lit_source_*` | Preserve source evidence and reconciliation | **Implemented:** blobs, revisions, version manifests, and per-file changes |
| `gold.lit_screening_labels` | Preserve independent review evidence | DOI, reviewer, label, reason, protocol/version, timestamps |
| `gold.lit_screening_models` | Preserve calibrated decisions | dataset/label version, anchors, threshold, metrics/CIs, validation split |
| `bronze.lit_enrichment_observations` | Version time-varying external metadata | DOI, provider, observed time, counts, status, response hash |

Existing tables gain version/run references only when their producer and readers can populate them coherently. No table should carry a decorative lineage column that remains null indefinitely.

## 6. Analytical architecture

### 6.1 Separation of concerns

- `data.py`: parameterized/read-only SQL and graceful absence handling.
- `loaders.py`: the only `@st.cache_data` layer; normalization, compatible joins, and cache bounds.
- `analytics.py` and `forecasting.py`: deterministic pure functions with explicit inputs and outputs.
- `charts.py`: theme-aware figure construction without data access.
- `components.py`: reusable Streamlit presentation and controlled Airflow actions.
- `pages/*.py`: one zero-argument `render()` controller per registered page, as required by this repository's established architecture.

The repository intentionally retains callable `render()` page controllers even though generic Streamlit guidance favors direct page scripts; changing that convention would require coordinated navigation and test refactoring with no current product benefit.

### 6.2 Page ownership

The 10-page ownership matrix in `PRD.md` is a design constraint. Before adding a chart, the implementer records its question, decision, population, metric, and owner page. If another view already answers the question, the new view replaces or becomes a selector alternative to the old one.

High-cost tab bodies use dynamic `st.tabs(..., on_change="rerun")` and render only the open branch. Cached functions must have a freshness or size bound when their key space can grow. Stable UI renders before slow queries; expensive independent panels may use parallel fragments after thread-safety review.

### 6.3 Analytical validity

Method-specific requirements are defined in `PRD.md` and elaborated in `METHODOLOGY.md`. The system enforces common output metadata:

- analytical question and population;
- `n_total`, `n_eligible`, `n_used`, and exclusions;
- dataset/as-of version;
- effect/estimate and uncertainty;
- method diagnostics and warnings;
- exploratory versus confirmatory status.

Hard-coded observation years and cluster-name assignments are transitional. Time comes from dataset metadata, and semantic labels must be derived/reviewed independently of unstable numeric cluster IDs.

### 6.4 Delivered quality and screening interfaces

| Interface | Grain/output | Methodological contract |
|---|---|---|
| Metadata coverage matrix | One field/source pair | Explicit source denominator; empty lists/strings count as absent; source-exclusive fields stay outside the common matrix |
| PDF selection-bias diagnostic | One metric | PDF-minus-non-PDF standardized mean difference; `log1p` for skewed counts; pairwise missing-value exclusion; stratified 95% bootstrap interval with seed 42 |
| Screening review validator | One reviewer/DOI decision | DOI normalization; controlled label vocabulary; duplicate/conflict audit; unknown DOI rejection |
| Consensus resolver | One DOI | Adjudication, then unanimous binary consensus; uncertainty/disagreement excluded from fitting |
| Threshold calibration | One candidate threshold and holdout report | Minimum support, stratified 70/30 split, recall target 0.98, specificity/precision/conservative tie-breaks, held-out metrics and bootstrap intervals |

## 7. Observability and service objectives

### 7.1 Required telemetry

- Parent/stage run status, duration, attempts, and triggering identity.
- Input additions/changes/removals and row counts by layer/source.
- Rejection reasons, duplicate decisions, chunk invalidation, and embedding coverage.
- External enrichment success/not-found/error/rate-limit counts and observation age.
- Data-quality check outcomes and active dataset version.
- Dashboard query/cache latency for expensive loaders and search latency by retrieval mode.

Logs use structured fields with parent run ID, stage run ID, and dataset version. A telemetry-write failure is surfaced; it must not be silently downgraded to debug-only information for a production run.

### 7.2 Initial SLOs

- 100% of successful published versions pass required quality gates.
- 100% of active source files reconcile to Raw manifest revisions.
- 100% abstract embedding compatibility before Semantic publication.
- Zero duplicate normalized DOI in Silver/Gold.
- Zero unreviewed data mutation initiated directly by dashboard page code.
- Full test suite remains green; correctness and Streamlit behavior are measured separately from wall-clock performance because local load dominates elapsed time.

## 8. Security and privacy

- The local single-user deployment relies on the enforced `lit_*` naming boundary in application code. A shared or multi-user deployment must create separate bootstrap, pipeline-write, and dashboard-read roles with database-enforced least privilege before exposure.
- Credentials remain in environment/secrets facilities and are never rendered or included in exported DataFrames.
- SQL identifiers are fixed allowlisted values; user values use parameter binding.
- Airflow's current “all admins” simple-auth setup is acceptable only for an isolated local environment. Binding it to a non-loopback or shared network requires authentication, authorization, TLS/reverse-proxy controls, and removal of unauthenticated admin access.
- Pipeline trigger actions require an audit identity and rate/concurrency controls in any shared deployment.
- PDF text may be copyrighted; access and redistribution remain local and controlled. Retrieval returns short evidence passages and DOI provenance rather than bulk document export.

## 9. Testing strategy

### 9.1 Current baseline

The repository has 343 passing pytest tests plus two opt-in MySQL tests skipped in the default run. They run with isolated in-memory SQLite sessions and additionally cover deterministic fingerprints, content-addressed retention, rename detection, Bronze deletion propagation, isolated Gold candidates, embedding contract failures, atomic materialization, exact Gold reactivation, an end-to-end correlated pipeline fixture, Airflow parent correlation, temporal enrichment observations, persistent human-review evidence, retrieval metrics, provenance coverage, persisted semantic diagnostics, the durable-label-to-calibration join with its approval digests, the injected-fixture citation-graph crawl, automatic abandoned-run recovery, and the plausible-year bound that keeps in-press records dated to next year inside every trend. The MySQL acceptance tests cover JSON/NULL/BLOB round trips, rollback, advisory locks, and the idempotent `lit_config` uniqueness migration. Ruff lint and format checks are required.

SQLite remains the default fast suite. MySQL-specific acceptance is recorded separately against MySQL 8.4 and must be rerun for changes to JSON/NULL behavior, BLOBs, DDL, transactions, or advisory locks.

### 9.2 Target test layers

1. **Unit tests:** pure parsing, normalization, statistics, and edge cases.
2. **Property tests:** idempotency, uniqueness, monotonic cumulative counts, vector dimensions, and deterministic seeded outputs.
3. **Contract tests:** each dataset invariant and quality gate, including deletion propagation.
4. **MySQL integration tests:** migrations, JSON SQL NULL behavior, constraints, transactions, locks, and binary vectors.
5. **Golden-corpus tests:** a small fixture with duplicates, removals, missing DOI, changed text, PDF ambiguity, and partial enrichment.
6. **Statistical validation tests:** simulated known effects/nulls, bootstrap coverage, leakage checks, and baseline comparisons.
7. **Streamlit tests:** `st.testing.v1.AppTest` smoke tests for navigation, filters, degraded states, and conditional heavy tabs.
8. **Browser tests:** only for behavior AppTest cannot simulate, such as rendered theme/CSS or chart selection events.

## 10. Scaling and extension points

Current in-process computation remains the default while measured refresh time, memory, and search latency meet targets. External vector infrastructure, distributed transforms, or materialized analytical tables require an ADR supported by benchmarks and operational cost.

Planned data extensions have explicit prerequisites:

- Citation histories before Sleeping Beauty or longevity analysis.
- Forward/backward citation graph before CD disruption.
- Reference publication years before Price's index.
- Verified access observations and confounders before OACA.
- Persistent author identifiers/overrides before person-level longitudinal claims.

## 11. Architecture decisions

### `ADR-01` — Four physical medallion databases

**Decision:** Retain the current databases and enforce the `lit_*` namespace.

**Alternative:** One database with schemas or one project-specific database.

**Rationale:** Migration risk outweighs current benefit, but shared-database isolation must be treated as a security contract. That contract now rests entirely on the table-name prefix: the least-privilege roles that once enforced it in the database were removed on 2026-09-22 as unjustified for a single-machine deployment (`ROADMAP.md` `WP-08`). Only `lit_`-prefixed tables may be created or touched, and nothing but code discipline enforces it.

### `ADR-02` — Gold is the canonical analytical layer

**Decision:** Dashboard research analyses consume a readiness-validated Gold version.

**Alternative:** Continue preferring Silver for richer quality fields.

**Rationale:** Silver omits approved cross-DOI curation. Needed quality fields should be propagated or joined explicitly, not used to bypass Gold.

### `ADR-03` — Versioned publication, not in-place evidential mutation

**Decision:** Source scans, enrichment observations, labels, and analytical outputs resolve to a dataset/run version.

**Alternative:** Keep only the latest state.

**Rationale:** The SLR and time-varying citation metrics require reproducible historical evidence.

### `ADR-04` — Local single-process compute by default

**Decision:** Retain pandas/NumPy/scikit-learn/ONNX at current scale.

**Alternative:** Spark, a feature platform, or remote vector database.

**Rationale:** They add operational complexity without demonstrated current benefit.

The 2026-09-21 active-corpus benchmark measured an 11.06 MiB vector matrix and a 92.72 ms warm-search P95 after binary-first loading. Retain local vectorized search and reconsider at 512 MiB, 250 ms P95, or a demonstrated concurrency requirement.

### `ADR-05` — Human authority for exclusion and cross-DOI identity

**Decision:** Models prioritize and measure; reviewers decide.

**Alternative:** Automatic semantic exclusion/merge.

**Rationale:** False exclusion and identity errors directly threaten SLR validity.

### `ADR-06` — Read-only dashboard

**Decision:** Data mutations remain in audited CLI/Airflow workflows.

**Alternative:** Direct page writes.

**Rationale:** Controlled workflows provide clearer permissions, recovery, and audit trails.

## 12. Known technical debt

- Analytical reads are already bound to the publication pointer: `dashboard/data.py::load_articles` routes Gold through `load_active_dataset_table`, which scopes every query by `active_dataset_version()`. The residual is narrower than previously recorded -- `assess_gold_articles` is still a structural check (non-empty, unique DOI, expected columns) rather than a read of the persisted `lit_quality_results` for that version.
- Resolved on 2026-09-21: the legacy direct embedding transform and the dashboard action that called it were removed. It wrote the live `lit_chunks` table, which `materialize_version` rebuilds from the candidate on every publish, so vectors generated from the Quality page were discarded at the next publication and never passed `embed_contract`. Embedding now has exactly one writer, `versioned_gold.py::build_dataset_embeddings`, operating on the immutable candidate.
- Parent correlation, dataset versions, cross-process advisory locking, heartbeats, stale-run recovery, and stage-specific Airflow retry/timeout policies are implemented.
- OpenAlex responses are append-only observations with observation time, response state, HTTP status, response hash, retry count, longitudinal counts, outgoing edges, and access metadata. A real refresh still requires local API credentials.
- Chunk and duplicate-candidate logical keys are not fully constrained in the database.
- Several retired analytical functions and inactive page helpers remain in source/tests.
- Expensive tabs are only conditionally rendered on some pages.
- Statistical functions need the validation improvements defined in `PRD.md` before confirmatory interpretation.

The ordered resolution of this debt is defined in `ROADMAP.md`.
