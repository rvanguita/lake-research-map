# System Design Document — lake-research-map

**Status:** Current architecture plus approved target design, including versioned publication and executable contracts delivered on 2026-09-21

**Evidence cutoff:** 2026-11-06

**Related documents:** [PRD.md](PRD.md) (requirements, open work, definition of done)

This document distinguishes facts about the running implementation from approved changes that remain open. Statements marked **Current** describe code in the repository. Statements marked **Target** are design decisions whose delivery is tracked in PRD §12.

## 1. System context

`lake-research-map` converts manually collected publisher exports into an auditable literature corpus and a read-only analytical application.

```text
IEEE CSV/BibTeX ─┐
Elsevier BibTeX ─┼─> Raw -> Bronze -> Silver -> Gold -> Embed -> Semantic
Local PDFs ──────┘                    │         │                  │
                                     │         └─ curated chunks ┘
OpenAlex ----------------------------┘
                                                │
CLI (pipeline, vectors, search, MCP) <-----------┤
Streamlit dashboard (read-only) ----------------┘
```

### 1.1 Architectural boundaries

- MySQL contains four databases named `raw`, `bronze`, `silver`, and `gold`.
- Those databases are shared with unrelated projects. This application may access only `lit_*` tables.
- Pipeline stages are Python functions invoked only through the CLI (`--stage <stage>`, `--stage all`); there is no orchestrator (`ADR-09`).
- Dashboard pages read MySQL and Qdrant and show the CLI commands that run the pipeline; they neither write analytical tables nor start runs.
- Mathematical functions live outside page controllers and must not import Streamlit.
- The current corpus is small enough for single-process pandas/NumPy/scikit-learn execution. Distributed compute is not justified without measured pressure.

## 2. Current component architecture

| Component | Responsibility | Writes |
|---|---|---|
| `ingest/` | Parse publisher files, source configuration, PDF inventory, and enrichment cache | Raw tables and enrichment cache |
| `config.py` | Paths under `data/`, per-layer database names and URLs from `.env` (dataclasses + `python-dotenv`) | Nothing |
| `pipeline.py` | Stage functions, run telemetry, the pipeline lock and the whole CLI (§4.3) | Pipeline-run records |
| `quality.py` | Executable per-stage contracts (`raw_contract` … `semantic_contract`, `publication_contract`), `persist_results`, `assert_contract` | `gold.lit_quality_results` |
| `db/engines.py` | One cached engine and session factory per layer (`get_engine`, `get_session`) | Nothing |
| `db/bootstrap.py` | `CREATE DATABASE IF NOT EXISTS`, `create_all`, the `_ADDITIVE_COLUMNS` migrations and the `lit_config` uniqueness migration | Schema only |
| `db/time.py` | Naive-UTC timestamp helper shared by the models | Nothing |
| `ingest/openalex.py` | Batched, resumable OpenAlex client: DOI refresh, `cites:` crawl, reference-year resolution | Bronze observation tables |
| `ingest/crossref.py` | Quota-free Crossref reference deposits and their years | `lit_crossref_*`, `lit_doi_years` |
| `ingest/enrichment.py` | Load the enrichment cache and observations and merge them (`merge_enrichment`) | Nothing; Bronze applies the result |
| `ingest/reconciliation.py` | Reconcile the active Raw projection with an immutable source snapshot | Raw tables |
| `ingest/snapshots.py` | Content-addressed source archive, manifests, `diff_manifests`, `build_fingerprint` | `data/.lake_research_map/objects`, Raw `lit_source_*` |
| `transform/bronze_articles.py` | Harmonize IEEE and Elsevier records | `bronze.lit_articles` |
| `transform/publication_categories.py` | Deterministic journal/conference/review/other classification with its basis | Nothing |
| `transform/silver_articles.py` | DOI deduplication, rejection audit, non-article flags, PDF matching | Silver tables |
| `transform/gold_articles.py` | Chunking, PDF extraction and merge helpers the Gold builder calls (plus a legacy live-table `build_gold_articles` that only tests call) | Nothing in a pipeline run |
| `transform/versioned_gold.py` | The candidate builders the stages run (`build_dataset_gold`, `build_dataset_embeddings`, `build_dataset_semantics`), `fill_vector_collection`, `materialize_version`, `publish_vector_alias` | `lit_dataset_*`, `lit_semantic_runs`, live Gold on publication, Qdrant |
| `transform/pdf_text.py` | Clean pypdf output before chunking: reference list, page furniture, line-break hyphens, reflow | Nothing; called by the Gold builders |
| `transform/embeddings.py` | Model identity, revision check (by snapshot or byte-identical model files), batch/thread bounds, the one model loader | Nothing; the writer is `versioned_gold.py` |
| `transform/vector_payloads.py` | What each Qdrant point carries: text, provenance, article filter fields, screening signals | Nothing |
| `db/vector_store.py` | The Qdrant access point: one immutable collection per version, exact search, the `lit_chunks_active` alias | Qdrant collections (through `versioned_gold.py` and `vectors` repairs only) |
| `transform/semantics.py` | Contrastive scoring, themes, projection, duplicate candidates (plus a legacy live-table `build_semantics` no stage calls) | Nothing in a pipeline run; `versioned_gold.py` writes |
| `transform/duplicate_resolution.py` | Validate persistent merge/keep/undo decisions | Duplicate overrides |
| `transform/review_workflows.py` | Protocols, dual assignments, append-only labels, adjudication, approvals | `lit_review_*`, `lit_model_approvals` |
| `transform/screening_calibration.py` | Label validation, consensus, agreement, threshold calibration and its digests | Nothing |
| `transform/evidence_samples.py` | Stratified subjects for the `evidence` commands (PDF, rejections, authors, taxonomy, retrieval) | Nothing; `reviews setup` persists them |
| `apps/qdrant/cli.py` | The `vectors status|backfill|rebuild|sync-alias` commands; repairs write only Qdrant | Qdrant collections and the alias (never Gold rows) |
| `apps/api/` | Article-level search, screening and evidence (`Retriever` in `api.py`); imports neither Streamlit nor `apps/dashboard/` | None (reads the published Gold version and its Qdrant collection) |
| `apps/api/index.py` | The published version's articles and chunk texts, loaded once per process | None |
| `apps/api/filters.py` | `ArticleFilters`, one definition as a pandas mask and a Qdrant filter | None |
| `apps/api/rank.py` | Pure BM25 index, RRF fusion, per-article collapse, excerpts | None |
| `apps/api/models.py` | Pinned optional cross-encoder and the untruncating token counter | None |
| `apps/api/cli.py`, `mcp_server.py` | The `search`/`screen`/`mcp` commands and the five MCP tools (stdio) | None |
| `apps/dashboard/data.py` | Fault-tolerant SQL reads returning DataFrames | None |
| `apps/dashboard/loaders.py` | Cached normalization, joins, and filtered datasets | Streamlit cache only |
| `apps/dashboard/analytics.py`, `forecasting.py` | Pure analytical computation | None |
| `apps/dashboard/qualis.py` | Pure CAPES/Qualis venue lookup (ENGENHARIAS IV) | None |
| `apps/dashboard/retrieval_eval.py` | Pure ranking metrics and per-mode latency measurement for the retrieval benchmark | None |
| `apps/dashboard/charts.py` | Plotly figure construction and chart contracts | None |
| `apps/dashboard/pages/` | English presentation controllers | None |
| `devtools/` | `docs_branch_guard.py` (pre-commit hook refusing docs-only commits on `main`) and `hero.py` (README illustration generator) | Nothing at runtime |

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

`bronze.lit_articles` has one harmonized source record, keyed by `(source, source_id)`; `source_id` is file-qualified (`csv:<file token>:<row>`, `bib:<file token>:<key>`), and the stage deletes and rebuilds the table on every run. It normalizes DOI, authors, keywords, numeric fields, and publication metadata while retaining Raw back-references. IEEE-only metadata is nullable/not applicable for Elsevier.

Current enrichment fills missing citation/reference counts after the rebuild (`bronze_articles._enrich_citation_counts`); IEEE CSV values take precedence when present. The fill comes from `ingest/enrichment.py::merge_enrichment`, which combines `data/enrichment_cache.json` with the append-only `lit_enrichment_observations` -- the 2026-09-23 OpenAlex refresh persisted observations for 3,099 of the 3,115 DOIs (§14) -- and never lets an observed zero overwrite a positive curated count.

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

PDF matching currently uses `rapidfuzz.fuzz.token_sort_ratio` with threshold 85 (`silver_articles.PDF_MATCH_THRESHOLD`). No project guideline prescribes another scorer; `WP-05` requires benchmarking it against alternatives (`token_set_ratio` among them) on a reviewed match set before the scorer and threshold become the canonical contract.

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

The Gold stage deletes and rebuilds the candidate version's article and chunk rows (and clears its semantic rows); stored and desired chunk text are compared only to report `chunks_reused`/`chunks_new`/`chunks_invalidated`, and every candidate chunk row records the SHA-256 of its text. Vectors are not carried by the Gold rebuild: they live in Qdrant (ADR-08), and the embed stage reuses any stored vector whose text hash and model revision match, so unchanged text is never re-embedded and changed text always is.

**Current:** Gold is the canonical input for corpus-level dashboard analyses when its minimum readiness contract passes: the table must be non-empty; DOI must be populated and unique; and the common analytical columns must exist. If that contract fails, the selector falls back to Silver and then Bronze, and every analytical page displays a degraded-mode warning with the reason. This protects approved Gold merges without turning incomplete initialization into a hard dashboard failure.

**Current extension:** the pipeline only materializes a candidate into the live Gold projection after persisted blocking checks pass. Gold dashboard reads resolve through the active publication pointer; the structural readiness check governs only the explicitly labeled Silver/Bronze degraded fallback.

### 3.6 Embedding contract

Embeddings use `BAAI/bge-small-en-v1.5` (384-dimension float32, unit norm) and are stored in **Qdrant**, one collection per dataset version (`lit_chunks_<version_id>`, named dense vector `dense`, cosine distance). Point ids are a UUIDv5 of `(doi, chunk_type, seq)`, so re-runs upsert rather than duplicate. Gold keeps the evidence of what each vector must be: `text_sha256`, model name, immutable model revision, dimension, dtype, normalization flag and embedded timestamp on every chunk row. No MySQL vector is written. The `embedding_bin` BLOBs and the retired JSON mirror stay, nullable, on the versions published before ADR-08 -- those snapshots are immutable -- and serve only as a copy source for `vectors backfill` and the first post-migration candidate.

- **Pending** means the chunk's point is missing or carries another text hash or model revision, or the Gold provenance is missing.
- **Reuse before compute:** reusable vectors are resolved in bulk before any batch -- the candidate's own collection (a killed run), the active version's, then qualifying BLOBs of the candidate or active version -- and only text no store has seen reaches the model.
- **Ordering:** each batch is upserted (and waited for) before Gold records provenance and commits, so the two stores cannot disagree in the direction that loses work.
- **Contract (`quality.embed_contract`):** every Gold chunk has a point with the pinned model and revision on both sides, the right dimension, a finite unit vector and a text hash equal to the row's current text; points without a chunk (orphans) fail; an unreachable store fails. Publication re-checks it, so the pointer never moves to a version with missing or stale vectors.
- **Model pin:** a snapshot other than the pinned revision is accepted only when its five model-defining files hash to `EMBED_MODEL_CONTENT_SHA256` (a README-only Hub commit on 2026-09-24 made the commit-hash-only check refuse an identical model).

### 3.7 Semantic and screening contract

Semantic outputs derive from abstract embeddings. The current stage calculates two anchor similarities, KMeans themes in a shared PCA space, t-SNE coordinates, and near-duplicate pairs.

The current quality gate rejects publication unless every candidate chunk has a compatible finite vector in the version's Qdrant collection and every eligible abstract has a semantic row. The stage reads the abstract matrix from Qdrant and writes the signals back onto the points as payload (including `relevance_margin`), so a retrieval client can filter by screening margin server-side. Semantic rows and duplicate candidates are versioned and materialized atomically with articles/chunks. Each semantic run persists its manifest in `lit_semantic_runs` -- model, revision, dimension, article count, anchor hashes and the stability diagnostics (`WP-06`, delivered); the clustering and projection constants in `transform/semantics.py` are covered by the code fingerprint rather than stored per run.

The current dashboard can export a deterministic four-stratum review sample and ingest one or more long-form CSV files in memory. It normalizes labels, rejects invalid/unknown decisions, preserves raw reviewers, gives explicit adjudication precedence, reports pairwise agreement when support is sufficient, and selects a candidate threshold on a 70% calibration split before evaluating it on a 30% holdout with stratified bootstrap intervals. Fewer than 40 resolved decisions or fewer than 10 observations in either class disables threshold recommendation. The result never changes filters or database state.

**Current:** screening labels are durable governed data, not dashboard session state. `reviews setup` stores a versioned protocol (`lit_review_protocols`) and deterministic dual assignments (`lit_review_assignments`: workflow, dataset version, protocol, subject, reviewer, stratum); `reviews import` appends raw labels with rationale and payload hash (`lit_review_labels`, never overwritten); `reviews adjudicate` records a final label separately (`lit_review_adjudications`); `reviews calibrate` evaluates the threshold against the stored labels and `reviews approve` persists a hash-bound decision (`lit_model_approvals`). The machinery is delivered; no screening label has been imported yet (PRD §12, `WP-09`).

## 4. Data flow, idempotency, and recovery

### 4.1 Current stage behavior

| Stage | Current behavior | Gap |
|---|---|---|
| Raw | Content-addressed scan, immutable manifest/revisions, transactional active reconciliation | Archive garbage collection is not automated |
| Bronze | Transactional rebuild with file-qualified natural keys and dataset lineage; append-only provider observations | None open: the 2026-09-23 OpenAlex refresh observed 3,099 of 3,115 DOIs |
| Silver | Transactional DOI rebuild/rejection audit with dataset lineage | PDF matcher validation remains in `WP-05` |
| Gold | Build immutable candidate article/chunk snapshot with text hashes | Database uniqueness for legacy live chunk keys remains application-enforced |
| Embed | Fill the version's Qdrant collection (reuse by text hash before compute), record provenance in Gold, delete orphan points, block incompatible/non-finite/unreachable output | `text_sha256` has no index on the deployed schema; lookups go through `dataset_version_id` |
| Semantic | Build candidate signals only at complete embedding coverage; publish all Gold outputs atomically | None open: each run records model, revision, anchor hashes and stability in `lit_semantic_runs` |

### 4.2 Current versioned run protocol

1. Create or resume a parent execution shared by the stage attempts of one CLI workflow (`--stage all`, or stage-by-stage with `--execution-id`/`--workflow`) under the project-scoped MySQL advisory lock.
2. Create a parent run containing code revision, configuration hash, trigger, and requested stages.
3. Scan inputs into a candidate dataset version and reconcile additions, changes, and removals.
4. Execute each stage in a transaction or staging tables appropriate to its size.
5. Run schema, uniqueness, referential, coverage, range, and business-rule checks.
6. Publish the new version only if required gates pass; otherwise retain the previous active version.
7. Record stage metrics, quality results, lineage, duration, and error details against the parent run.
8. **Target:** invalidate dashboard caches after successful publication. **Current:** nothing invalidates them; cached loaders expire on their TTLs (10 s to 5 min, reference-data caches never) and the Pipeline page's *Refresh data* button clears them by hand (§12).

There is no automatic stage retry: the per-stage retry and timeout policy lived in the Airflow DAGs and left with them (`ADR-09`). A failed stage is re-run by hand -- every stage is idempotent and `embed` resumes from its own collection -- and the next run's start-up sweep recovers the abandoned execution. External enrichment keeps its own bounded retries, timeouts, and durable response status; a failed batch does not masquerade as “not found.” Backfills create new versions rather than rewriting the evidential history.

Abandoned executions are recovered without human intervention. `run()` sweeps executions still marked `running` whenever it actually acquired the MySQL advisory lock, because holding that lock proves no other writer is live and therefore that every surviving `running` row was left by a dead process. Under SQLite the lock is bypassed, the flag is false, and no sweep occurs. The manual `maintenance recover-stale` command shares the same implementation and keeps its age cutoff for operators who want one.

### 4.3 Command-line surface

**Current:** `lake-research-map` (`pipeline:main`) is the only entry point. "Lock" means the command takes the
project's MySQL advisory lock (`pipeline._pipeline_lock`), so it cannot overlap a stage run; the rule is
`pipeline._mutates_published_inputs` -- a command takes it when it changes what the next publication would contain.

| Command | What it does | Lock |
|---|---|---|
| `--stage raw\|bronze\|silver\|gold\|embed\|semantic\|all` | Run one stage, or the chain (`all`, the default) | Yes |
| `--execution-id ID --workflow STAGE` | Run one stage of a workflow under a shared execution, so separate processes record one execution | Yes |
| `--source-policy append\|snapshot` | Raw scanning: keep archived sources absent from disk (`append`, default) or mirror the files on disk | (with the stage) |
| `duplicates list` | Unresolved cross-DOI candidates | No |
| `duplicates merge\|keep\|undo` | Record or remove a persistent duplicate decision | Yes |
| `versions list` | Dataset versions and publication status | No |
| `versions activate` | Reactivate a validated Gold snapshot | Yes |
| `reviews setup` | Create a protocol and dual-review assignments | No |
| `reviews export` / `import` | Export one reviewer's assignments; import append-only labels | No |
| `reviews adjudicate` | Record a final adjudication | No |
| `reviews calibrate` | Evaluate the screening threshold against stored labels | No |
| `reviews approve` | Persist an evaluated model decision | No |
| `reviews status` | Labelling progress, agreement, disagreement queue | No |
| `enrichment refresh-openalex` | Refresh the active DOI population's OpenAlex observations | Yes |
| `enrichment refresh-citations` | Crawl incoming citation edges (`cites:`) for observed works | No (feeds audits only) |
| `enrichment crossref-references` | Collect Crossref reference deposits and date their references | No (feeds audits only) |
| `enrichment resolve-references` | Record the publication year of every cited work | No (feeds audits only) |
| `evidence pdf-matches\|rejections\|authors\|taxonomy\|retrieval` | Stratified candidate sets for human review (PDF links around the threshold, silver rejections, author-name pairs, taxonomy classes plus an unclassified stratum, pooled retrieval results) | No |
| `audit reference-corpus\|citation-graph\|citation-years` | Read-only acceptance audits (§7.2) | No |
| `maintenance recover-stale` | Fail executions still `running` past an age cutoff | Yes (taken in its handler) |
| `vectors status` | Compare each version's collection with its Gold chunks and the alias | No |
| `vectors backfill\|rebuild\|sync-alias` | Copy pre-ADR-08 BLOBs, restore a collection from Gold text, re-point `lit_chunks_active`; write only Qdrant | Yes |
| `search` / `screen` | Article-level ranking for a query; score every abstract against a subject (`--json` for one JSON document) | No |
| `mcp` | The same operations as five MCP tools over stdio | No (a long-running server must never block a run) |

The `reviews` commands write only the review tables, which no stage reads, so they run unlocked.

## 5. Additive schema evolution

All unattended bootstrap changes remain additive. New columns are nullable during deployment, backfilled and validated, then enforced through application checks; destructive or non-null migrations require a separately reviewed migration path.

Every dataset below is **implemented** (`db/*_models.py`). Two rows of the original proposal never shipped
under their proposed names: `gold.lit_screening_labels` became the review tables (`lit_review_*`), which serve
every human-evidence workflow rather than screening alone, and `gold.lit_screening_models` became
`lit_model_approvals`.

| Dataset | Purpose | Key fields |
|---|---|---|
| `gold.lit_dataset_versions` | Identify reproducible corpus states | Version ID, source/config/code/curation hashes, parent, status, timestamps |
| `gold.lit_quality_results` | Persist contract outcomes | Version/run, check ID, severity, observed/expected, pass/fail, details |
| `gold.lit_pipeline_executions` | Correlate stage attempts | Execution/workflow/trigger/version/status/timestamps |
| `gold.lit_publication_state` | Separate published and working state | Active and working version pointers plus publishing execution |
| `gold.lit_dataset_articles` | Immutable Gold article snapshot per version | `(dataset_version_id, doi)` unique; the curated article columns |
| `gold.lit_dataset_chunks` | Immutable chunk snapshot per version, with embedding provenance | `(dataset_version_id, doi, chunk_type, seq)` unique; text, `text_sha256` (unindexed), model/revision/dimension/dtype/normalized/embedded-at; legacy `embedding_bin` |
| `gold.lit_dataset_semantics` | Per-article screening signals and map position per version | `(dataset_version_id, doi)` unique; relevance/off-topic scores, theme, `map_x`/`map_y` |
| `gold.lit_dataset_duplicate_pairs` | Near-duplicate candidates per version | `(dataset_version_id, doi_a, doi_b)` unique, `doi_a < doi_b`, similarity |
| `gold.lit_semantic_runs` | Manifest of one semantic derivation | Model, revision, dimension, article count, anchor hashes, stability diagnostics |
| `gold.lit_review_protocols` | Versioned reviewer instructions (SDD Appendix A) | Workflow, protocol version, instruction text and SHA-256 |
| `gold.lit_review_assignments` | Deterministic dual-review assignment | Workflow, dataset version, protocol, subject, reviewer, stratum |
| `gold.lit_review_labels` | Append-only raw human label (replaces the proposed `lit_screening_labels`) | Assignment, label revision, label, rationale, payload hash, import time |
| `gold.lit_review_adjudications` | Final label after disagreement, kept apart from raw labels | Workflow, dataset version, subject, final label, adjudicator, rationale |
| `gold.lit_model_approvals` | Hash-bound approval of a measured threshold or retrieval default (replaces the proposed `lit_screening_models`) | Workflow, dataset version, model hash, parameters, metrics, label-set hash, approved/by/at |
| `raw.lit_source_*` | Preserve source evidence and reconciliation | Blobs, revisions, version manifests, and per-file changes |
| `bronze.lit_enrichment_observations` | Version time-varying external metadata (current) | DOI, provider, provider work id, observed time, status/HTTP status, counts, response hash, retry count, payload |
| `bronze.lit_external_works` | Corpus works as OpenAlex identifies them, with forward-crawl state | Provider work id, DOI, year, `citing_crawled_at`, `citing_truncated` |
| `bronze.lit_reference_works` | Publication year of cited works outside the corpus | Provider work id, year, status, observed time |
| `bronze.lit_crossref_reference_lists` | Whether Crossref holds a reference deposit for a corpus DOI | DOI, observed time, status, deposited count |
| `bronze.lit_crossref_references` | One deposited reference of a citing work | Citing DOI, position, reference DOI, inline year, observed time |
| `bronze.lit_doi_years` | Year looked up for a referenced DOI the deposit left undated | Provider, DOI, year, status, observed time |
| `bronze.lit_citation_year_counts` | Annual citation trajectory per work | Provider work id, year, count, observed time |
| `bronze.lit_citation_edges` | Observed citation edges | Citing/cited work ids, observed time, `discovered_via` (`referenced_works` vs `cites:`) |
| `bronze.lit_access_observations` | Open-access status observations | Provider work id, observed time, `is_oa`, OA status, licence, landing page |

Existing tables gain version/run references only when their producer and readers can populate them coherently. No table should carry a decorative lineage column that remains null indefinitely.

## 6. Analytical architecture

### 6.1 Separation of concerns

- `data.py`: parameterized/read-only SQL and graceful absence handling.
- `loaders.py`: the only `@st.cache_data` layer; normalization, compatible joins, and cache bounds.
- `analytics.py` and `forecasting.py`: deterministic pure functions with explicit inputs and outputs.
- `charts.py`: theme-aware figure construction without data access.
- `components.py`: reusable Streamlit presentation, including the read-only panel of pipeline commands.
- `pages/*.py`: one zero-argument `render()` controller per registered page, as required by this repository's established architecture.

The repository intentionally retains callable `render()` page controllers even though generic Streamlit guidance favors direct page scripts; changing that convention would require coordinated navigation and test refactoring with no current product benefit.

### 6.2 Page ownership

The 10-page ownership matrix in `PRD.md` is a design constraint. Before adding a chart, the implementer records its question, decision, population, metric, and owner page. If another view already answers the question, the new view replaces or becomes a selector alternative to the old one.

High-cost tab bodies use dynamic `st.tabs(..., on_change="rerun")` and render only the open branch. Cached functions must have a freshness or size bound when their key space can grow. Stable UI renders before slow queries; expensive independent panels may use parallel fragments after thread-safety review.

### 6.3 Analytical validity

Method-specific requirements are defined in `PRD.md` and the methods themselves in §6.5. The system enforces common output metadata:

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

### 6.5 Analytical methods

What the dashboard can infer from the current corpus, how each model is validated, and which formal indicators are intentionally unavailable.

This document defines what the dashboard can infer from the current corpus, how each model is validated, and which formal indicators are intentionally unavailable.

#### Evidence boundaries

The analytical unit is an article in the active, immutable Gold version. Publication metadata comes from IEEE Xplore and Elsevier exports; citation and reference data may be enriched from append-only OpenAlex observations. The schema now retains annual citation counts, outgoing reference edges, publication years, and access observations, but the active corpus has not yet passed the external-data coverage and confounding gates. These fields therefore remain unavailable as analytical evidence until a credentialed refresh and audit succeed.

Missing numeric values remain missing. Analytical functions use complete cases and report `n_total`, `n_used`, and coverage whenever a filtered sample can change interpretation. IEEE-only fields such as country, online date, document type, and license must display source coverage and cannot support corpus-wide causal comparisons.

The following indicators are unavailable until their required data is ingested:

- Price's index: requires publication years for every cited reference.
- Sleeping Beauty and beauty coefficient: require annual citation trajectories.
- CD disruption index: requires a forward and backward citation graph.
- Open Access Citation Advantage: requires verified access status and confounder controls.
- Citation half-life or longevity: requires citations indexed by citing year.

The dashboard does not estimate substitutes under those names, and as of 2026-09-21 it no longer carries
the code either. Implementations of all five existed but were unreachable from any page, and the
open-access one substituted a hash of the DOI for real access status whenever too few articles looked
open access -- inventing a 20% share that was not even stable between processes. They were deleted
rather than left dormant: an unreachable estimator that fabricates its input is one page wiring away
from becoming a published result.

#### Citation count model

`citation_determinants_glm` models cumulative citations as counts. The exposure offset is `log(article_age + 1)`, which accounts for older papers having had longer to accumulate citations. Reference count and team size are standardized, source is an indicator, and rows with missing predictors are excluded. Predictors with no variation in the selected sample are omitted.

Four count families are fitted -- Poisson, negative binomial, and their zero-inflated counterparts -- and the family with the lowest AIC is reported. The zero-inflation part is intercept-only, because with three predictors and a few hundred complete rows a fully specified inflation equation is not identifiable. A candidate whose Hessian could not be inverted is refused outright rather than selected: it has point estimates but no standard errors, and a family that cannot state its own uncertainty must not win on AIC. When no zero-inflated candidate converges the panel says so. All candidates report HC3 robust confidence intervals, p-values, incidence-rate ratios, sample coverage, and the selected family.

The same panel reports the misspecification evidence rather than leaving it implicit: the missingness profile of each predictor, the design condition number and per-predictor variance inflation factors, the count of observations above the 4/n Cook's-distance cut, and the gap between the observed and model-predicted fraction of zero-citation articles. Cook's distance needs a hat matrix that a zero-inflated maximum-likelihood fit does not have, so influence is always measured on the Poisson GLM and the panel names which fit produced it rather than reporting a failed inversion as a diagnostic.

The exposure term is reported as a sensitivity rather than assumed. `log(age + 1)` as a fixed-coefficient offset forces citations to accumulate exactly proportionally to log age, so the model is refit with log age and with linear age as free covariates, and a predictor whose sign changes between the three specifications raises an explicit warning instead of being reported as an association. Results describe conditional associations in this corpus and do not establish causal effects; the source indicator in particular also encodes source-specific missingness.

#### Citation distributions

Heavy-tail fitting counts zero-citation articles explicitly and then fits above a data-driven `x_min`, chosen by sweeping candidate quantiles and minimising the KS distance subject to at least ten tail observations. Power-law, log-normal and exponential families are compared by AIC, not by KS distance alone, because KS does not penalise the extra free parameter. The reported goodness-of-fit p-value for the power law comes from a parametric bootstrap that refits the exponent on each simulated sample; the log-normal and exponential p-values reuse their own fitted parameters and are therefore optimistic. The panel labels which is which instead of presenting one number. Log-likelihood ratios between families are shown without a Vuong test and are labelled descriptive.

#### Forecasts

Publication-volume forecasts compare a persistence baseline, linear trend, and log-linear trend. Selection uses rolling-origin validation over complete historical years. A partial current year may be shown as monitoring evidence but never participates in selection or final fitting.

Every forecast reports its skill against persistence: `1 - MAE(model) / MAE(last complete year repeated)` over the rolling-origin folds. A model that does not beat the naive baseline is labelled as such rather than presented as a projection on equal footing.

Uncertainty bands use the empirical 90th percentile of rolling one-step absolute errors, and the radius expands with the square root of the forecast horizon. Coverage of those bands is measured on folds held out of the radius calibration; scoring them on the same errors that set the radius returns the nominal level by construction and cannot fail. When the series is too short to hold folds back, no coverage is reported at all. The holdout is only three folds on an annual series, so the figure is coarse and the panel states the fold count rather than presenting a bare percentage.

Each forecast year is backtested at its own horizon: a two-step fold trains only on years at least two steps before its target, so a two-year claim never borrows one-year information. Where the series is too short to hold folds back at a horizon the panel prints "not testable" rather than reusing the shorter horizon's number. Alongside skill against persistence, MASE divides the error by the mean absolute one-step change, which is what makes a sparse keyword series and the corpus series comparable at all; below 1 beats a naive carry-forward.

These are corpus-volume projections, not forecasts of all publications in the field.

Bass diffusion is displayed only when nonlinear least squares converges to finite, admissible parameters and the data provide enough observations. Failed fits remain unavailable; no synthetic fallback curve is presented as an estimate. The fit's covariance is retained rather than discarded, giving per-parameter standard errors and an 80% interval for the peak year from a parametric bootstrap over that covariance -- the peak is a non-linear function of the innovation and imitation coefficients, so its spread cannot be read off their standard errors directly.

#### Concept bursts

Concept bursts use a two-state implementation of Kleinberg's model. For every year, the observation is the number of matching documents out of all corpus documents in that year. Dynamic programming chooses between a baseline binomial state and an elevated state. Entering the elevated state incurs a transition penalty. Consecutive elevated years form a burst interval; strength is the interval's improvement in binomial log likelihood over the baseline state.

The concept dictionary is declared in code, so results detect changes for those named concepts rather than discovering unrestricted vocabulary.

#### Semantic analyses

Contrastive screening uses the difference between similarity to an in-scope anchor and an off-topic logistics anchor. A zero margin is a semantic decision boundary, not a measured-recall guarantee. Calibration metrics are valid only against reviewed labels.

PCA and t-SNE projections are visualization aids and do not preserve every high-dimensional distance. A projection is labeled by the algorithm actually used; a failed UMAP computation must not silently appear as UMAP. Semantic isolation is a nearest-neighbor distance and must not be described as interdisciplinarity or conceptual novelty without external validation.

The map states how much of itself is structure. A bootstrap adjusted Rand index over subsamples and seeds measures whether the same articles keep grouping together when the sample changes. Three neighbourhood measures sit beside it, because trustworthiness alone only penalises neighbours the projection invents: a layout that tears one real cluster in two scores well on it while being badly wrong. Continuity is the same measure with the spaces swapped and penalises neighbours the projection loses, and kNN overlap reports plainly what share of a point's neighbours survive. All are computed in the shared PCA space the themes were built in, not the raw 384-dimensional one, because that is where the clustering happened.

Theme selection evaluates candidate cluster counts with silhouette score and rejects solutions with any cluster below 2% of the corpus; near-ties favour the smaller model. Because silhouette is nearly flat over that range, it is the tie-break rather than the maximum that decides `k` in practice -- so every candidate it considered is reported with its silhouette, smallest-cluster share and rejection flag, and the panel states the spread. A claim that `k` was chosen is only auditable if the rejected alternatives are visible. The diagnostics are persisted with the semantic run that produced the layout rather than recomputed on render, so a past map keeps its own caveats. Theme labels remain descriptive summaries of the corpus, and their numbering is not stable across runs.

#### Trend testing and multiplicity

Keyword trends use Mann-Kendall with Sen slopes on zero-filled annual series. The Benjamini-Hochberg adjustment is computed over every keyword meeting the minimum-prevalence rule, not over the extreme slopes displayed: adjusting a set already selected for being extreme inflates significance instead of controlling it. The table shows the charted terms carrying their family-adjusted values, and the caption states the family size.

Mann-Kendall assumes independent observations, which annual counts are not. The Hamed-Rao variance inflation, computed from the autocorrelation of the detrended ranks, is reported alongside the uncorrected value as a sensitivity: a 40-point random walk with no true trend receives an inflation factor near 4.7, while white noise and a genuine trend both receive 1.0. It is a second column rather than a replacement, because the uncorrected p-value already has callers and redefining it in place would silently rewrite published trend tables; the caption states how many keywords survive the correction.

The slope chart reports each term's 95% confidence interval as an error bar, and the caption states how many of the charted terms have an interval excluding zero -- on short annual series most do not. The breakpoint test searches every admissible split and then takes the maximum F, whose null distribution is not F: measured over 200 pure-noise series, reading it against the F table reported a regime change 32.5% of the time at alpha 0.05. The reported p-value is now empirical, from permuting the series under a no-break null, which brings the same measurement to 8.0%. The F-table value is kept beside it so the gap stays visible, and the resolution floor of 1/(draws+1) is reported so a small empirical p-value is not read as precision it does not have.

#### Author and network indicators

Author names are canonicalized heuristically. Homonyms can merge and spelling variants can split. The h, g, e, i10, and m indicators are computed only from articles present in this corpus and must be labeled as corpus-scoped indicators rather than full-career metrics.

Co-authorship and keyword networks describe relations within the collected corpus. Community labels, centrality, small-world diagnostics, and concentration measures inherit the corpus selection boundary.

Observed clustering is compared against degree-preserving rewirings of the same graph, reported as a z-score and an empirical p-value. The null matters because a co-authorship graph is clustered simply by virtue of papers having several authors; preserving each author's degree separates collaborative structure from that artefact. Degree assortativity is scored against those same rewirings, since the degree sequence alone forces part of it and a bare coefficient would be unreadable. Robustness reports the giant-component share after removing the highest-degree decile and after removing the same count at random: neither figure means anything alone, but the gap between them distinguishes a network held together by a few authors from one with distributed structure. New versus returning collaboration ties are counted per period, a tie counting as new in the period holding its first-ever collaboration -- a static recurrent-edge count cannot separate a field still recruiting collaborators from one consolidating into fixed teams, because both produce the same number of repeat pairs. Path length and the small-world sigma are computed on the largest connected component while density and the centralities cover the whole graph including fragments, so the two are not on the same population and the panel says so.

Lotka's law is fitted on distinct-DOI counts per canonical author. Like every other person-level indicator here it is corpus-scoped, and it inherits the heuristic-identity caveat: homonyms merge and spelling variants split.

#### Engineering taxonomies

The seven taxonomies on the Engineering-evidence page are regular expressions over
title and abstract, enumerated in `analytics.py::TAXONOMY_REGISTRY`. They are
**exploratory**: no reviewed sample exists, so their precision and recall are
unmeasured, and every panel says so.

Each one reports its measured coverage, because the share it never matched is the
difference between "among the articles this rule recognised, which class is most
common?" and "what does this corpus do?". Only objective functions exceeds half the
corpus, at 78.3%, and it assigns 1,457 of those articles to more than one class; the
other six classify between 16.1% and 31.8%. The classes are not mutually exclusive,
so per-class counts sum past the classified population and must not be read as a
partition.

When a reviewed sample is imported, `taxonomy_precision_from_labels` reports per-class
precision, recall and F1 against it, and the exploratory notice is replaced by those
numbers. `ambiguous` labels are excluded from both metrics and reported separately: a
reviewer who could not decide is evidence about where the class boundary sits, not a
negative example.

#### Retrieval and anomaly audit

Hybrid retrieval combines BM25 and dense embedding ranks through reciprocal rank fusion (k = 60). The unit ranked is the article: by default each article competes through its one abstract chunk, because ranking every full-text chunk would favour the ~3% of articles that have a PDF; with the all-chunks scope an article ranks where its best chunk ranks. Eligibility filters apply before ranking, identically in the lexical and dense paths. An optional cross-encoder reorders the fused candidates. The interface exposes component ranks/scores so a reviewer can inspect why an item appeared. No retrieval-quality claim is made without a labeled query relevance set.

Isolation Forest results are audit priorities. They identify multivariate outliers and do not imply data errors, exceptional scientific quality, or fraud.

#### Reproducibility

Every published analytical population resolves to a deterministic dataset version derived from the ordered source manifest, search configuration, transformation code, dependency lock, and duplicate-curation decisions. Consumed source bytes are retained content-addressably, while candidate Gold articles, chunks, embeddings, semantics, and duplicate pairs remain isolated until blocking contracts pass. Re-running unchanged evidence reuses the same logical version; rollback reactivates an exact prior Gold snapshot.

Analytical functions are pure Python functions without Streamlit imports. Database access belongs in `apps/dashboard/data.py`, cached normalization and joins in `apps/dashboard/loaders.py`, Plotly construction in `apps/dashboard/charts.py`, and rendering in page controllers. Randomized algorithms use fixed seeds. Tests cover missingness, minimum sample sizes, degenerate inputs, deterministic behavior, source-change propagation, publication isolation, and method-specific invariants.

## 7. Observability and service objectives

### 7.1 Required telemetry

- Parent/stage run status, duration, attempts, and triggering identity.
- Input additions/changes/removals and row counts by layer/source.
- Rejection reasons, duplicate decisions, chunk invalidation, and embedding coverage.
- External enrichment success/not-found/error/rate-limit counts and observation age.
- Data-quality check outcomes and active dataset version.
- **Target:** dashboard query/cache latency for expensive loaders and search latency by retrieval mode. **Current:** no loader is timed; per-mode search latency is measured only on demand, by the retrieval benchmark harness (`apps/dashboard/retrieval_eval.py`) behind the Quality page's benchmark.

**Target:** logs use structured fields with parent run ID, stage run ID, and dataset version. **Current:** the CLI logs plain text (`%(asctime)s %(levelname)s %(name)s: %(message)s`) to stderr and to `data/.lake_research_map/pipeline.log` (`LAKE_RESEARCH_MAP_LOG` overrides it); execution, stage-run and version ids live in the `lit_pipeline_*` rows, not in log fields. A telemetry-write failure is surfaced; it must not be silently downgraded to debug-only information for a production run.

### 7.2 Read-only audits

Telemetry records what a run did; the audits answer whether the corpus is fit to
analyse. They are read-only, safe to run at any time, and are the surface where
coverage becomes visible.

| Command | Reports | Denominator |
|---|---|---|
| `audit reference-corpus` | Active version, article/chunk/semantic counts, blocking quality failures | The active dataset version |
| `audit citation-graph` | Coverage per direction and the population usable for a disruption index (their **intersection**, never the larger side); **graph integrity** -- self-loops, dangling `cites:` edges, and agreement between OpenAlex's `referenced_works` and `cites:` indexes over corpus-internal pairs (gate 95%); **access validation** -- OpenAlex open-access status against the IEEE CSV licence, an independent source (gate 90%) | Observed works, plus a per-registrant table against the corpus |
| `audit citation-years` | Annual trajectories known (an empty series from a never-cited work is a known zero), left-censored works published before the provider's series begins, the population complete from publication, the share of cited references carrying a publication year over one list per work -- the Crossref deposit when one exists, else OpenAlex -- and a validation verdict (provider year vs corpus year, reference not newer than its citer) (gate 80% coverage each) | Observed works and reference pairs, plus a per-registrant table against the corpus |

Every audit that reports a share of externally collected data also reports it
per registrant against the corpus, because an aggregate over a partial
collection inherits whatever ordered that collection — see `ADR-07`. A
publisher that the crawl has not reached prints as `0 / N  <- not reached`
rather than being averaged into a reassuring total.

A share of zero rows is not a share of zero evidence. Nor is every zero an answer: a provider's
zero counts as known only when no other source contradicts it, because
OpenAlex's `reference_count` is empty whenever it lacks the list. Two of the first
full-corpus figures were wrong for that reason alone -- backward coverage read
89.5% because 324 works reporting zero references produced no edge, and
trajectory coverage read 76.7% because 677 never-cited works produced no
series -- and both audits now count an upstream zero as a known answer.

### 7.2.1 Crawl resumption semantics

Both OpenAlex passes are resumable and self-limiting. The backward pass skips
DOIs that already carry a *successful* observation, so failures are retried and
successes are not re-fetched; the forward pass skips works carrying
`citing_crawled_at`, which is what distinguishes "already crawled, nobody cites
it" from "not yet crawled" — edge presence cannot. Both stop after five
consecutive refusals, commit progress as they go, and report `stopped_early`. A
throttled or rejected batch is never stamped as crawled: stamping it would make
the resume skip it permanently and record it as truncated for no reason, which
is what the first per-work loop did. The forward pass and the reference-year
resolver both OR-join 50 ids per query, and decide truncation from OpenAlex's
own `meta.count` rather than from whether the page budget ran out. Cited
references are resolved into `lit_reference_works`, kept separate from
`lit_external_works` so that adding 56,000 cited works cannot change the
population every coverage audit divides by.

### 7.3 Initial SLOs

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
- No network service can start a pipeline run: the CLI is the only entry point, so starting a run needs a shell on the machine. Qdrant and the dashboard bind to loopback.
- PDF text may be copyrighted; access and redistribution remain local and controlled. Retrieval returns short evidence passages and DOI provenance rather than bulk document export. The MCP server speaks stdio to a local client and opens no network port; its `get_evidence` tool returns at most a few chunks of one article.

## 9. Testing strategy

### 9.1 Current baseline

The suite (its dated size is in §14) runs with isolated in-memory SQLite sessions and `qdrant-client`'s local mode; four opt-in tests are skipped by the default run (two `mysql`, two `qdrant`). Besides unit coverage it exercises deterministic fingerprints, content-addressed retention, rename detection, Bronze deletion propagation, isolated Gold candidates, embedding contract failures, atomic materialization, exact Gold reactivation, an end-to-end correlated pipeline fixture, stage-by-stage parent correlation, temporal enrichment observations, persistent human-review evidence, retrieval metrics, provenance coverage, persisted semantic diagnostics, the durable-label-to-calibration join with its approval digests, the injected-fixture citation-graph crawl, automatic abandoned-run recovery, and the plausible-year bound that keeps in-press records dated to next year inside every trend. The MySQL acceptance tests cover JSON/NULL/BLOB round trips, rollback, advisory locks, and the idempotent `lit_config` uniqueness migration. Ruff lint and format checks are required.

SQLite remains the default fast suite. MySQL-specific acceptance is recorded separately against MySQL 8.4 and must be rerun for changes to JSON/NULL behavior, BLOBs, DDL, transactions, or advisory locks. The vector store follows the same split: `qdrant-client`'s in-process local mode stands in for the server in the default suite, and `RUN_QDRANT_INTEGRATION=1 uv run pytest -m qdrant` checks what local mode ignores (payload indexes, exact-search parameters) against the compose service.

### 9.2 Target test layers

Each layer with its status on 2026-11-06 (**exists**, **partial**, **missing**) and the files that provide it.

1. **Unit tests:** pure parsing, normalization, statistics, and edge cases. **Exists** -- most files under `tests/`.
2. **Property tests:** idempotency, uniqueness, monotonic cumulative counts, vector dimensions, and deterministic seeded outputs. **Partial** -- example-based idempotency/determinism tests only (`test_vector_store.py`, `test_gold_articles.py`, `test_crawl_sampling.py`); no generated inputs (no Hypothesis).
3. **Contract tests:** each dataset invariant and quality gate, including deletion propagation. **Partial** -- embed/publication/semantic gates (`test_vector_collections.py`, `test_versioning_contracts.py`) and the raw contract (`test_raw_config.py`); the bronze/silver/gold contracts run only on the passing path of the end-to-end fixture, with no failure-path test.
4. **MySQL integration tests:** migrations, JSON SQL NULL behavior, constraints, transactions, locks, and binary vectors. **Partial** -- two opt-in tests in `test_mysql_contracts.py` (`RUN_MYSQL_INTEGRATION=1`): JSON/BLOB/NULL/transaction/advisory lock, and the `lit_config` uniqueness migration.
5. **Golden-corpus tests:** a small fixture with duplicates, removals, missing DOI, changed text, PDF ambiguity, and partial enrichment. **Partial** -- `test_versioning_contracts.py` covers add/edit/rename/remove/reactivate; no golden case for duplicates, missing DOI, PDF ambiguity or partial enrichment.
6. **Statistical validation tests:** simulated known effects/nulls, bootstrap coverage, leakage checks, and baseline comparisons. **Partial** -- `test_advanced_statistics.py`, `test_forecasting.py`; no bootstrap-coverage simulation.
7. **Streamlit tests:** `st.testing.v1.AppTest` smoke tests for navigation, filters, degraded states, and conditional heavy tabs. **Partial** -- `test_dashboard_pages.py`, `test_dashboard_app.py` (and `test_production_page.py`); no navigation, sidebar-filter or conditional-tab tests.
8. **Browser tests:** only for behavior AppTest cannot simulate, such as rendered theme/CSS or chart selection events. **Missing.**

## 10. Scaling and extension points

Current in-process computation remains the default while measured refresh time, memory, and search latency meet targets. Vector storage moved to Qdrant under ADR-08 for consumption reasons (external RAG clients and the MCP server), not for speed: search stays exact. Approximate search, distributed transforms, or materialized analytical tables still require an ADR supported by benchmarks and operational cost.

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

**Rationale:** Migration risk outweighs current benefit, but shared-database isolation must be treated as a security contract. That contract now rests entirely on the table-name prefix: the least-privilege roles that once enforced it in the database were removed on 2026-09-22 as unjustified for a single-machine deployment (`WP-08`, §13). Only `lit_`-prefixed tables may be created or touched, and nothing but code discipline enforces it.

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

**Superseded for vector storage by `ADR-08` (2026-11-06).** The concurrency requirement this ADR named arrived: the corpus is to be consumed by a RAG client outside the dashboard process. Everything else here -- pandas/NumPy/scikit-learn/ONNX, in-process semantic computation -- stands.

### `ADR-05` — Human authority for exclusion and cross-DOI identity

**Decision:** Models prioritize and measure; reviewers decide.

**Alternative:** Automatic semantic exclusion/merge.

**Rationale:** False exclusion and identity errors directly threaten SLR validity.

### `ADR-06` — Read-only dashboard

**Decision:** Data mutations remain in audited CLI workflows.

**Alternative:** Direct page writes.

**Rationale:** Controlled workflows provide clearer permissions, recovery, and audit trails.

### `ADR-07` — Ordering an external crawl is sampling, not iteration

**Context:** the OpenAlex refresh walked `sorted(dois)`, which reads as neutral
iteration over a stable key. A DOI's registrant prefix is its publisher, so the
sorted key is correlated with source: `10.1016` (Elsevier) sorts entirely before
`10.1109` (IEEE). Combined with a request quota that stops the crawl part-way,
the first live run observed 989 Elsevier works and zero IEEE ones, and every
coverage figure computed on that subset described one publisher while reading as
a statement about the corpus.

**Decision:** external collection is ordered so that any prefix of it is
representative. `interleave_by_registrant` positions each DOI at
`(index + 0.5) / group_size` within its registrant and sorts by that fraction,
which makes the first N proportional across publishers. The ordering is
deterministic and seedless, so the same corpus always produces the same
sequence. Coverage audits report per registrant against the corpus denominator,
not only in aggregate.

**Consequences:** a crawl interrupted at any point yields a usable sample rather
than a publisher-complete block, and a source at zero is visible in the audit
output instead of being averaged away. The cost is that no single publisher is
ever finished early, which matters only if a downstream analysis needed one
source complete — none does.

**Generalization:** this applies to any bounded traversal of an ordered key
whose order correlates with a property under analysis, not just to DOIs.

### `ADR-08` — Qdrant is the vector store; Gold keeps the evidence

**Context:** the corpus is to feed a RAG client that filters the most relevant
articles for a subject, alongside the dashboard and an MCP server. Vectors in
MySQL BLOBs are readable only by loading every one of them into the reading
process, which is fine for the dashboard and wrong for an external client that
needs filtered nearest-neighbour search behind a standard API. This is the
"demonstrated concurrency requirement" `ADR-04` named, not novelty.

**Decision:** the embed stage writes vectors to Qdrant only, one collection per
dataset version (`lit_chunks_<version_id>`), with text, provenance, article
filter fields and screening signals as payload. Gold keeps chunk text, metadata,
embedding provenance and the publication pointer, and stays the only thing a
publication commits: the embed contract checks every chunk against its point
before the pointer can move. External clients query the `lit_chunks_active`
alias, moved after each publication commit; internal readers resolve
collections through the Gold pointer and never depend on the alias. Search is
exact (brute force) so a query ranks identically on every run. Qdrant runs as
the opt-in `qdrant` compose profile, bound to loopback.

**Alternatives:** keep BLOBs and add an HTTP search service of our own (a
second implementation of filtering and pagination to maintain); store vectors
in both places (two sources of truth that can disagree); a hosted vector service
(PDF full text would leave the machine).

**Consequences:** Qdrant is now required for `embed`, `semantic`, publication
and dense search; without it the pipeline fails fast naming the compose
command, and the dashboard degrades dense search to lexical. Losing the Qdrant
volume loses no evidence: every vector is a function of Gold text and the
pinned model revision, and `vectors rebuild` recomputes it. The BLOBs of
versions published before the migration stay in MySQL, unchanged, as immutable
history and as the source for `vectors backfill`; deleting them would mutate
published snapshots and needs its own decision.

### `ADR-09` — No orchestrator: the CLI runs the pipeline

**Context:** Airflow ran each stage as a DAG of thin `BashOperator` wrappers
around the CLI, and the dashboard triggered those DAGs through Airflow's REST
API. It added a container with its own virtual environment, metadata database
and authentication, a REST client in the dashboard, and per-stage retry and
timeout policies -- for a single researcher who starts runs by hand. On
2026-11-06 the owner removed the DAGs and asked for Airflow to go entirely.

**Decision:** the CLI is the only way to run the pipeline (`--stage <stage>`,
`--stage all`). The Airflow service, image, DAGs, REST client and dashboard
trigger buttons are removed; the Pipeline page shows the commands instead.
`--execution-id`/`--workflow` stay, so a workflow can still run stage by stage
as separate processes recorded as one execution (cron or CI can drive that).

**Alternatives:** keep Airflow for scheduling (no schedule exists to serve);
replace it with another orchestrator (the same cost for the same absent need).

**Consequences:** no automatic retries or timeouts -- a failed stage is re-run
by hand, which every stage's idempotency and `embed`'s resumability make safe;
no run history beyond `lit_pipeline_executions`/`lit_pipeline_runs`, which
already held it; and no page can start a run, so `ADR-06` now holds trivially.
Docker Compose keeps the dashboard and the opt-in Qdrant service.

## 12. Known technical debt

- Analytical reads are already bound to the publication pointer: `apps/dashboard/data.py::load_articles` routes Gold through `load_active_dataset_table`, which scopes every query by `active_dataset_version()`. The residual is narrower than previously recorded -- `assess_gold_articles` is still a structural check (non-empty, unique DOI, expected columns) rather than a read of the persisted `lit_quality_results` for that version.
- No automatic stage retry or timeout since `ADR-09`: a failed stage is re-run by hand (every stage is idempotent and `embed` resumes).
- OpenAlex responses are append-only observations with observation time, response state, HTTP status, response hash, retry count, longitudinal counts, outgoing edges, and access metadata. A real refresh still requires local API credentials.
- Chunk and duplicate-candidate logical keys are not fully constrained in the database.
- The core imports an application: `transform/evidence_samples.py` takes `canonical_author`, `OPTIMIZATION_METHOD_PATTERNS` and `taxonomy_haystack`, and `transform/versioned_gold.py` takes `semantic_stability_diagnostics`, all from `apps/dashboard/analytics.py`. Moving those four names into the core would make `apps/` a strict leaf. (`pipeline.py` also imports `apps.qdrant` and `apps.api`, but only to wire their commands into the CLI.)
- Dashboard caches are not invalidated after a publication (§4.2 step 8): pages can show the previous version until a loader's TTL expires or someone presses *Refresh data*.
- Logs are plain text, not structured fields carrying run/stage/version ids (§7.1).
- No dashboard loader latency telemetry; search latency by mode exists only in the on-demand retrieval benchmark (§7.1).
- No property-based tests (no Hypothesis) and no browser tests (§9.2).
- `NFR-05` has no measurement on the Qdrant architecture: the only latency figures (§14, 2026-09-21) are for the retired in-process search.
- Statistical functions need the validation improvements defined in `PRD.md` before confirmatory interpretation.

Open work that is blocked on human evidence is listed in PRD §12.

## 13. Delivery history

Work packages that are delivered, one line each; the ids remain because code comments and tests cite them.
Open ones are PRD §12. The full status notes and dated evidence records they summarised live in git history
(`git log -- docs/ROADMAP.md docs/evidence/`).

| Id | Title | Outcome |
|---|---|---|
| `WP-01` | Dataset versions and correlated pipeline runs | 2026-09-21 — Additive version/run schema; parent and stage IDs; source-manifest/config/code hashes; active/published/failed state; lineage in run metrics. |
| `WP-02` | Source reconciliation and deletion propagation | 2026-09-21 — Immutable source revisions or tombstones, active-snapshot reconciliation, stale Raw/Bronze cleanup, and removal metrics. |
| `WP-03` | Executable data contracts and publication gates | 2026-09-21 — `lit_quality_results`, severity policy, uniqueness/referential/coverage/range checks, stage gate API, and dashboard contract status. |
| `WP-04` | Gold as the canonical analytical population | 2026-09-21 — Gold readiness contract; propagation/join of required quality fields; explicit degraded pre-Gold mode; updated loaders and page coverage captions. |
| `WP-05` | Identity, PDF matching, and rejection validation | Open — see PRD §12. |
| `WP-06` | Embedding and Semantic compatibility contract | 2026-09-21 — Text hash, model revision, dimension/dtype checks, binary-first pending logic, JSON fallback migration, semantic-run metadata, and atomic complete-coverage gate. |
| `WP-07` | Enrichment observations and temporal semantics | 2026-09-23 — Provider observations with `observed_at`, status, identifiers, counts, response hash, retry metadata, and deterministic selection of the analysis snapshot. |
| `WP-08` | Concurrency, recovery, and operational security | 2026-09-22 — Project advisory lock, reliable terminal run status, deployment-mode security guidance, and audited trigger identity. The stage retry/timeout matrix it also delivered lived in the Airflow DAGs and was removed with them (`ADR-09`). Least-privilege database roles were dropped from the deliverable, not deferred. |
| `WP-09` | Persistent dual-review screening workflow | Open — see PRD §12. |
| `WP-10` | Screening threshold validation | Open — see PRD §12. |
| `WP-11` | Author identity audit and overrides | Open — see PRD §12. |
| `WP-12` | Engineering-taxonomy validation | 2026-09-22 — Machinery only: the stratified taxonomy sample (`evidence taxonomy`, with an unclassified stratum), the label guide (Appendix A.6, still a draft), measured coverage, and `taxonomy_precision_from_labels` for per-class precision/recall/F1 with ambiguous/unclassified reporting. No human taxonomy labels exist (an AI-assisted pass of 120 labels does, like the retrieval one), so no precision or recall has been measured (§6.5; PRD §12 `WP-21`). |
| `WP-13` | Retrieval evaluation corpus | Open — see PRD §12. |
| `WP-14` | Citation distribution inference | 2026-09-21 — Documented zero handling, `x_min` selection/sensitivity, MLE comparisons, likelihood ratios, bootstrap goodness-of-fit, and uncertainty for parameters. |
| `WP-15` | Citation count-model diagnostics | 2026-09-22 — Missingness profile, VIF/condition number, residual/influence checks, Poisson/NB/zero-inflated comparison when identifiable, alternative age specifications, and coefficient forest with CIs. |
| `WP-16` | Multiple testing and temporal trend validity | 2026-09-22 — Test-family definitions, minimum prevalence, Benjamini-Hochberg adjusted values, effect-size thresholds, serial-dependence sensitivity, and exploratory breakpoint correction/bootstrap. |
| `WP-17` | Semantic stability and projection diagnostics | 2026-09-22 — Bootstrap/subsample ARI, cluster-count sensitivity, projection trustworthiness, neighborhood preservation, seed stability, and drift computed in embedding space before visualization. |
| `WP-18` | Forecast and Bass validation | 2026-09-22 — Expanding-window backtests by horizon, MAE/MASE or baseline skill, empirical interval coverage/width, dynamic complete-year cutoff, and Bass parameter bootstrap/sensitivity. |
| `WP-19` | Network null models and temporal collaboration | 2026-09-22 — Component-aware metrics, degree-preserving null models, assortativity/robustness checks, and periodized new-versus-repeated collaboration ties. |
| `WP-20` | Analytical inventory and redundancy removal | 2026-09-22 — Machine-readable or documented chart inventory with owner/question/population; removal of dead helpers/tests; relocation or deletion of duplicates; Overview reduced to macro state. |
| `WP-21` | Integrate validated analyses into existing pages | Open — see PRD §12. |
| `WP-22` | Streamlit performance, behavior, and accessibility | 2026-09-22 — Dynamic heavy tabs, bounded caches, stable loading slots, forms for expensive searches, native responsive layout where practical, stable widget keys, English sentence-case/accessibility review, and `st.testing.v1.AppTest` smoke… |
| `WP-23` | Reference-year and citation-history ingestion | 2026-09-23 — Governed cited-reference publication years and annual citation trajectories with coverage/provenance. |
| `WP-24` | Citation graph and verified access data | Delivered — Versioned citation graph; verified access observations; documented sampling/coverage and model protocol. |
| `WP-25` | Scale-triggered storage and compute evolution | 2026-11-06 — Closed by adopting Qdrant under `ADR-08`, for a consumption requirement (an external RAG client and the MCP server) with exact search kept. No benchmark report or ADR comparing Faiss, MySQL storage or managed systems was produced, and no latency has been measured on the Qdrant architecture: `NFR-05` is unmeasured there (§12). |
| `WP-26` | Qdrant vector store | 2026-11-06 — `db/vector_store.py` (one immutable collection per version, deterministic point ids, exact search, `lit_chunks_active` alias); the embed stage writing only Qdrant and copying reusable vectors by text hash before computing; the… |
| `WP-27` | Full-text cleaning before chunking | 2026-11-06 — `transform/pdf_text.py` (four pure cleaners in a fixed order), applied by both Gold builders, which report `pdf_*` counts in their stats. |
| `WP-28` | Article-level retrieval API, CLI and MCP server | 2026-11-06 — `apps/api/` (`Retriever.search/screen/evidence/article/status`, filters shared by pandas and Qdrant, BM25 index built once, RRF, optional pinned cross-encoder, untruncated token counts); `lake-research-map search|screen|mcp`;… |

## 14. Measured baselines

Re-measure before quoting; the corpus grows with every export (CLAUDE.md keeps the data quirks).

| Date | Measurement | Result |
|---|---|---|
| 2026-09-21 | Dense search, binary-first, 7,552 chunks | 11.06 MiB matrix; load 11.6 s → 2.3 s once the JSON mirror was excluded; warm cosine P50 37.8 ms, P95 92.7 ms |
| 2026-09-22 | `enrichment_cache.json` vs OpenAlex, 1,875 comparable rows | reference counts agree 99.7%, citation counts 90.0% (median error 0) |
| 2026-09-23 | OpenAlex DOI refresh | 3,099 of 3,115 DOIs observed; 16 absent from OpenAlex |
| 2026-11-06 | Qdrant backfill of the active version | 7,552 vectors bit-identical to the MySQL BLOBs; relevance recomputed within 5.1e-7; legacy import skipped (no provenance) |
| 2026-11-06 | First publication on Qdrant (`4ffe43c7…`) | 7,552 vectors copied, 0 embedded; semantics identical (Δrelevance 0, themes 100%) |
| 2026-11-06 | PDF cleaning, 97 linked PDFs | reference lists cut in 88; 6,323 hyphenations rejoined; 7,368 furniture lines dropped; full-text chunks 4,437 → 3,623 |
| 2026-11-06 | Regeneration with cleaned text (`7cbb67e3…`) | 3,115 abstract vectors copied, 3,623 full-text chunks embedded (~32 min at 2 threads); 206 duplicate pairs, 4 themes, unchanged |
| 2026-11-06 | Retrieval disclosures | 95 articles without a usable abstract (< 50 chars); 21 abstract chunks exceed the 512-token window |
| 2026-11-06 | Test collection (`uv run pytest --collect-only -q`) | 521 tests collected, of which 4 opt-in (`mysql`/`qdrant` markers) |
| 2026-11-06 | Model snapshot drift | Hub snapshot `aa8f8b06…` differs from the pin `52398278…` only in README; re-embedding 200 chunks reproduced the stored vectors exactly |

## Appendix A. Review protocols

The instructions human reviewers follow. `reviews setup --instructions-file` stores the text with its SHA-256 in
`lit_review_protocols`; the stored text is authoritative, and a version is immutable once used. To start a new
version, copy its section here into a file, edit it, and set it up under a new version id.

### A.1 `author-v1`

**Workflow:** `author` · **Labels:** `same` · `different` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

##### 1. What you are judging

Each subject is `<name A>||<name B>`. The question is: **are these two names the
same person?**

##### 2. Why this sample looks the way it does

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

##### 3. Decision rules

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

### A.2 `pdf-v1`

**Workflow:** `pdf` · **Labels:** `match` · `mismatch` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

##### 1. What you are judging

Each subject is `<DOI>::<path to PDF>`. You are judging **the pairing**, not either
side alone: the article can be correct and its PDF still be the wrong file.

##### 2. Why this sample looks the way it does

PDFs are linked to articles by fuzzy title match at threshold 85. The sample is
banded to straddle that threshold, because a sample of confident matches cannot tell
you where the threshold should sit.

On the current corpus the bands are badly uneven — **96 of the 97 linked PDFs score
95 or above and exactly one falls in 85–90**. A sample of 16 therefore cannot
calibrate the threshold; it can only detect whether high-confidence matches are
trustworthy. Record that limitation rather than reporting a threshold as validated.

##### 3. Decision rules

- `match` — the PDF is the full text of that DOI.
- `mismatch` — it is a different article. A preprint or an extended conference
  version of the same work is **`mismatch`**: the DOI identifies one specific
  record, and analyses that read page counts or full text must not silently use
  another version.
- `uncertain` — the file does not open, is truncated, is scanned without usable
  text, or the title is genuinely ambiguous between two records.

Do **not** look at `pdf_match_score` before deciding. It is the thing under test.

Compare the PDF's first page — title, authors, venue, year — against the record.
Record a `rationale` for every `mismatch` and `uncertain`.

### A.3 `retrieval-v1`

**Workflow:** `retrieval` · **Labels:** `relevant` · `not_relevant` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

##### 1. What you are judging

Each subject is `<query id>::<DOI>`, for example `q03::10.1016/j.est.2026.121555`.
The query texts are versioned in `transform/evidence_samples.py::RETRIEVAL_QUERIES`;
read the one you are judging before you start on its results.

The question is: **would a researcher asking this query want this article back?**

##### 2. Why this sample looks the way it does

The 96 judgements pool the top-10 of every retrieval mode — dense, BM25 and their
RRF fusion — into one set. Pooling is what makes the three comparable: judging only
what the current default returns would build the ground truth out of the default's
own hits and guarantee it wins.

So a subject is *a result some mode returned*, not a result the default returned.
You cannot tell which mode produced it, and you should not try.

##### 3. Decision rules

- `relevant` — the article addresses what the query asks. Partial coverage counts:
  a paper on DG placement that ignores sizing is still `relevant` to q01.
- `not_relevant` — it addresses a different problem. Sharing vocabulary is not
  enough; the logistics papers in this corpus match "distribution" and "network"
  while being about freight.
- `uncertain` — the abstract is too thin to tell.

Judge each `<query, DOI>` pair independently. The same DOI under two queries is two
judgements and may legitimately differ.

Do **not** rank. Recall@k, MRR and nDCG are computed from these binary labels; a
ranking judgement here would be applied twice.

### A.4 `screening-v1`

**Workflow:** `screening` · **Labels:** `include` · `exclude` · `uncertain`

> **Draft.** The operational rules are mine. **Section 3 is a genuine research
> decision and the criteria below are a placeholder — replace them with the
> review's own inclusion criteria before labelling anything**, then bump to `v2`
> (a protocol version is immutable once used).

##### 1. What you are judging

Each subject is `reject::<source>::<source id>`. These are records dropped at the
silver stage for having no DOI, and they are invisible to every downstream analysis.

The question is: **should this record have been in the corpus?**

##### 2. Why this sample looks the way it does

48 records were rejected, all for `no_doi`, and 30 are sampled — 62% coverage, high
enough to bound the bias directly.

The point is not to recover the records. It is to answer whether dropping them was
unbiased. If the rejected set turns out to be systematically in-scope work, "records
without a DOI were excluded" stops being a footnote and becomes a limitation on
every count in the review.

##### 3. Inclusion criteria — PLACEHOLDER, REPLACE BEFORE USE

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

### A.5 `screening-v2`

**Workflow:** `screening` · **Labels:** `include` · `exclude` · `uncertain`

> **Approved for use on 2026-09-22.** This version is immutable once persisted.
> Reviewers must work independently and must not inspect the other reviewer's
> decision before submitting their own.

##### 1. Review question

Each subject is `reject::<source>::<source id>`: a publisher record excluded at
Silver because no DOI could be normalized. Decide whether the record would have
belonged in the review independently of its missing DOI.

The review scope is research on planning electric-power distribution systems or
distribution networks. A planning contribution makes a medium- or long-term
decision about investments, capacity, topology, location, sizing, reinforcement,
expansion, replacement, or staged development of distribution assets or resources.

##### 2. Evidence to inspect

Use the title, abstract, publication type, venue, keywords, and publisher metadata
available for the rejected record. Do not infer relevance from the search query,
publisher, venue, authors, or neighboring records alone. A missing DOI is the
reason for this audit and is never itself a reason to exclude the record.

##### 3. Decision rules

Choose `include` when all applicable evidence supports both conditions:

1. The physical system is an electric-power distribution network, including an
   active distribution network, microgrid, or distribution-connected resource.
2. The work makes or evaluates a planning decision. Eligible decisions include
   feeder/substation expansion, network reinforcement, topology design, asset
   replacement, DER/storage/EV-infrastructure siting or sizing, resilience
   investment, and multi-stage distribution development.

Primary studies, systematic or technical reviews, and substantive book chapters
are eligible when they satisfy both conditions. A method paper is eligible when
its evaluated application and contribution are explicitly distribution planning.

Choose `exclude` when any of the following is established:

- the work concerns logistics, supply chains, freight, water, gas, transport,
  telecommunications, transmission-only, or generation-only planning;
- it addresses operation, control, protection, state estimation, short-term
  scheduling, forecasting, or power flow without a planning decision;
- DER, storage, electric vehicles, reliability, resilience, or reconfiguration is
  discussed without a siting, sizing, investment, expansion, reinforcement, or
  long-horizon asset decision;
- it is front matter, an index, preface, table of contents, editorial, correction,
  announcement, or other non-substantive publication;
- it is demonstrably a duplicate of a record already represented in the corpus.

Choose `uncertain` only when the available metadata cannot establish whether both
inclusion conditions hold. Typical cases are a generic title with no abstract, a
planning term whose physical domain is unspecified, or an abstract that mentions
distribution planning only as future work. Do not use `uncertain` merely because
the method, result, or writing quality is weak.

##### 4. Precedence and rationale

Apply publication-type exclusions first, then physical-system scope, then the
planning-decision requirement. When a record contains both eligible and ineligible
material, label `include` if electric-distribution planning is a substantive study
objective or evaluated application rather than background context.

Every `include` and `uncertain` decision requires a concise rationale quoting or
paraphrasing the decisive metadata. For `exclude`, record the first applicable
exclusion category. Reviewers must not attempt to recover or merge the record;
adjudication and any later remediation are separate steps.

##### 5. Interpretation

The audit estimates selection bias caused by DOI-based exclusion. It does not
retroactively add records to Gold. Disagreements preserve both raw labels and are
resolved through the versioned adjudication workflow.

### A.6 `taxonomy-v1`

**Workflow:** `taxonomy` · **Labels:** `present` · `absent` · `ambiguous`

> **Draft.** The operational rules below are mine; the inclusion criteria are a
> research decision. Read section 3 before labelling and change what you disagree
> with — then bump to `v2`, because a protocol version is immutable once used.

##### 1. What you are judging

Each subject is `<Class>::<DOI>`, for example
`Genetic Algorithms (GA)::10.1016/j.apenergy.2017.07.004`.

You are judging **one class against one article**, not the article as a whole. The
same DOI appears once per class it was sampled for, and the answers are independent:
a paper can be `present` for MILP and `absent` for GA.

The question is: **does this article actually use, propose or evaluate this method?**

##### 2. Why this sample looks the way it does

The classifier is a regex over title + abstract. It matched 990 of 3,115 articles;
the other **2,125 (68%) matched nothing** and are sampled under `unclassified::`.
That stratum is the point of the exercise — precision measured only on articles the
regex already matched cannot see a single false negative.

For an `unclassified::<DOI>` subject the question inverts: **does this article belong
to any of the nine classes?** Answer `absent` if it genuinely uses none of them, and
`present` if it uses one — naming which in the `rationale` column, since that is a
miss the regex should have caught.

##### 3. Decision rules — review these before labelling

- `present` — the method is used, proposed, extended, or evaluated as a contribution
  of this work.
- `absent` — the method is not used. Mentioning it only in related work, in a
  comparison table of other people's results, or as future work is **`absent`**.
- `ambiguous` — the abstract is too thin to tell, the acronym is genuinely
  overloaded (`GA`, `PSO` and `conic` all collide with unrelated terms), or the
  method is used by a tool the paper merely invokes without claiming it.

Judge from title, abstract and keywords. Do not open the full text: the classifier
only sees title + abstract, so validating it against more evidence than it has would
measure the wrong thing.

Record a one-line `rationale` for every `ambiguous`, and for any `present` on an
`unclassified` subject.

##### 4. What is out of scope

Whether the taxonomy's nine classes are the right nine. That is a separate decision;
here, take the classes as given and judge the assignment.
