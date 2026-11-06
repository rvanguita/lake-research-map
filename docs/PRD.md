# Product and Research Requirements Document — lake-research-map

**Status:** Active specification

**Evidence cutoff:** 2026-09-21

**Implementation baseline:** the active 2026-11-06 Gold version `7cbb67e3…` contains 3,115 articles and 6,738 chunks (3,115 abstracts, 3,623 cleaned full-text chunks from 97 PDFs), every one embedded in its Qdrant collection; the application has 10 registered dashboard pages and 490 passing automated tests (plus four opt-in MySQL/Qdrant-server tests skipped by the default run).

**Related documents:** [SDD.md](SDD.md) (design, analytical methods, delivery history, review protocols)

This document defines the product and research problem, the decisions the platform must support, and the evidence required to claim that it succeeds. `SDD.md` defines how those requirements are implemented; §12 lists the work still open. Counts above are a dated corpus snapshot, not architectural constants.

## 1. Problem and decision context

The project supports a Systematic Literature Review (SLR) of distribution-system planning in electric power networks. The input is a hand-curated set of IEEE Xplore and Elsevier ScienceDirect exports whose formats, coverage, identifiers, and metadata quality differ. The search term is also polysemous: it retrieves both electric-power planning and logistics/supply-chain research.

The product must help a researcher make four defensible decisions:

1. Which source records represent the same scholarly work?
2. Which works belong in the review, and which require manual screening?
3. What descriptive, bibliometric, engineering, network, and semantic evidence can be supported by the available data?
4. Can every reported result be reproduced from a known source snapshot, transformation version, and set of human decisions?

The system is an evidence-management and analytical support tool. It does not replace protocol design, dual-reviewer screening, domain interpretation, or manuscript authorship.

## 2. Users and outcomes

### 2.1 Primary user

The primary user is the researcher who collects exports, operates the pipeline, reviews exclusions and duplicate candidates, explores the corpus, and prepares an SLR. A successful session ends with an auditable research decision, not merely a chart view.

### 2.2 Secondary users

- A reviewer or collaborator verifying search provenance, screening decisions, and analytical limitations.
- A maintainer diagnosing pipeline failures, drift, or incomplete derived data.
- A retrieval client or LLM agent using Gold chunks to locate evidence while preserving DOI provenance -- served by the `search`/`screen` commands, the MCP server, and the `lit_chunks_active` Qdrant alias (`FR-13`, `FR-14`).

### 2.3 Required outcomes

- A versioned, traceable corpus whose exclusions and merges are reversible.
- Screening decisions calibrated against human labels rather than pseudo-labels.
- Analyses whose population, coverage, assumptions, uncertainty, and limitations are visible.
- A dashboard in which each view answers a distinct question and does not repeat another page's evidence.

## 3. Scope and non-goals

### 3.1 In scope

- Ingestion of IEEE CSV/BibTeX, Elsevier BibTeX, search configuration, and local PDFs.
- DOI normalization, within-source reconciliation, cross-source DOI deduplication, non-article tagging, and human-reviewed cross-DOI merges.
- Local text extraction, chunking, embedding, semantic screening, discovery, and hybrid retrieval.
- Descriptive bibliometrics, scientometrics, engineering-evidence taxonomies, network analysis, and cautious forecasting.
- CLI execution with a read-only analytical dashboard.

### 3.2 Non-goals

- Automated scraping of publisher interfaces or bypassing access controls.
- Treating the corpus as a census of the entire research field.
- Causal claims from observational bibliographic associations.
- Fully autonomous inclusion/exclusion decisions.
- Full-career author metrics or authoritative author identity resolution.
- Production multi-tenancy or arbitrary research topics in the current version.
- Price's index, Sleeping Beauty, CD disruption, Open Access Citation Advantage, or citation longevity before their required longitudinal/reference data exists.

## 4. Research questions and evidence decisions

| ID | Question | Decision informed | Minimum evidence |
|---|---|---|---|
| `RQ-01` | What entered, changed, or left the corpus? | Accept a dataset version for analysis | Source hashes, search provenance, reconciliation report, quality gates |
| `RQ-02` | Which records are duplicates or alternate manifestations? | Merge, keep separate, or investigate | Normalized DOI, metadata comparison, semantic similarity, human rationale |
| `RQ-03` | Which articles are in scope? | Include, exclude, or prioritize review | Human labels, contrastive margin, threshold metrics and uncertainty |
| `RQ-04` | How has publication output evolved? | Describe growth without treating partial years as complete | Complete-year counts, source coverage, uncertainty for forecasts |
| `RQ-05` | Where is research published and concentrated? | Identify core venues and coverage boundaries | Venue normalization, Bradford/concentration diagnostics, Qualis match coverage |
| `RQ-06` | Which concepts and engineering methods structure the field? | Identify established, emerging, or under-studied areas | Validated multi-label taxonomies, keywords, embeddings, time windows |
| `RQ-07` | How are impact and collaboration distributed? | Identify influential works and structural collaboration patterns | Age-aware citations, corpus-scoped author identities, sensitivity analyses |
| `RQ-08` | Is retrieval adequate for evidence discovery? | Approve dense, lexical, or hybrid retrieval for use | Labeled queries and Recall@k, MRR, nDCG, latency, and failure examples |

## 5. Methodological protocol

### 5.1 Population and sampling frame

The sampling frame is the union of manually exported IEEE Xplore and ScienceDirect result sets described by `raw.lit_config`. It is not the complete universe of distribution-planning research. Search dates, queries, filters, publisher coverage, and export limits must accompany every report.

### 5.2 Unit of analysis

- Source-record analyses use one Raw or Bronze record.
- SLR and bibliometric analyses use one curated Gold work after approved merge decisions.
- Retrieval uses one Gold chunk but reports the parent DOI.
- Author analyses use heuristic corpus identities and must not be interpreted as complete careers.
- Network edges and topic observations exist only inside the collected corpus.

No analysis may silently mix these grains.

### 5.3 Inclusion, exclusion, and deduplication

- DOI is normalized by removing DOI URL prefixes, trimming, and casefolding.
- Records without a usable DOI are excluded from Silver and retained in `silver.lit_rejected`; this is a methodological exclusion and a potential selection bias.
- Same-DOI records are consolidated in Silver. Cross-DOI semantic candidates remain distinct until a reviewer records `merge` or `keep` with a rationale.
- Non-article material is tagged, not deleted, so reporting can include or exclude it explicitly.
- PDF matching is probabilistic and requires a validated threshold plus an audit sample before being treated as ground truth.

### 5.4 Screening

Contrastive semantic screening uses the margin between an electric-power anchor and a logistics anchor. A zero margin means “closer to the power anchor,” not “98% recall.” Automated exclusion is prohibited until reviewed labels support the selected threshold.

The target screening protocol is:

1. Draw a reproducible stratified sample covering clear and borderline margins.
2. Collect independent labels from two reviewers using a versioned decision schema.
3. Resolve disagreements and report Cohen's kappa or an equivalent agreement statistic.
4. Select a threshold against sensitivity, specificity, precision, F2, workload, and confidence intervals.
5. Validate on a held-out or later review batch before operational use.

### 5.5 Missingness and coverage

Missing values remain missing unless a transformation is explicitly defined. Each analysis must disclose `n_total`, `n_eligible`, `n_used`, exclusion reasons, and coverage by source/time segment. IEEE-only fields (`countries`, `online_date`, `document_type`, `license`) are never generalized to the full corpus. PDF-based analyses must compare records with and without PDFs to expose availability bias.

**The denominator rule.** Every published figure names the population it was
computed over, and a figure derived from partially collected external data also
reports its coverage per source. An aggregate percentage over a partial
collection silently inherits whatever ordered the collection: on 2026-09-22 the
OpenAlex crawl had observed 989 Elsevier works and zero IEEE ones, purely
because `10.1016` sorts before `10.1109`, and "81.9% annual-count coverage" read
as a statement about the corpus when it described one publisher. A share of an
observed subset and a share of the corpus are different quantities and must be
labelled as such.


### 5.6 Temporal validity and leakage prevention

- Citation/reference counts are point-in-time observations and require an observation timestamp.
- Partial current years are monitoring-only for forecasting and trend fitting.
- Model selection, preprocessing, thresholds, and uncertainty calibration must be fitted within each temporal training fold.
- Threshold selection and final evaluation must not reuse the same manual labels without explicit resampling or a holdout.
- Metrics derived after a corpus snapshot may not be joined to an earlier snapshot without a version check.

### 5.7 Statistical reporting standard

Every inferential or predictive result must report effect size, uncertainty, sample size, diagnostics, and practical interpretation. P-values alone are insufficient. Exploratory analyses must be labeled exploratory, and multiple related tests require false-discovery-rate control or an explicit family-wise strategy.

## 6. Product requirements

### 6.1 Functional requirements

| ID | Requirement | Acceptance signal |
|---|---|---|
| `FR-01` | Ingest supported artifacts without losing their source representation. | Every present input has a manifest entry and reproducible parsed rows. |
| `FR-02` | Detect changed and removed inputs. | Reconciliation reports additions, modifications, removals, and downstream effects. |
| `FR-03` | Produce one Silver row per normalized DOI and audit every exclusion. | DOI uniqueness holds; rejection counts reconcile with Bronze. |
| `FR-04` | Apply persistent, reversible cross-DOI review decisions in Gold. | Merge/keep/undo is idempotent and linked to reviewer rationale and time. |
| `FR-05` | Reconcile chunks without retaining vectors for changed text. | Unchanged text reuses its stored vector by text hash; changed text is re-embedded; points for removed chunks are deleted. |
| `FR-06` | Publish semantic outputs only for a complete compatible embedding set. | Coverage and model/dimension checks pass before atomic replacement. |
| `FR-07` | Record a correlated pipeline run and dataset version. | All stage events share a parent run and input/output version identifiers. |
| `FR-08` | Serve Gold as the canonical analytical population. | Dashboard counts agree with curated Gold and approved merges. |
| `FR-09` | Provide auditable screening calibration. | Labels, reviewer agreement, threshold version, metrics, and holdout results persist. |
| `FR-10` | Provide inspectable lexical, dense, and hybrid retrieval. | Results expose parent DOI and component ranks/scores. |
| `FR-11` | Keep dashboard data access read-only. | Mutations occur only through CLI workflows; no page starts a run. |
| `FR-12` | Degrade safely when a table or optional analysis is unavailable. | Pages show a scoped explanation rather than crashing or inventing results. |
| `FR-13` | Store chunk vectors in a vector store a retrieval client can query, one immutable collection per dataset version. | Every published version has a complete, contract-checked Qdrant collection; external clients find the published one under a stable alias; Gold remains the source of the evidence. |
| `FR-14` | Rank articles, not chunks, for a query, and serve that ranking outside the dashboard. | One hit per article over the abstracts by default; filters applied before ranking, selecting the same articles in the lexical and dense paths; DOI provenance and component ranks/scores in JSON; one `Retriever` behind the CLI, the MCP server and the dashboard. |
| `FR-15` | Embed the paper's own text, not its reference list or page furniture. | Full-text chunks exclude the reference section and page-number lines, rejoin line-break hyphenations, and end at sentence boundaries; each build reports what the cleaning changed. |

### 6.2 Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| `NFR-01` | Reproducibility | A report resolves to source snapshot, code revision, configuration, models, anchors, and decisions. |
| `NFR-02` | Idempotency | Re-running an unchanged version changes neither logical rows nor decisions. |
| `NFR-03` | Integrity | Dataset and stage contracts block publication of incompatible derived data. |
| `NFR-04` | Auditability | Exclusions, overrides, quality failures, and pipeline errors are retained. |
| `NFR-05` | Performance | Current corpus remains responsive without distributed compute; scaling changes require benchmarks. |
| `NFR-06` | Security | Only `lit_*` tables are addressed; secrets are not committed or sent to the browser. A shared or multi-user deployment must restore database-enforced least privilege. |
| `NFR-07` | Accessibility | UI labels are non-empty, English, legible in the fixed dark theme, and usable without color alone. |
| `NFR-08` | Testability | Mathematical logic remains Streamlit-free; UI behavior has headless smoke coverage. |

### 6.3 How each requirement is measured

An acceptance signal states what success looks like; it does not say what to run
to see it. Without that, a reviewer cannot distinguish a requirement that is
verified from one that is merely asserted, and the difference is the whole point
of an evidence-managing system. Each row below names the artifact that proves
its requirement, or says plainly that none exists.

| ID | Proof | Where |
|---|---|---|
| `FR-01` | `raw` contract: non-empty manifest, snapshot coverage, child-source references | `quality.raw_contract`; `audit reference-corpus` |
| `FR-02` | Golden-corpus mutation test (add, edit, rename, remove, reactivate) | `ingest/snapshots.py::diff_manifests`; `tests/test_versioning_contracts.py` |
| `FR-03` | `silver.doi_unique`, `silver.doi_normalized`, `silver.bronze_partition` | `quality.silver_contract`; `tests/test_silver_articles.py` |
| `FR-04` | Idempotent merge/keep/undo with rationale and timestamps | `transform/duplicate_resolution.py`; `tests/test_duplicate_resolution.py` |
| `FR-05` | Reused vs. invalidated chunk counts on an unchanged rebuild; copy-before-compute and orphan deletion in the embed stage | `transform/versioned_gold.py`; `tests/test_gold_articles.py`, `tests/test_vector_collections.py` |
| `FR-06` | Dimension, model, revision, finiteness, orphan and coverage gates on the Qdrant points before publication | `quality.embed_contract`, `quality.publication_contract`; `tests/test_vector_collections.py` |
| `FR-07` | Parent execution and stage rows sharing one version | `tests/test_versioning_contracts.py` |
| `FR-08` | Article reader bound to the publication pointer | `dashboard/data.py::load_articles`; `tests/test_dashboard_pages.py` |
| `FR-09` | Durable labels joined to calibration, with both approval digests | `reviews calibrate`; `tests/test_review_label_reader.py` |
| `FR-10` | Component ranks and scores exposed per mode | `retrieval/rank.py`, `retrieval/api.py::ArticleHit`; `tests/test_retrieval_rank.py`, `tests/test_retrieval_api.py` |
| `FR-11` | No write call anywhere under `dashboard/` | `tests/test_theme.py` |
| `FR-12` | Every page rendered headless in populated and empty states | `tests/test_dashboard_pages.py` |
| `FR-13` | Per-version collection, deterministic point ids, exact search, alias moved after publication, backfill/rebuild repairs | `db/vector_store.py`; `tests/test_vector_store.py`, `tests/test_vector_collections.py`, opt-in `tests/test_qdrant_server.py`; `vectors status` |
| `FR-14` | One hit per article, filters before ranking, pandas/Qdrant filter parity, JSON-only stdout, no pipeline lock, five MCP tools | `retrieval/`; `tests/test_retrieval_api.py`, `tests/test_retrieval_filters.py`, `tests/test_retrieval_cli.py`, `tests/test_mcp_server.py` |
| `FR-15` | Cleaner unit tests and the measured before/after on the 97 linked PDFs; `pdf_*` keys in the Gold stats | `transform/pdf_text.py`; `tests/test_pdf_text.py`, `tests/test_gold_articles.py`; SDD §14 |
| `NFR-01` | Source/config/code/curation fingerprints; per-page dataset-version caption | `ingest/snapshots.py::build_fingerprint`; `tests/test_population_provenance.py` |
| `NFR-02` | Unchanged re-run produces no new logical rows | `tests/test_gold_articles.py`, `tests/test_raw_bib.py` |
| `NFR-03` | Contract violation blocks publication and keeps the prior version active | `quality.assert_contract`; `tests/test_mysql_contracts.py` |
| `NFR-04` | Rejections, overrides, quality results and run errors persisted | `lit_rejected`, `lit_quality_results`, `lit_pipeline_runs` |
| `NFR-05` | **Not measured.** Latency and memory were benchmarked once by hand (2026-09-21); no performance test guards them | SDD §14: 11.06 MiB vector matrix, 92.72 ms warm-search P95 on 2026-09-21 |
| `NFR-06` | Secret hygiene only. The per-table privilege boundary was traded away on 2026-09-22 and is **not** enforced | `.gitignore`; `WP-08` evidence |
| `NFR-07` | Portuguese-shaped copy, empty headings and empty metric labels all fail the build | `tests/test_dashboard_language.py` |
| `NFR-08` | `analytics.py` and `forecasting.py` import no Streamlit; every page has `AppTest` coverage | `tests/test_theme.py`, `tests/test_dashboard_pages.py` |

Two rows say "not measured" on purpose. `NFR-05` and the roles half of `NFR-06`
are currently aspirations, and writing them as though they were verified is the
failure mode this table exists to prevent.

## 7. Analytical ownership and visualization contract

Each analytical question has exactly one canonical page. Overview may link to a deep dive but must not reproduce its chart.

| Page | Exclusive decision/question | Canonical evidence | Planned hardening |
|---|---|---|---|
| **Overview** | What is the corpus state now? | Counts, recency, source composition, readiness | Deep correlation/concentration/temporal views removed; add snapshot/as-of status |
| **Production and venues** | How did output and qualified venue composition change? | Annual and cumulative series, Qualis evolution | Explicit complete/partial-year status; no generic venue ranking duplication |
| **Topics and scientific structure** | What concepts and venues structure the field? | Keyword prevalence, Bradford/Zipf, c-TF-IDF, semantic venue groups | FDR now covers the full keyword family; the growth ranking, slope chart, and searched-breakpoint test still report no adjustment |
| **Impact and citations** | How is impact distributed and conditionally associated? | Citation distributions, age normalization, count models, specification diagnostics | Delivered: AIC family comparison, bootstrap tail p-value, VIF/condition number, influence, zero-inflation gap. Open: a fitted zero-inflated model and alternative age specifications |
| **Researchers and collaboration** | How are corpus authors and ties organized over time? | Productivity, Lotka's law, corpus indices, network structure vs. a degree-preserving null | Identity ambiguity audit; assortativity; temporal formation/repetition of ties |
| **Engineering evidence** | Which methods, objectives, uncertainties, networks, and tools occur? | Multi-label engineering taxonomy | Labeled audit set, precision/recall, unknown/ambiguous coverage |
| **Trends and fronts** | What can be projected, with what historical error? | Persistence-baseline skill, rolling-origin forecast, held-out interval coverage, Bass/burst diagnostics | Skill and coverage by horizon (every fold is currently one-step), MASE, Bass parameter stability |
| **Screening and discovery** | What should be reviewed, included, or reconciled? | Margin, themes, projections with bootstrap ARI and trustworthiness, isolation, duplicate queue, persistent review protocols/labels | Collect independent labels; persist stability; sweep cluster count |
| **Quality and RAG** | Are metadata, text, embeddings, and retrieval fit for use? | Coverage matrix, PDF selection bias, chunk diagnostics, anomaly audit, inspectable dense/BM25/RRF search | Apply the implemented Recall@k/MRR/nDCG/latency harness to labeled technical queries |
| **Pipeline and provenance** | Can results be traced and trusted operationally? | Funnel, runs, rejections, source configuration, quality gates | Snapshot, freshness, removal reconciliation |

Visualization selection follows the question: bars for discrete comparisons, lines for ordered time, ECDF/CCDF for distributions, scatterplots for associations with uncertainty, heatmaps for dense matrices, and tables when exact values or audit context dominate. Multiple chart types for the same knowledge are alternatives behind one selector, not separate claims.

## 8. Analytical validity requirements

### 8.1 Citation distributions and models

- Heavy-tail fitting must estimate or justify `x_min`, compare plausible families using likelihood-based evidence, and bootstrap goodness-of-fit because parameters are estimated from the same sample.
- Count models must assess missingness, overdispersion, zero inflation, multicollinearity, influential observations, specification sensitivity, and temporal exposure.
- Coefficients are associations, not causal effects. Source indicators also encode source-specific missingness and collection differences.

### 8.2 Trends and forecasts

- Trend grids include zero-count years rather than silently skipping them.
- Related keyword tests use adjusted p-values and minimum prevalence/sample rules.
- Breakpoint significance accounts for breakpoint search or is labeled exploratory.
- Forecasts always compare against persistence; report MAE and empirical interval coverage by horizon; do not extrapolate partial-year suppression.
- Bass fits require admissible parameters, uncertainty/sensitivity evidence, and enough temporal support.

### 8.3 Semantic and unsupervised models

- Embedding model, revision, dimension, text hash, anchor version, random seed, and corpus version are recorded.
- Cluster count selection is complemented by bootstrap stability and sensitivity to preprocessing.
- Projection quality reports trustworthiness/neighborhood preservation; 2D distance is not treated as high-dimensional evidence.
- Semantic isolation is an audit/discovery signal, not validated interdisciplinarity.

### 8.4 Networks, authors, and taxonomies

- Author metrics remain corpus-scoped; identity collisions and splits are sampled and disclosed.
- Weighted network distances use an explicit strength-to-distance transform; disconnected components and null-model choices are reported.
- Regex taxonomies are multi-label classifiers. Coverage, ambiguous matches, false positives, and false negatives require a labeled audit sample.

### 8.5 Retrieval

- No retrieval-quality claim is valid without labeled queries and relevance judgments.
- Dense, BM25, and RRF variants are compared on identical query sets using Recall@k, MRR, nDCG, latency, and qualitative failure analysis.
- Full-text coverage is reported separately because PDF availability changes the candidate passage population.
- Articles rank over their abstracts by default: one chunk per article, so every article competes on equal terms. Full text ranks only on request (`scope="all"`, labelled as favouring articles with a PDF) and otherwise serves as evidence passages for an article already found.
- Filters apply before ranking and select the same articles in the lexical (pandas) and dense (Qdrant) paths; an article with no screening score is never filtered out by the margin filter.
- The cross-encoder reranker is experimental. Neither it nor any other mode becomes the default until the `WP-13` human relevance labels support it.

## 9. Current evidence boundaries and risks

| ID | Risk | Consequence | Required mitigation |
|---|---|---|---|
| `RISK-01` | Search coverage is limited and manually exported. | Results may omit relevant literature. | Preserve full search provenance and report the sampling frame. |
| `RISK-02` | No-DOI records are excluded. | Systematic selection bias is possible. | Audit rejected records and evaluate alternative stable identifiers. |
| `RISK-03` | PDFs cover a small, non-random subset. | Full-text analyses may be biased. | Compare PDF/non-PDF groups and label abstract-only conclusions. |
| `RISK-04` | Citation counts are unversioned snapshots. | Results drift and cannot be reproduced precisely. | Persist source and observation time before longitudinal claims. |
| `RISK-05` | Author names are heuristic identities. | Homonyms merge and variants split. | Add identity audit/overrides before person-level conclusions. |
| `RISK-06` | Screening anchors lack independent gold-standard validation. | Threshold performance may be overstated. | Persist dual-review labels and validate out of sample. |
| `RISK-07` | Many analytical panels invite multiple testing. | False discoveries become likely. | Define test families, effect-size thresholds, and FDR control. |
| `RISK-08` | An expensive tab body could again execute while hidden. | Reruns slow down. | Mitigated: every tab group renders only its open branch (`st.tabs(..., on_change="rerun")`); keep new tabs gated. |
| `RISK-09` | External enrichment is collected under a request quota and may stay partial indefinitely. | A partial collection ordered by any key correlated with source produces a biased sample, and coverage reported in aggregate hides it. | Collect in publisher-proportional order so any prefix is representative; report coverage per registrant against the corpus denominator; treat an unfinished crawl as a stated limitation, never as missing data at the provider. |

## 10. Success criteria

The product is successful when all of the following are evidenced, not merely asserted:

- Every published analytical result identifies a dataset version and population coverage.
- Raw-to-Gold counts reconcile, including changed and removed inputs.
- Gold contains no duplicate DOI and approved cross-DOI decisions are reproducible and reversible.
- Semantic outputs are rejected when embedding coverage/model compatibility is incomplete.
- Screening evaluation reports human-label agreement and uncertainty; no automatic exclusion is enabled without the approved sensitivity target.
- Retrieval evaluation demonstrates the chosen method against lexical and dense baselines on a labeled query set.
- Forecasts beat or explicitly fail to beat persistence and report empirical interval coverage.
- Every dashboard chart maps to one question in the ownership table; no other page repeats that knowledge.
- The full automated suite and lint/format checks pass, with future Streamlit behavior covered through `st.testing.v1.AppTest`.

## 11. Decision record

| Decision | Evidence and rationale | Status |
|---|---|---|
| Use DOI as the principal conformed work key. | It is the most reliable shared identifier, but no-DOI exclusions remain auditable. | Implemented |
| Keep cross-DOI semantic duplicates human-reviewed. | Similar abstracts can represent versions or genuinely distinct publications. | Implemented |
| Use local BGE embeddings and binary float32 storage. | Appropriate for corpus scale and offline operation; performance claims require benchmarks. | Superseded for storage on 2026-11-06 |
| Store vectors in Qdrant, one collection per dataset version (ADR-08). | A RAG client outside the dashboard needs filtered nearest-neighbour search behind a standard API; Gold keeps the evidence and the publication pointer. | Implemented |
| Run the pipeline from the CLI only; remove Airflow (ADR-09). | One researcher starts runs by hand; the orchestrator's container, REST triggers and retry policies served no schedule. | Implemented on 2026-11-06 |
| Keep dashboard pages read-only. | Separates analytical presentation from controlled pipeline mutations. | Implemented |
| Treat contract-valid Gold as the canonical analytical population. | Gold is where approved merges and curated chunks exist; invalid or absent Gold falls back visibly to Silver/Bronze. | Implemented: readers bind to the publication pointer (`WP-04`) |
| Keep uploaded screening calibration non-persistent and non-operative. | The current phase can validate the method without allowing a dashboard session to mutate corpus decisions. | Superseded: labels persist in `lit_review_labels` and feed calibration (`WP-09`/`WP-10`); the approval decision awaits human labels (§12) |
| Version corpus, enrichment, models, anchors, and labels. | Necessary for reproducible statistical and semantic results. | Implemented: dataset versions (`WP-01`), enrichment observations (`WP-07`), model revision and anchor hashes (`WP-06`), versioned review protocols and labels (`WP-09`) |
| Trust `data/enrichment_cache.json` as a citation source. | Validated against OpenAlex on 2026-09-22 over 1,875 comparable rows: reference counts agree 99.7%, citation counts 90.0% with a median error of 0. Every divergence is one-directional -- the cache is never higher than OpenAlex, only older -- which is the signature of staleness rather than error. | Validated; refresh before longitudinal claims |
| Add new methods only when they answer a distinct decision question and meet data prerequisites. | Prevents dashboard breadth from exceeding evidential validity. | Active governance rule |

## 12. Open work

Everything not listed here is delivered; its history is in SDD §13. The ids are kept because code, tests and
dashboard captions cite them. Each item is blocked on human evidence, not on code: the review queues export the
resolved context each reviewer needs, and the protocols are SDD Appendix A.

| Id | Title | What is missing | Completion |
|---|---|---|---|
| `WP-05` | Identity, PDF matching, and rejection validation | Reviewed stratified samples of PDF links, duplicate decisions and no-DOI rejections (protocols `pdf-v1`, `screening-v2`), so the matcher, its threshold and the rejection bias are chosen from measured precision/recall. | The canonical matcher/scorer is selected from evidence, its threshold is versioned, and errors/ambiguous cases are retained for review. |
| `WP-09` | Persistent dual-review screening workflow | The workflow is delivered (protocol `screening-v2`, 60 assignments over 30 no-DOI subjects); the two independent reviewers' labels are not. | Two reviewers label the same version independently; adjudication preserves raw labels; agreement and class prevalence are reproducible. |
| `WP-10` | Screening threshold validation | Calibration and the hash-bound approval are delivered; the approval decision and a later-batch validation wait for `WP-09`'s labels. | The selected threshold meets the approved sensitivity target on validation data, or automatic exclusion stays disabled with the failure documented. |
| `WP-11` | Author identity audit and overrides | Human decisions on the ambiguity sample (protocol `author-v1`), then reversible identity overrides and an audited error rate. | Person-level panels use the resolved identity version and show unresolved ambiguity coverage; corpus metrics can be recomputed after undo. |
| `WP-13` | Retrieval evaluation corpus | Two independent human reviews of the pooled sample (protocol `retrieval-v1`); only an AI-assisted pass exists (96 labels), so no metric or default mode is approved. Regenerate the pool with `evidence retrieval` to include the article-level modes. | All retrieval modes run on identical judgments; the default is selected by an explicit metric/latency rule and reproduced in tests. |
| `WP-21` | Integrate validated analyses into existing pages | The taxonomy-validation (`taxonomy-v1`) and identity-audit panels, which wait for `WP-11`/human taxonomy labels. | Each addition references one `RQ-*`, exposes population/coverage/limitations, and has no equivalent view elsewhere. |

## 13. Definition of done for a change

Every change that alters data or analytical behaviour includes:

- additive migration and a rollback/recovery procedure;
- SQLite unit coverage, plus MySQL or Qdrant integration coverage where store semantics matter;
- fixtures for missing, degenerate and failure states;
- updates to this document and `SDD.md` where a requirement, design decision or measured fact changes;
- dashboard coverage and English UI copy when user-visible;
- `uv run pytest`, `uv run ruff check` and `uv run ruff format --check` passing;
- the measured baseline, result and residual risk recorded in SDD §14.

## 14. Deferred ideas

Not scheduled until a decision need and an evidence contract exist:

- New standalone dashboard pages.
- Generative summaries presented as research evidence.
- Causal ranking of venues, authors, or methods.
- Deep-learning forecasts on short annual series.
- Further infrastructure without a measured need. (Qdrant was adopted under `ADR-08` for a consumer
  requirement, not novelty.)
- Automatic merging or screening without human-reviewed validation.

## 15. Requirements traceability

| Requirements | Architecture decisions | Open work |
|---|---|---|
| `FR-01`, `FR-02`, `NFR-01`, `NFR-02` | `ADR-03` | — |
| `FR-03`, `FR-04`, `FR-08`, `NFR-04` | `ADR-02`, `ADR-05` | `WP-05` |
| `FR-05`, `FR-06`, `NFR-03` | `ADR-03`, `ADR-04` | — |
| `FR-07`, `NFR-01`, `NFR-04` | `ADR-03`, `ADR-09` | — |
| `FR-09` | `ADR-05` | `WP-09`, `WP-10` |
| `FR-10`, `FR-14`, `NFR-05`, `RQ-08` | `ADR-04`, `ADR-08` | `WP-13` |
| `FR-11`, `NFR-06` | `ADR-01`, `ADR-06`, `ADR-09` | — |
| `FR-12`, `NFR-07`, `NFR-08` | `ADR-02`, `ADR-06` | `WP-21` |
| `FR-13` | `ADR-03`, `ADR-08` | — |
| `FR-15` | `ADR-03` | — |
| `RQ-04` through `RQ-07` | `ADR-02`, `ADR-04`, `ADR-05` | `WP-11`, `WP-21` |
