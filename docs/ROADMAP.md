# Engineering and Research Roadmap — lake-research-map

**Status:** Prioritized delivery plan

**Planning baseline:** 2026-09-21

**Related documents:** [PRD.md](PRD.md), [SDD.md](SDD.md), [METHODOLOGY.md](METHODOLOGY.md)

This roadmap prioritizes evidential reliability before adding analytical breadth. A work package is complete only when its deliverable and acceptance evidence exist in the repository; implemented code without validation, migration, or documentation is not complete.

## 1. Prioritization model

Priority follows five rules:

1. Prevent incorrect or irreproducible evidence before improving presentation.
2. Establish human ground truth before optimizing models against it.
3. Validate a method before adding its visualization.
4. Remove redundant or unsupported analyses before introducing new ones.
5. Add infrastructure only when a measured limit justifies its operational cost.

Dependencies use work-package IDs. Horizons are sequencing bands rather than calendar promises:

- **Short term:** data reliability and methodological foundations.
- **Medium term:** statistical hardening and dashboard consolidation.
- **Long term:** new external data and scale-dependent capabilities.

## 2. Delivered baseline

The following capabilities are implemented and covered by the current 229-test suite (241 before `WP-20` removed the tests of unreachable analytics, plus regression tests for the defects this cycle fixed):

- Raw/Bronze/Silver/Gold ingestion and transformations with DOI normalization and rejection audit.
- Local PDF inventory/matching, full-text extraction, chunk reconciliation, and local BGE embeddings.
- Binary float32 vectors as the single canonical representation, dense/BM25/RRF retrieval, and optional Faiss indexing. The JSON mirror is no longer written.
- Contrastive screening signals, KMeans themes, PCA/t-SNE/optional UMAP, novelty/isolation, and duplicate candidates.
- Persistent, reversible cross-DOI `merge`/`keep` decisions applied in Gold.
- Pipeline-run history and seven Airflow DAGs matching the six stages plus the complete flow.
- Ten workflow-oriented Streamlit pages with global filters, theme support, and read-only analytical access.
- Contract-valid Gold-first analytical selection with an explicit Silver/Bronze degraded mode.
- Canonical publication-category classification (`journal`, `conference`, `review`, `other`) propagated through Bronze, Silver, Gold, global filters, and the three-section Production and venues view.
- Source-denominated metadata completeness, PDF-selection-bias effect sizes, and in-memory human-label screening calibration with holdout evaluation.
- Bibliometric, network, engineering-taxonomy, burst, forecasting, and anomaly functions described in `METHODOLOGY.md`.
- Analytical self-disclosure on the four pages that carry inferential claims: AIC family comparison and a bootstrap goodness-of-fit for the citation tail; VIF, condition number, influence and zero-inflation diagnostics beside the count model; bootstrap ARI and projection trustworthiness beside the semantic map; a degree-preserving null model beside the collaboration network; and persistence skill plus held-out interval coverage beside every forecast.
- One canonical owner per analytical question: no public analytical function is unreachable from a page, and no page claims a method the code does not implement.

This baseline does not imply that every analytical method is confirmatory. The limitations and required validation are explicit in `PRD.md`.

## 3. Short term — trustworthy data and reproducible runs

### `WP-01` — Dataset versions and correlated pipeline runs

- **Status:** Complete on 2026-09-21; SQLite contracts, MySQL 8.4 bootstrap, and active-corpus audit passed.
- **Objective:** Make every result resolve to an immutable input/output version and one parent execution.
- **Justification:** The former stage records did not identify a shared `all` run, code revision, configuration, or active dataset snapshot.
- **Dependencies:** None.
- **Deliverable:** Additive version/run schema; parent and stage IDs; source-manifest/config/code hashes; active/published/failed state; lineage in run metrics.
- **Completion:** Two unchanged full runs produce the same logical version; a changed file creates a new version; every stage/output can be traced to its parent run.
- **Evidence:** Deterministic source/config/code/curation fingerprints, parent executions, stage attempts, active/working publication pointers, Airflow correlation, and no-change stage records are covered by SQLite acceptance tests. Completion still requires the MySQL/reference-corpus evidence defined in Section 9.

### `WP-02` — Source reconciliation and deletion propagation

- **Status:** Complete on 2026-09-21; SQLite contracts, MySQL 8.4 bootstrap, and active-corpus audit passed.
- **Objective:** Detect input additions, modifications, and removals and propagate them deterministically.
- **Justification:** Removed files and their records previously survived in Raw and Bronze.
- **Dependencies:** `WP-01`.
- **Deliverable:** Immutable source revisions or tombstones, active-snapshot reconciliation, stale Raw/Bronze cleanup, and removal metrics.
- **Completion:** Golden-corpus tests add, edit, rename, and remove files; downstream rows/counts reconcile exactly and a rollback can reactivate the previous version.
- **Evidence:** Content-addressed source retention, immutable revisions/manifests, per-file change audit, transactional Raw cleanup, file-qualified Bronze keys, full Bronze rebuild, and exact Gold snapshot reactivation are covered by an end-to-end SQLite golden-corpus mutation test (add, edit, rename, remove, and reactivate). MySQL/reference-corpus evidence remains required for completion.

### `WP-03` — Executable data contracts and publication gates

- **Status:** Complete on 2026-09-21; persisted gates report zero blocking failures for the active version.
- **Objective:** Turn layer invariants into persisted, blocking quality checks.
- **Justification:** Warnings and dashboard diagnostics previously did not prevent incomplete derived data from becoming visible.
- **Dependencies:** `WP-01`, `WP-02`.
- **Deliverable:** `lit_quality_results`, severity policy, uniqueness/referential/coverage/range checks, stage gate API, and dashboard contract status.
- **Completion:** Required-check failure keeps the prior version active, records expected versus observed values, and returns a failed pipeline status.
- **Evidence:** Persisted Raw/Bronze/Silver/Gold/Embed/Semantic contracts block publication in automated tests; candidate Gold tables are isolated until atomic materialization; the existing pipeline page exposes versions, execution lineage, file changes, and gate outcomes. MySQL transaction and BLOB/JSON behavior still require integration evidence.

### `WP-04` — Gold as the canonical analytical population

- **Status:** Complete on 2026-09-21; analytical readers bind articles, chunks, semantics, duplicate candidates, search, and projections to the active immutable version.
- **Objective:** Ensure approved duplicate curation is reflected throughout the dashboard.
- **Justification:** Gold-first selection now protects approved merges, but readiness is not yet tied to an immutable published dataset version or persisted quality-gate result.
- **Dependencies:** `WP-03`.
- **Deliverable:** Gold readiness contract; propagation/join of required quality fields; explicit degraded pre-Gold mode; updated loaders and page coverage captions.
- **Completion:** Dashboard article counts and DOI sets equal the active Gold version; merge/undo changes appear after the documented rebuild sequence; degraded mode is visibly labeled.

### `WP-05` — Identity, PDF matching, and rejection validation

- **Status:** `blocked-awaiting-evidence`; the persistent review workflow is available, but no reviewed identity/PDF/rejection sample has been supplied.

- **Objective:** Quantify false merges, missed duplicates, PDF mislinks, and the bias introduced by no-DOI rejection.
- **Justification:** DOI and fuzzy-title rules are high-impact methodological decisions currently tested mainly for mechanics.
- **Dependencies:** `WP-01`.
- **Deliverable:** Reviewed stratified audit sets; comparison of `token_sort_ratio` and `token_set_ratio`; precision/recall with confidence intervals; documented threshold; sampled no-DOI disposition.
- **Completion:** The canonical matcher/scorer is selected from evidence, its threshold is versioned, and errors/ambiguous cases are retained for review.

### `WP-06` — Embedding and Semantic compatibility contract

- **Status:** Complete on 2026-09-21; the contract is enforced, the corpus was re-embedded to satisfy it, and the JSON fallback is retired.

- **Objective:** Prevent missing, stale, mixed-model, or dimensionally invalid vectors from feeding semantic outputs.
- **Justification:** Embed selects JSON-null rows while binary is canonical, and Semantic publishes partial coverage after a warning.
- **Dependencies:** `WP-01`, `WP-03`.
- **Deliverable:** Text hash, model revision, dimension/dtype checks, binary-first pending logic, JSON fallback migration, semantic-run metadata, and atomic complete-coverage gate.
- **Completion:** JSON-only, wrong-length, changed-text, mixed-model, and partial-coverage fixtures all fail or repair deterministically; Semantic never replaces a valid complete run with an incomplete one.
- **Evidence:** The previously active version carried none of the seven metadata fields on any of its 7,552 chunks, so it could not have passed the contract that `quality.py::embed_contract` now enforces. There is no in-place backfill and there should not be -- `_assert_mutable_candidate` refuses to write a published version -- so the repair was recompute: the candidate's pending selector caught every chunk on `embed_revision IS NULL`, and a full run re-embedded the corpus. The JSON mirror is no longer written (`versioned_gold.py`) or read (`search.py`, `data.py`). The legacy `build_embeddings` path and the dashboard action that called it were removed outright: they wrote the live `lit_chunks` table that publication rebuilds, so their vectors were discarded on the next publish and never met the contract. Embedding now has exactly one writer.

### `WP-07` — Enrichment observations and temporal semantics

- **Status:** Append-only observation, as-of selection, response provenance, and OpenAlex evidence schema implemented; credentialed corpus refresh remains open.

- **Objective:** Make citation/reference data reproducible and refreshable.
- **Justification:** The current cache stores only latest counts and cannot support an as-of analysis.
- **Dependencies:** `WP-01`.
- **Deliverable:** Provider observations with `observed_at`, status, identifiers, counts, response hash, retry metadata, and deterministic selection of the analysis snapshot.
- **Completion:** A refresh appends an observation without rewriting history; the same dataset/as-of selection reproduces the same citation inputs.

### `WP-08` — Concurrency, recovery, and operational security

- **Status:** Advisory locking, persistent logs, signal handling, resumable embeddings, heartbeat recovery, Airflow retry/timeouts, local-only ports, resource bounds, and per-table role provisioning delivered; role creation awaits local passwords.
- **Objective:** Make mutation predictable under failure and prevent unsafe overlapping runs.
- **Justification:** Writer serialization and resumable embeddings now protect the common local failure path, but automatic retry/timeout policy, abandoned-run recovery, and shared-network hardening are not complete.
- **Dependencies:** `WP-01`.
- **Deliverable:** Project advisory lock, stage retry/timeout matrix, reliable terminal run status, least-privilege database roles, deployment-mode security guidance, and audited trigger identity.
- **Completion:** Concurrent-run and injected-failure integration tests prove one active writer, safe retry behavior, visible telemetry failure, and unchanged prior published state.

## 4. Short to medium term — human ground truth

### `WP-09` — Persistent dual-review screening workflow

- **Status:** Persistent protocols, dual assignments, append-only label revisions, CSV export/import, and adjudication delivered; independent reviewer labels remain open.
- **Objective:** Calibrate screening against reproducible human judgments.
- **Justification:** The dashboard now measures agreement from uploaded decisions, but labels, assignments, and adjudication are not durable or dataset-versioned.
- **Dependencies:** `WP-01`, `WP-04`, `WP-06`.
- **Deliverable:** Label/adjudication schema, protocol version, deterministic assignments, independent reviewer exports/imports, disagreement queue, and Cohen's kappa/agreement report.
- **Completion:** Two reviewers can label the same version independently; adjudication preserves raw labels; agreement and class prevalence are reproducible.

### `WP-10` — Screening threshold validation

- **Status:** Calibration and durable hash-bound approval schema/CLI delivered; approval and later-batch validation await independent labels.
- **Objective:** Select a workload-aware threshold with uncertainty and out-of-sample evidence.
- **Justification:** Zero contrastive margin is meaningful geometrically but has no guaranteed recall.
- **Dependencies:** `WP-09`.
- **Deliverable:** PR analysis, sensitivity/specificity/precision/F2, bootstrap confidence intervals, calibration/validation split, threshold record, and manual-review workload estimate. ROC is intentionally omitted because it duplicates the screening trade-off under class imbalance.
- **Completion:** The selected threshold meets the approved sensitivity target on validation data or automatic exclusion remains disabled with the failure documented.

### `WP-11` — Author identity audit and overrides

- **Status:** `blocked-awaiting-evidence`; reviewer persistence is available, while resolved author overrides and an audited error rate require human decisions.

- **Objective:** Bound errors from heuristic author canonicalization.
- **Justification:** Homonyms can merge and spelling variants can split, invalidating rankings and network structure.
- **Dependencies:** `WP-04`.
- **Deliverable:** Ambiguity candidate generation, reviewed audit sample, reversible identity overrides, and error-rate disclosure.
- **Completion:** Person-level panels use the resolved identity version and show unresolved ambiguity coverage; corpus metrics can be recomputed after undo.

### `WP-12` — Engineering-taxonomy validation

- **Status:** `blocked-awaiting-evidence`; taxonomy labels can be persisted, but class-level evaluation requires the reviewed sample.

- **Objective:** Treat regex extraction as a measured multi-label classifier.
- **Justification:** Frequency charts currently lack precision/recall evidence and an explicit unknown class.
- **Dependencies:** `WP-04`.
- **Deliverable:** Stratified labeled articles, label guide, per-class precision/recall/F1, ambiguous/unclassified reporting, and refined non-overlapping rules where justified.
- **Completion:** Each displayed taxonomy reports evaluated coverage and meets a declared minimum precision or is labeled exploratory.

### `WP-13` — Retrieval evaluation corpus

- **Status:** `blocked-awaiting-evidence`; retrieval judgments can be persisted, but Recall@k/MRR/nDCG and default-mode selection require labeled technical queries.

- **Objective:** Choose dense, lexical, or hybrid retrieval from measured relevance and latency.
- **Justification:** RRF is implemented, but no labeled query set supports a quality claim.
- **Dependencies:** `WP-04`, `WP-06`.
- **Deliverable:** Versioned technical queries, pooled judgments, abstract/full-text strata, Recall@k, MRR, nDCG, latency, and failure taxonomy.
- **Completion:** All retrieval modes run on identical judgments; the default is selected by an explicit metric/latency rule and reproduced in tests.

## 5. Medium term — statistical and model hardening

### `WP-14` — Citation distribution inference

- **Status:** Complete on 2026-09-21; the estimator and its dashboard panel both ship.

- **Objective:** Replace descriptive best-KS selection with defensible tail comparison.
- **Justification:** Current KS p-values reuse fitted data, force `x_min` to the observed minimum, and omit zeros.
- **Dependencies:** `WP-07`.
- **Deliverable:** Documented zero handling, `x_min` selection/sensitivity, MLE comparisons, likelihood ratios, bootstrap goodness-of-fit, and uncertainty for parameters.
- **Completion:** Simulation tests recover known generating families at acceptable rates; the dashboard presents CCDF/diagnostics once without duplicate histograms.
- **Evidence:** Zero handling, a KS-minimising `x_min` sweep, tail MLE, AIC-based family selection, likelihood ratios, and a refit-per-sample bootstrap are implemented in `analytics.py::fit_heavy_tail_distributions`. The Impact page now reports the AIC comparison table, the tail/zero population, and the bootstrap p-value separately from the reused-parameter KS p-values that only apply to the log-normal and exponential fits. A Vuong test for the likelihood ratios is deliberately not claimed; the ratios are labelled descriptive.

### `WP-15` — Citation count-model diagnostics

- **Status:** Complete on 2026-09-22; the family is selected by AIC across four candidates and the age specification is reported as a sensitivity rather than assumed.

- **Objective:** Report robust associations without hiding misspecification or selection effects.
- **Justification:** The current Poisson/NB choice uses a heuristic dispersion rule and lacks multicollinearity, influence, zero-inflation, and sensitivity diagnostics.
- **Dependencies:** `WP-07`.
- **Deliverable:** Missingness profile, VIF/condition number, residual/influence checks, Poisson/NB/zero-inflated comparison when identifiable, alternative age specifications, and coefficient forest with CIs.
- **Completion:** Synthetic/fixture tests cover convergence and known coefficients; unsupported models are rejected; the UI states association rather than causation.
- **Evidence:** Missingness profile, condition number, VIF, Cook's-distance influence count, observed-versus-predicted zero fraction, and the candidate AICs are computed in `analytics.py::citation_determinants_glm` and displayed in the Impact page's specification-diagnostics panel. The dispersion > 1.5 heuristic is gone: Poisson, negative binomial, ZIP and ZINB are all fitted and the minimum AIC wins, with `zero_inflated_status` recording when no zero-inflated candidate converged. A candidate whose Hessian could not be inverted is refused outright rather than selected -- it has point estimates but NaN intervals and p-values, and a family that cannot state its own uncertainty must not win on AIC. Cook's distance is likewise reported against the Poisson GLM with `influence_basis` naming it, because a zero-inflated MLE fit has no hat matrix and would otherwise return a NaN dressed as a diagnostic. The fixed `log(age + 1)` offset is now one of three specifications -- offset, free log-age covariate, free linear-age covariate -- and `age_specification_signs_agree` drives an explicit warning when a coefficient changes sign between them.

### `WP-16` — Multiple testing and temporal trend validity

- **Status:** Partially delivered; the keyword trend family is now corrected as a whole, while the other ranked surfaces and serial-dependence sensitivity remain open.

- **Objective:** Control false discoveries across keyword/topic trend panels.
- **Justification:** Many Mann-Kendall and breakpoint tests are interpreted independently.
- **Dependencies:** `WP-04`.
- **Deliverable:** Test-family definitions, minimum prevalence, Benjamini-Hochberg adjusted values, effect-size thresholds, serial-dependence sensitivity, and exploratory breakpoint correction/bootstrap.
- **Completion:** Every ranked trend table shows raw effect, uncertainty, adjusted significance, sample span, and zero-filled years.
- **Evidence:** Benjamini-Hochberg previously adjusted only the 7 most-positive and 7 most-negative slopes -- a family already selected for being extreme, which inflates significance rather than controlling it. Mann-Kendall now runs over every keyword meeting the minimum-prevalence rule, the adjustment is applied to that full family, and only then is the table narrowed to the charted terms; the caption states the family size. Still open: the keyword growth ranking on Trends, the OLS slope chart that drives the selection, and the searched-breakpoint Chow test all report no uncertainty or adjustment, and no serial-dependence correction exists anywhere.

### `WP-17` — Semantic stability and projection diagnostics

- **Status:** Partially delivered; bootstrap ARI and trustworthiness are computed and displayed beside the map, while persistence and cluster-count sensitivity remain open.

- **Objective:** Separate robust high-dimensional structure from unstable 2D presentation.
- **Justification:** Silhouette-selected KMeans and t-SNE/UMAP views can change with samples and parameters.
- **Dependencies:** `WP-06`.
- **Deliverable:** Bootstrap/subsample ARI, cluster-count sensitivity, projection trustworthiness, neighborhood preservation, seed stability, and drift computed in embedding space before visualization.
- **Completion:** Theme/novelty panels show stability and coverage; unstable labels remain numbered/exploratory rather than receiving fixed ontological names.
- **Evidence:** `analytics.py::semantic_stability_diagnostics` computes subsample/seed bootstrap ARI and projection trustworthiness, and the Screening page's stability panel reports both next to the projection along with the population used and an explicit warning that theme numbering carries no ontological claim. Still open: nothing is persisted to a table, there is no cluster-count sweep, no neighbourhood-preservation metric beyond trustworthiness, and no embedding-space drift measure.

### `WP-18` — Forecast and Bass validation

- **Status:** Partially delivered; the complete-year cutoff, baseline skill, and an out-of-sample coverage metric now ship, while by-horizon backtests, MASE, and Bass stability remain open.

- **Objective:** Quantify whether projections improve on persistence and whether intervals cover future observations.
- **Justification:** Short annual series can make model selection and conformal bands unstable.
- **Dependencies:** `WP-07`, `WP-16`.
- **Deliverable:** Expanding-window backtests by horizon, MAE/MASE or baseline skill, empirical interval coverage/width, dynamic complete-year cutoff, and Bass parameter bootstrap/sensitivity.
- **Completion:** Forecasts that do not outperform persistence are labeled accordingly; displayed intervals meet the declared backtest coverage tolerance or carry a warning.
- **Evidence:** The cutoff was already dynamic (`LAKE_RESEARCH_MAP_COMPLETE_YEAR`, defaulting to the previous calendar year); the remaining hard-coded years in the forecast path -- the keyword baseline year, the ranking axis, and the CV label -- now derive from it. Skill against persistence is displayed and says "No better than naive" when it is not positive. Interval coverage was scored on the same errors that set the conformal radius, so it returned ~0.9 by construction and could never fail; it is now measured on held-out folds and returns nothing when the series is too short to spare any. Still open: every fold scores one step ahead, so the two-year horizon is unvalidated; there is no MASE; and the Bass fit discards its covariance, so no parameter uncertainty is available.

### `WP-19` — Network null models and temporal collaboration

- **Status:** Partially delivered; the degree-preserving null model ships and does not depend on WP-11. Assortativity and periodized tie dynamics remain open, and every claim stays bounded by heuristic author identity.

- **Objective:** Distinguish structural collaboration evidence from corpus-size artifacts.
- **Justification:** Static centralities and a single random comparison do not describe network uncertainty or evolution.
- **Dependencies:** `WP-11`.
- **Deliverable:** Component-aware metrics, degree-preserving null models, assortativity/robustness checks, and periodized new-versus-repeated collaboration ties.
- **Completion:** Network claims include identity coverage, component scope, null distribution, and sensitivity to the selected author subset.
- **Evidence:** `analytics.py::network_null_model_diagnostics` rewires the graph preserving every author's degree and reports observed clustering, null mean/standard deviation, z-score, and an empirical p-value; the Researchers page displays it and states that path length and small-world sigma cover only the largest component while density and centralities cover the whole graph. Still open: assortativity, robustness checks, and periodized new-versus-repeated ties (the existing recurrent-edge count is static).

## 6. Medium term — dashboard consolidation

### `WP-20` — Analytical inventory and redundancy removal

- **Status:** Complete on 2026-09-21; the inventory was taken and every unreachable analytical controller was removed.
- **Objective:** Make every visualization answer one unique question.
- **Justification:** The repository retains inactive helpers and analytical functions for retired pages, while Overview contains deep-dive diagnostics.
- **Dependencies:** `WP-04`, validity work packages relevant to each method.
- **Deliverable:** Machine-readable or documented chart inventory with owner/question/population; removal of dead helpers/tests; relocation or deletion of duplicates; Overview reduced to macro state.
- **Completion:** A review finds one canonical owner per question; no unreachable analytical controller remains; removed methods are not claimed in docs/tests.
- **Evidence:** Twenty public functions had no page controller calling them, and seventeen still had passing tests, so the suite stayed green while the features were unreachable. Five implemented methods that `METHODOLOGY.md` already declares unavailable were deleted outright (`price_index_analysis`, `sleeping_beauties_detection`, `disruption_index_estimation`, `citation_longevity_and_decay`, `open_access_impact_analysis`) -- the last of these fabricated a 20% open-access share from a salted `hash()` when real access data was missing, which is exactly the kind of manufactured evidence `ADR-05` forbids. Eleven further exploratory functions with no owner page in the section 7 matrix, plus `fit_quantile_forecast`, were removed with their tests, and `tests/test_strategic_analytics.py` disappeared entirely. Five orphans were wired into their owner pages under `WP-14` through `WP-19` instead of being deleted. The `src/lake_literature` compatibility shim was kept because external importers cannot be ruled out from inside the repository.

### `WP-21` — Integrate validated analyses into existing pages

- **Status:** Partially delivered; the Impact, Screening, Researchers, and Trends increments landed on 2026-09-21. Retrieval benchmark, taxonomy validation, identity audit, and temporal tie dynamics remain blocked on their evidence packages.
- **Objective:** Add knowledge without adding pages or decorative charts.
- **Justification:** New methods should extend established workflows and preserve navigation stability.
- **Dependencies:** `WP-10` through `WP-19`, `WP-20`.
- **Deliverable:**
  - **Qualidade e RAG:** missingness pattern and PDF-selection-bias diagnostics; retrieval benchmark.
  - **Triagem e descoberta:** reviewer agreement, threshold validation, cluster/projection stability.
  - **Impacto e citações:** tail and count-model diagnostics.
  - **Pesquisadores e colaboração:** identity audit and temporal tie dynamics.
  - **Evidências de engenharia:** taxonomy validation coverage.
  - **Tendências e frentes:** baseline skill and interval coverage.
  - **Pipeline e proveniência:** active version, freshness, quality gates, and removal reconciliation.
- **Completion:** Each addition references one `RQ-*`, exposes population/coverage/limitations, and has no equivalent view elsewhere.

### `WP-22` — Streamlit performance, behavior, and accessibility

- **Status:** Dynamic heavy tabs now cover Semantics, Quality, Forecasting, Pipeline, Topics, and Highlights; headless smoke coverage and binary-first search are delivered. Remaining mixed-language copy, Researchers sub-tabs, application-wide AppTests, and the under-five-second suite target remain open.
- **Objective:** Keep the 10-page app responsive and testable as analytical depth grows.
- **Justification:** Some pages still compute hidden tab content, and first-party AppTest coverage does not yet span navigation, filters, and all degraded states.
- **Dependencies:** `WP-20`, `WP-21`.
- **Deliverable:** Dynamic heavy tabs, bounded caches, stable loading slots, forms for expensive searches, native responsive layout where practical, stable widget keys, English sentence-case/accessibility review, and `st.testing.v1.AppTest` smoke tests.
- **Completion:** Navigation/filter/degraded-state tests pass without MySQL; heavy hidden branches do not execute; no deprecated `use_container_width`; measured rerun targets are met on the reference corpus.

## 7. Long term — conditional evidence expansion

### `WP-23` — Reference-year and citation-history ingestion

- **Status:** OpenAlex work and annual-count persistence implemented; credentialed coverage validation remains open.

- **Objective:** Enable valid literature-age and delayed-recognition analyses.
- **Justification:** Price's index, citation longevity, and Sleeping Beauty metrics cannot be reconstructed from cumulative counts.
- **Dependencies:** `WP-01`, `WP-07`.
- **Deliverable:** Governed cited-reference publication years and annual citation trajectories with coverage/provenance.
- **Completion:** Coverage and validation gates pass; only then may Price, longevity, or Sleeping Beauty panels enter `WP-21` review.

### `WP-24` — Citation graph and verified access data

- **Status:** Outgoing citation-edge and access-observation persistence implemented; incoming-edge collection, coverage, and confounding audits remain open.

- **Objective:** Enable disruption and access-association research with the required observables.
- **Justification:** CD disruption requires forward/backward citation relations; OACA requires verified access status and confounder control.
- **Dependencies:** `WP-01`, `WP-07`.
- **Deliverable:** Versioned citation graph; verified access observations; documented sampling/coverage and model protocol.
- **Completion:** Graph integrity and access-validation audits pass; CD/OACA remain unavailable if coverage or confounding control is inadequate.

### `WP-25` — Scale-triggered storage and compute evolution

- **Status:** Complete for the current corpus; binary-first loading and vectorized local search meet the recorded memory/latency thresholds, so no infrastructure migration is approved.

- **Objective:** Adopt external vector/search or distributed compute only when the current design misses SLOs.
- **Justification:** Current corpus size does not justify additional infrastructure.
- **Dependencies:** Observability from `WP-03` and benchmark from `WP-13`.
- **Deliverable:** Benchmark report and ADR comparing in-process linear/Faiss search, MySQL storage, and candidate managed systems on latency, recall, cost, recovery, and operations.
- **Completion:** Migration occurs only if an agreed corpus-size/latency/memory threshold is exceeded and the selected alternative demonstrates a material benefit.

## 8. Cross-cutting completion gates

Every work package that changes data or analytical behavior must include:

- additive migration and rollback/recovery procedure;
- SQLite unit coverage plus MySQL integration coverage where database semantics matter;
- fixtures for missing, degenerate, and failure states;
- documentation updates across PRD/SDD/ROADMAP/METHODOLOGY as applicable;
- dashboard coverage and English UI copy when user-visible;
- `uv run pytest`, `uv run ruff check`, and `uv run ruff format --check` passing;
- an evidence note containing measured baseline, result, residual risk, and follow-up decision.

## 9. Requirements traceability

| Requirements | Architecture decisions | Delivery packages |
|---|---|---|
| `FR-01`, `FR-02`, `NFR-01`, `NFR-02` | `ADR-03` | `WP-01`, `WP-02`, `WP-03` |
| `FR-03`, `FR-04`, `FR-08`, `NFR-04` | `ADR-02`, `ADR-05` | `WP-04`, `WP-05` |
| `FR-05`, `FR-06`, `NFR-03` | `ADR-03`, `ADR-04` | `WP-03`, `WP-06` |
| `FR-07`, `NFR-01`, `NFR-04` | `ADR-03` | `WP-01`, `WP-07`, `WP-08` |
| `FR-09` | `ADR-05` | `WP-09`, `WP-10` |
| `FR-10`, `NFR-05` | `ADR-04` | `WP-13`, `WP-25` |
| `FR-11`, `NFR-06` | `ADR-01`, `ADR-06` | `WP-08`, `WP-22` |
| `FR-12`, `NFR-07`, `NFR-08` | `ADR-02`, `ADR-06` | `WP-03`, `WP-20`, `WP-21`, `WP-22` |
| `RQ-04` through `RQ-07` | `ADR-02`, `ADR-04`, `ADR-05` | `WP-11` through `WP-21` |
| `RQ-08` | `ADR-02`, `ADR-04` | `WP-06`, `WP-13`, `WP-21`, `WP-25` |

## 10. Deferred ideas

The following are intentionally not scheduled until a decision need and evidence contract exist:

- New standalone dashboard pages.
- Generative summaries presented as research evidence.
- Causal ranking of venues, authors, or methods.
- Deep-learning forecasts on short annual series.
- A remote vector database solely for architectural novelty.
- Automatic merging or screening without human-reviewed validation.

Deferral prevents complexity from growing faster than the project's capacity to validate and maintain it.
