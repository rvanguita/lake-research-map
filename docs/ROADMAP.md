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

The following capabilities are implemented and covered by the current 382 passing tests plus two opt-in MySQL tests skipped in the default run (229 before this cycle, followed by regression, application-wide page, evidence-workflow, retrieval, provenance, semantic-persistence, MySQL migration, and year-bound coverage):

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
- **Evidence:** Persisted Raw/Bronze/Silver/Gold/Embed/Semantic contracts block publication in automated tests; candidate Gold tables are isolated until atomic materialization; the existing pipeline page exposes versions, execution lineage, file changes, and gate outcomes. MySQL 8.4.11 acceptance on 2026-09-22 verified JSON/NULL/LONGBLOB round trips, transaction rollback, advisory locks, and the idempotent `lit_config` uniqueness migration. End-to-end injected-failure publication remains a separate WP-08 gate.

### `WP-04` — Gold as the canonical analytical population

- **Status:** Complete on 2026-09-21; analytical readers bind articles, chunks, semantics, duplicate candidates, search, and projections to the active immutable version.
- **Objective:** Ensure approved duplicate curation is reflected throughout the dashboard.
- **Justification:** Gold-first selection now protects approved merges, but readiness is not yet tied to an immutable published dataset version or persisted quality-gate result.
- **Dependencies:** `WP-03`.
- **Deliverable:** Gold readiness contract; propagation/join of required quality fields; explicit degraded pre-Gold mode; updated loaders and page coverage captions.
- **Completion:** Dashboard article counts and DOI sets equal the active Gold version; merge/undo changes appear after the documented rebuild sequence; degraded mode is visibly labeled.

### `WP-05` — Identity, PDF matching, and rejection validation

- **Status:** `blocked-awaiting-evidence`; the persistent review workflow is available, but no reviewed identity/PDF/rejection sample has been supplied. The queue itself was the blocker until 2026-09-22: `export_assignments` wrote only `assignment_id` and an opaque `subject_id`, so no reviewer could answer the protocol's question from the file they were given -- `review/rene-pdf.csv` held its header and nothing else. Exports now carry the resolved evidence per workflow; see `docs/evidence/2026-09-22-review-queue-context.md`.

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

- **Status:** Complete on 2026-09-23. The OpenAlex refresh has observed **3,099 of 3,115 corpus DOIs (99.5%)**; the remaining 16 are `not_found` -- OpenAlex holds no record for them -- and the 75 that were throttled in the first window succeeded on retry. The pass is batched (50 DOIs per request, so the corpus costs 63 requests rather than 3,115), resumable, publisher-proportional in order (`ADR-07`), and stops itself on throttling. Observations are append-only with `observed_at`, so a later refresh adds a snapshot rather than rewriting one, and the dashboard reads its citation-age year from the latest persisted observation. See `docs/evidence/2026-09-23-openalex-closure.md`.

- **Objective:** Make citation/reference data reproducible and refreshable.
- **Justification:** The current cache stores only latest counts and cannot support an as-of analysis.
- **Evidence:** `_run_enrichment_command` required both `OPENALEX_API_KEY` and `OPENALEX_EMAIL`. OpenAlex issues no API key for the public corpus -- `ingest/openalex.py`'s own docstring says so -- so every refresh was unreachable by construction, whatever the operator put in `.env`, and it failed pointing at a missing credential rather than at an impossible requirement. The same shape as the database roles this cycle removed under `WP-08`. The gate now requires the polite-pool contact address alone, states the reason in the error, and accepts a key only if one exists. The `User-Agent` also carried a fixed `mailto:researcher@example.com`: an address nobody reads is worse than none, because it is what OpenAlex would contact about a misbehaving crawl, so the contact is now read from the environment and omitted when absent.
- **Dependencies:** `WP-01`.
- **Deliverable:** Provider observations with `observed_at`, status, identifiers, counts, response hash, retry metadata, and deterministic selection of the analysis snapshot.
- **Completion:** A refresh appends an observation without rewriting history; the same dataset/as-of selection reproduces the same citation inputs.

### `WP-08` — Concurrency, recovery, and operational security

- **Status:** Complete on 2026-09-22. Advisory locking, persistent logs, signal handling, resumable embeddings, heartbeat recovery, Airflow retry/timeouts, local-only ports, and resource bounds delivered. Per-table role provisioning was removed on 2026-09-22 -- see Evidence -- so the package no longer waits on passwords that were never going to be set. Abandoned-run recovery became automatic on 2026-09-22; the retry/timeout policy was already delivered in `STAGE_POLICY` and the open status was stale.
- **Objective:** Make mutation predictable under failure and prevent unsafe overlapping runs.
- **Justification:** Writer serialization and resumable embeddings protect the common local failure path; a killed run stayed recorded as `running` until a human remembered to type `maintenance recover-stale`, and the dashboard's status reads those rows.
- **Dependencies:** `WP-01`.
- **Deliverable:** Project advisory lock, stage retry/timeout matrix, reliable terminal run status, deployment-mode security guidance, and audited trigger identity. Least-privilege database roles were dropped from the deliverable, not deferred.
- **Evidence:** `db/roles.py` provisioned `lake_pipeline` and `lake_dashboard` with per-table `lit_*` grants, and `config.py` selected between them with `LAKE_RESEARCH_MAP_DB_ROLE`. The whole mechanism was removed on 2026-09-22 as unjustified for a single-developer, single-machine deployment. It was never mandatory -- the role variables always fell back to `MYSQL_USER`/`MYSQL_PASSWORD` -- but the fallback was per-variable rather than per-pair, so setting only `MYSQL_PIPELINE_PASSWORD` connected as `root` with the pipeline credential and failed as a bare `Access denied for user 'root'`, which points at the server instead of at two lines of `.env`. A guard was added first and then removed with the feature it protected: a mechanism that needs a guard to be used safely, and that this deployment does not need at all, is complexity without a payer. The cost is recorded rather than glossed: the dashboard container ran with read-only grants through `LAKE_RESEARCH_MAP_DB_ROLE: dashboard` and now connects with write-capable credentials. It stays read-only by code -- no page writes, and pipeline execution goes through Airflow's REST API -- but that is an invariant in code, no longer a privilege boundary. Restoring the boundary is a matter of re-adding the module if this ever leaves one machine.
- **Completion:** Concurrent-run and injected-failure integration tests prove one active writer, safe retry behavior, visible telemetry failure, and unchanged prior published state.

## 4. Short to medium term — human ground truth

### `WP-09` — Persistent dual-review screening workflow

- **Status:** Persistent protocols, dual assignments, append-only label revisions, CSV export/import, and adjudication delivered. Screening inclusion criteria were approved and persisted as protocol `v2` on 2026-09-22, with 60 assignments over the 30 no-DOI subjects. Independent reviewer labels remain open. They were also, until 2026-09-22, impossible to produce: the exported queue carried no title, abstract or venue for a `reject::` subject. That is fixed -- see `docs/evidence/2026-09-22-review-queue-context.md` -- so what remains is genuinely a human decision rather than missing code.
- **Objective:** Calibrate screening against reproducible human judgments.
- **Justification:** The dashboard now measures agreement from uploaded decisions, but labels, assignments, and adjudication are not durable or dataset-versioned.
- **Dependencies:** `WP-01`, `WP-04`, `WP-06`.
- **Deliverable:** Label/adjudication schema, protocol version, deterministic assignments, independent reviewer exports/imports, disagreement queue, and Cohen's kappa/agreement report.
- **Evidence:** The agreement report existed only behind a browser upload, so the labels the CLI persisted had no reader at all. `reviews status` now reports progress, pairwise kappa, the label-set digest, and the disagreement queue from `lit_review_labels`/`lit_review_adjudications`. `assignment_progress` breaks progress out per reviewer, because a round where one reviewer finished and the other never started produces plenty of labels and no meaningful agreement, and one completion percentage hides exactly that case. A screening subject is also not always a DOI -- `evidence rejections` emits `reject::<source>::<id>` for records dropped before they ever had a margin -- so those are now reported as `unknown_doi` instead of vanishing in the join, since a round whose labels mostly disappear there otherwise looks identical to a round nobody labelled.
- **Completion:** Two reviewers can label the same version independently; adjudication preserves raw labels; agreement and class prevalence are reproducible.

### `WP-10` — Screening threshold validation

- **Status:** Calibration, durable hash-bound approval schema/CLI, and the join between them delivered; the approval decision and later-batch validation await independent labels.
- **Objective:** Select a workload-aware threshold with uncertainty and out-of-sample evidence.
- **Justification:** Zero contrastive margin is meaningful geometrically but has no guaranteed recall.
- **Dependencies:** `WP-09`.
- **Deliverable:** PR analysis, sensitivity/specificity/precision/F2, bootstrap confidence intervals, calibration/validation split, threshold record, and manual-review workload estimate. ROC is intentionally omitted because it duplicates the screening trade-off under class imbalance.
- **Evidence:** `reviews import` wrote decisions to `lit_review_labels` while `calibrate_screening_threshold` accepted only a CSV uploaded in the browser, so a durable label could never reach the threshold it exists to validate and the panel's evidence vanished with the tab. `collect_labels` closes that join, returning the newest revision per assignment -- the table is append-only, so an earlier revision is history rather than a second opinion -- and surfacing an adjudication as a row under the reserved reviewer `adjudicated`, the marker `resolve_review_consensus` already honours. `approve_model` had always demanded a `label_set_sha256` that **nothing computed**, leaving it to be typed by hand, which binds an approval to nothing; `label_set_digest` computes it over the raw reviewer rows rather than the resolved consensus, because two different label sets can resolve identically and an approval that cannot tell them apart has not recorded its evidence. `screening_model_digest` identifies the rule -- embedding model, both anchors, cut, sensitivity target -- since changing any one stops the approved sensitivity describing what the dashboard would do. `reviews calibrate` runs the chain off the database, prints both digests and the exact `reviews approve` call, and applies nothing: a threshold that excludes work does not get to select itself.
- **Completion:** The selected threshold meets the approved sensitivity target on validation data or automatic exclusion remains disabled with the failure documented.

### `WP-11` — Author identity audit and overrides

- **Status:** `blocked-awaiting-evidence`; reviewer persistence is available, while resolved author overrides and an audited error rate require human decisions. The queue itself was the blocker until 2026-09-22: `export_assignments` wrote only `assignment_id` and an opaque `subject_id`, so no reviewer could answer the protocol's question from the file they were given -- `review/rene-pdf.csv` held its header and nothing else. Exports now carry the resolved evidence per workflow; see `docs/evidence/2026-09-22-review-queue-context.md`.

- **Objective:** Bound errors from heuristic author canonicalization.
- **Justification:** Homonyms can merge and spelling variants can split, invalidating rankings and network structure.
- **Dependencies:** `WP-04`.
- **Deliverable:** Ambiguity candidate generation, reviewed audit sample, reversible identity overrides, and error-rate disclosure.
- **Completion:** Person-level panels use the resolved identity version and show unresolved ambiguity coverage; corpus metrics can be recomputed after undo.

### `WP-12` — Engineering-taxonomy validation

- **Status:** Complete on 2026-09-22 under the second branch of its own completion rule -- every displayed taxonomy reports measured coverage and is labelled exploratory. An explicitly AI-assisted draft now covers all 120 stratified subjects, while the 240 independent human assignments remain outstanding; per-class precision is therefore still unapproved human evidence.

- **Objective:** Treat regex extraction as a measured multi-label classifier.
- **Justification:** Frequency charts currently lack precision/recall evidence and an explicit unknown class.
- **Dependencies:** `WP-04`.
- **Deliverable:** Stratified labeled articles, label guide, per-class precision/recall/F1, ambiguous/unclassified reporting, and refined non-overlapping rules where justified.
- **Completion:** Each displayed taxonomy reports evaluated coverage and meets a declared minimum precision or is labeled exploratory.
- **Evidence:** The completion rule has two branches and this closes on the second, deliberately: *evaluated coverage* is measurable today, *declared minimum precision* is not, and labelling the taxonomies exploratory is the honest reading rather than a way around the bar. All seven regex taxonomies on the Engineering-evidence page -- optimization methods, objective functions, uncertainty paradigms, planning horizons, computational solvers, mathematical complexity, benchmark feeders -- were previously inline pattern dicts; they are hoisted to module constants and enumerated in `analytics.py::TAXONOMY_REGISTRY`, which is what makes "each displayed taxonomy" checkable instead of a promise, and a test asserts the registry matches the page. `taxonomy_coverage` reports the classified share, the unclassified count and the multi-label count beside every chart. The measured numbers are the finding: **objective functions classify 78.3% of the corpus but assign 1,457 of those articles to more than one class, while the other six classify between 16.1% and 31.8%** -- planning horizons 17.2%, computational solvers 16.1%, benchmark feeders 19.9%, uncertainty paradigms 20.3%, mathematical complexity 28.3%, optimization methods 31.8%. A frequency chart over the matched subset answers "among the ones I recognised, which is most common?" while looking like it answers "what does this corpus do?", so each panel now says which question it is answering and over what share. `taxonomy_precision_from_labels` scores a human-reviewed sample per class, excluding `ambiguous` from precision and recall and counting it separately -- a reviewer who could not decide is evidence about the class boundary, not a negative -- and the panel swaps the exploratory notice for measured precision only when human labels exist. Reviewer `codex-assisted` completed 120 immutable labels under taxonomy protocol `v1`: 108 `present`, 11 `absent`, and 1 `ambiguous`; label-set SHA-256 `37f8c9e019ff40f148325021054f66c83f4b1a10c2ff3d27ba1a06b1e975e0a2`. Three `unclassified` subjects were flagged as likely misses (MILP, Machine Learning & AI, and Stochastic Programming). This is a documented draft rather than human ground truth: reviewers `rene` and `revisor2` retain 120 assignments each, the dashboard excludes `-assisted` reviewers from validation metrics, and no taxonomy subject is resolved until independent agreement or adjudication exists.

### `WP-13` — Retrieval evaluation corpus

- **Status:** `blocked-awaiting-evidence`; the common Recall@k/MRR/nDCG/latency harness and inspectable dense/BM25/RRF ranks are implemented. One explicitly AI-assisted review labeled all 96 pooled subjects (92 relevant, 3 not relevant, 1 uncertain), but the two independent human reviews remain outstanding, so metric values and default-mode selection are not approved. Their queues now carry the query text and the article beside each judgement, which the id-only export did not.

- **Objective:** Choose dense, lexical, or hybrid retrieval from measured relevance and latency.
- **Justification:** RRF is implemented, but no labeled query set supports a quality claim.
- **Dependencies:** `WP-04`, `WP-06`.
- **Deliverable:** Versioned technical queries, pooled judgments, abstract/full-text strata, Recall@k, MRR, nDCG, latency, and failure taxonomy.
- **Completion:** All retrieval modes run on identical judgments; the default is selected by an explicit metric/latency rule and reproduced in tests.
- **Evidence:** `dashboard/retrieval_eval.py` evaluates identical judged rankings and reports Recall@k, reciprocal rank, nDCG@k, and deterministic latency summaries for each mode; hand-computed fixtures protect the metric definitions. Search results retain dense and BM25 component ranks/scores beside the RRF rank, and the Quality page can execute each mode independently. Reviewer `codex-assisted` completed 96 immutable labels under retrieval protocol `v1`; label-set SHA-256 `4aef1027dc39602b6584f8befc457f157298e27f1707d689e2163871fc308fd4`. These labels are a documented draft, not human ground truth. Reviewers `rene` and `revisor2` each retain 96 independent assignments, and no retrieval subject is resolved until independent agreement or adjudication exists. No winner is claimed.

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

- **Status:** Complete on 2026-09-22; every ranked trend surface now carries uncertainty, the searched breakpoint is scored against a permutation null, and serial dependence is reported as a sensitivity.

- **Objective:** Control false discoveries across keyword/topic trend panels.
- **Justification:** Many Mann-Kendall and breakpoint tests are interpreted independently.
- **Dependencies:** `WP-04`.
- **Deliverable:** Test-family definitions, minimum prevalence, Benjamini-Hochberg adjusted values, effect-size thresholds, serial-dependence sensitivity, and exploratory breakpoint correction/bootstrap.
- **Completion:** Every ranked trend table shows raw effect, uncertainty, adjusted significance, sample span, and zero-filled years.
- **Evidence:** Benjamini-Hochberg previously adjusted only the 7 most-positive and 7 most-negative slopes -- a family already selected for being extreme, which inflates significance rather than controlling it. Mann-Kendall now runs over every keyword meeting the minimum-prevalence rule, the adjustment is applied to that full family, and only then is the table narrowed to the charted terms; the caption states the family size. The three remaining gaps are closed. The OLS slope chart is now fitted by `analytics.py::linear_slope_with_ci`, which returns the standard error and 95% interval, and the bars carry those as error bars with a caption saying how many of the charted terms have an interval that excludes zero -- on short annual series most do not. `detect_structural_breaks` searched every split and then read the maximum F against the F table, which is the wrong null: measured over 200 pure-noise series, that reported a regime change **32.5% of the time** at alpha 0.05. The reported p-value is now empirical, from permuting the series under a no-break null (**8.0%** on the same series), with `p_value_naive` kept beside it so the gap stays visible and `p_value_resolution` stating the 1/(draws+1) floor an empirical p-value cannot go below. Serial dependence is handled by `_hamed_rao_variance_factor`: a 40-point random walk with no true trend gets a variance inflation of 4.7, white noise and a genuine trend both get exactly 1.0. The correction is opt-in (`serial_correction=True`) because `p_value` is read by existing callers and silently redefining it would rewrite published trend tables; the Trends table shows both adjusted columns and says how many keywords survive the correction.

### `WP-17` — Semantic stability and projection diagnostics

- **Status:** Complete on 2026-09-22; the k sweep, two further neighbourhood measures, and run-level persistence all ship beside the map.

- **Objective:** Separate robust high-dimensional structure from unstable 2D presentation.
- **Justification:** Silhouette-selected KMeans and t-SNE/UMAP views can change with samples and parameters.
- **Dependencies:** `WP-06`.
- **Deliverable:** Bootstrap/subsample ARI, cluster-count sensitivity, projection trustworthiness, neighborhood preservation, seed stability, and drift computed in embedding space before visualization.
- **Completion:** Theme/novelty panels show stability and coverage; unstable labels remain numbered/exploratory rather than receiving fixed ontological names.
- **Evidence:** `analytics.py::semantic_stability_diagnostics` computes subsample/seed bootstrap ARI and projection trustworthiness, and the Screening page's stability panel reports both next to the projection along with the population used and an explicit warning that theme numbering carries no ontological claim. The three named gaps are closed. `transform/semantics.py::theme_sweep` re-runs the same search `discover_themes` performs and keeps every candidate's silhouette, smallest-cluster share and rejection flag, so the page can show that silhouette is nearly flat across `MIN_THEMES..MAX_THEMES` and that the near-tie rule, not a maximum, is what selects `k`. Trustworthiness only penalises neighbours a projection invents, so a layout that tears one real cluster in two scores well on it; `projection_continuity` (the same measure with the spaces swapped) and `knn_overlap` are reported beside it, and on a scrambled control they fall to 0.55 and 0.33 against 0.95 and 0.80 for a faithful one. Persistence is an additive nullable `stability` JSON column on `lit_semantic_runs` (`db/bootstrap.py::_ADDITIVE_COLUMNS`), attached to the run that produced the layout rather than recomputed, so a past map keeps its own caveats. Embedding-space drift remains deliberately unimplemented: with a single embedding revision in the corpus there is no second point to measure drift against, and a drift number computed from one revision would be decorative.

### `WP-18` — Forecast and Bass validation

- **Status:** Complete on 2026-09-22; each forecast year is backtested at its own horizon, MASE ships beside baseline skill, and the Bass peak carries an interval.

- **Objective:** Quantify whether projections improve on persistence and whether intervals cover future observations.
- **Justification:** Short annual series can make model selection and conformal bands unstable.
- **Dependencies:** `WP-07`, `WP-16`.
- **Deliverable:** Expanding-window backtests by horizon, MAE/MASE or baseline skill, empirical interval coverage/width, dynamic complete-year cutoff, and Bass parameter bootstrap/sensitivity.
- **Completion:** Forecasts that do not outperform persistence are labeled accordingly; displayed intervals meet the declared backtest coverage tolerance or carry a warning.
- **Evidence:** The cutoff was already dynamic (`LAKE_RESEARCH_MAP_COMPLETE_YEAR`, defaulting to the previous calendar year); the remaining hard-coded years in the forecast path -- the keyword baseline year, the ranking axis, and the CV label -- now derive from it. Skill against persistence is displayed and says "No better than naive" when it is not positive. Interval coverage was scored on the same errors that set the conformal radius, so it returned ~0.9 by construction and could never fail; it is now measured on held-out folds and returns nothing when the series is too short to spare any. The three remaining gaps are closed. `_rolling_origin_cv` and `_holdout_interval_coverage` now take a `horizon`, and the training window ends that many years before the target, so a two-year claim never borrows one-year information; the Forecasting page shows a per-horizon table and prints "not testable" where the series is too short to hold folds back at that horizon rather than reusing the shorter number. MASE divides the CV error by the mean absolute one-step change, which is what makes a sparse keyword series and the corpus series comparable at all. `fit_bass_diffusion_nls` kept `popt` and threw `pcov` away; it now reports per-parameter standard errors and an 80% interval for the peak year from a parametric bootstrap over the fitted covariance -- the peak is a non-linear function of p and q, so its spread cannot be read off their standard errors directly. On a noise-free synthetic diffusion the interval collapses to the point estimate and widens monotonically as noise is added.

### `WP-19` — Network null models and temporal collaboration

- **Status:** Complete on 2026-09-22; assortativity, robustness and periodized tie dynamics all ship. Every claim stays bounded by heuristic author identity, which is `WP-11`'s business, not this package's.

- **Objective:** Distinguish structural collaboration evidence from corpus-size artifacts.
- **Justification:** Static centralities and a single random comparison do not describe network uncertainty or evolution.
- **Dependencies:** `WP-11`.
- **Deliverable:** Component-aware metrics, degree-preserving null models, assortativity/robustness checks, and periodized new-versus-repeated collaboration ties.
- **Completion:** Network claims include identity coverage, component scope, null distribution, and sensitivity to the selected author subset.
- **Evidence:** `analytics.py::network_null_model_diagnostics` rewires the graph preserving every author's degree and reports observed clustering, null mean/standard deviation, z-score, and an empirical p-value; the Researchers page displays it and states that path length and small-world sigma cover only the largest component while density and centralities cover the whole graph. The three remaining gaps are closed. Degree assortativity is scored against the same degree-preserving ensemble that was already being built, since the degree sequence alone forces part of it and a bare coefficient would be unreadable. `_robustness_under_removal` reports the giant-component share after removing the top-decile hubs and after removing the same count at random, averaged over 20 draws: on a scale-free control the gap is 0.15 (0.75 vs 0.90) against 0.02 for a small-world control, which is the signature that distinguishes a network held together by a few authors from one with distributed structure. `periodized_collaboration_ties` splits author pairs into equal-year periods and counts a tie as new in the period holding its first-ever collaboration and returning thereafter, which is what separates a field still recruiting collaborators from one consolidating into fixed teams -- a distinction the static recurrent-edge count cannot make, because both produce the same number of repeat pairs.

## 6. Medium term — dashboard consolidation

### `WP-20` — Analytical inventory and redundancy removal

- **Status:** Complete on 2026-09-22; the inventory was refreshed and obsolete ingestion/analytical paths were removed without deleting intentional scale-up hooks. A second audit the same day found one survivor the inventory had missed -- `theme.apply_dashboard_theme()`, 205 lines of injected CSS with no caller anywhere while `.streamlit/config.toml` supplied the real palette. Removing it left `theme.py` with no Streamlit import at all, and the project skill that still documented it as the styling entry point was corrected with it.
- **Objective:** Make every visualization answer one unique question.
- **Justification:** The repository retains inactive helpers and analytical functions for retired pages, while Overview contains deep-dive diagnostics.
- **Dependencies:** `WP-04`, validity work packages relevant to each method.
- **Deliverable:** Machine-readable or documented chart inventory with owner/question/population; removal of dead helpers/tests; relocation or deletion of duplicates; Overview reduced to macro state.
- **Completion:** A review finds one canonical owner per question; no unreachable analytical controller remains; removed methods are not claimed in docs/tests.
- **Evidence:** Twenty public functions had no page controller calling them, and seventeen still had passing tests, so the suite stayed green while the features were unreachable. Five implemented methods that `METHODOLOGY.md` already declares unavailable were deleted outright (`price_index_analysis`, `sleeping_beauties_detection`, `disruption_index_estimation`, `citation_longevity_and_decay`, `open_access_impact_analysis`) -- the last of these fabricated a 20% open-access share from a salted `hash()` when real access data was missing, which is exactly the kind of manufactured evidence `ADR-05` forbids. Eleven further exploratory functions with no owner page in the section 7 matrix, plus `fit_quantile_forecast`, were removed with their tests, and `tests/test_strategic_analytics.py` disappeared entirely. Five orphans were wired into their owner pages under `WP-14` through `WP-19` instead of being deleted. The `src/lake_literature` compatibility shim was kept because external importers cannot be ruled out from inside the repository.
- **Reconciliation:** The second inventory removed the obsolete dashboard-side Raw upload module, the file-cache OpenAlex enrichment path, an unused layer selector, a redundant source serializer, and an unused thematic-centroid function. The optional vector-index builder remains intentionally reachable as the measured WP-25 scale-up path; it is not the current serving path.

### `WP-21` — Integrate validated analyses into existing pages

- **Status:** Partially delivered; the Impact, Screening, Researchers, and Trends increments landed on 2026-09-21, and temporal tie dynamics shipped with `WP-19` on 2026-09-22 -- it never depended on an evidence package. Two further increments landed on 2026-09-22 that had been recorded as blocked but were not: the **Pipeline and provenance** bullet (active version and freshness) was never blocked at all, and the **retrieval benchmark** needed a surface rather than labels. Taxonomy validation and identity audit remain genuinely blocked on theirs.
- **Evidence:** `article_population_status()` had returned `dataset_version_id` since `WP-04` and no page ever read it -- only `is_canonical` was consumed -- so nine of ten pages reported figures that could not be tied to a snapshot, against `PRD.md` §10's first success criterion. `loaders.render_population_provenance` now states the version, its publication time, and the filtered-versus-total population, rendered once inside `require_articles` so a new page cannot forget it, and in `loaders` rather than `components` because that module imports this one. The retrieval harness (`dashboard/retrieval_eval.py`) and the pooled sampler were likewise reachable only from `tests/`: the project could compute Recall@k, MRR and nDCG and had nowhere to show them. The Quality and RAG page now scores all three modes over the versioned query set, and refuses the conclusion rather than the computation -- with only the AI-assisted pass stored, it prints the table under an explicit **not approved evidence** warning naming `WP-13`'s requirement for two independent reviewers. A sweep for the unpunctuated half of the dropped-clause defect also ran over every page and repaired six captions, among them one claiming articles "are typical of Researches" and one ending mid-phrase on `the card '`.
- **Objective:** Add knowledge without adding pages or decorative charts.
- **Justification:** New methods should extend established workflows and preserve navigation stability.
- **Dependencies:** `WP-10` through `WP-19`, `WP-20`.
- **Deliverable:**
  - **Quality and RAG:** missingness pattern and PDF-selection-bias diagnostics; retrieval benchmark.
  - **Screening and discovery:** reviewer agreement, threshold validation, cluster/projection stability.
  - **Impact and citations:** tail and count-model diagnostics.
  - **Researchers and collaboration:** identity audit and temporal tie dynamics.
  - **Engineering evidence:** taxonomy validation coverage.
  - **Trends and fronts:** baseline skill and interval coverage.
  - **Pipeline and provenance:** active version, freshness, quality gates, and removal reconciliation.
- **Completion:** Each addition references one `RQ-*`, exposes population/coverage/limitations, and has no equivalent view elsewhere.

### `WP-22` — Streamlit performance, behavior, and accessibility

- **Status:** Complete on 2026-09-22; every tab group is lazy, the mixed-language copy is gone, and `AppTest` spans all ten pages. The under-five-second target is withdrawn and replaced by a measured one -- see Evidence. The language claim took two passes: a second audit the same day found a Portuguese fragment and an empty `st.metric` label that the first scan test could not see, because its word list held domain nouns while the survivor was built from function words. The test now covers closed-class words and rejects an empty metric label through the AST, since `st.metric` takes its label positionally and leaves no `st.subheader("")` to find.
- **Objective:** Keep the 10-page app responsive and testable as analytical depth grows.
- **Justification:** Some pages still compute hidden tab content, and first-party AppTest coverage does not yet span navigation, filters, and all degraded states.
- **Dependencies:** `WP-20`, `WP-21`.
- **Deliverable:** Dynamic heavy tabs, bounded caches, stable loading slots, forms for expensive searches, native responsive layout where practical, stable widget keys, English sentence-case/accessibility review, and `st.testing.v1.AppTest` smoke tests.
- **Completion:** Navigation/filter/degraded-state tests pass without MySQL; heavy hidden branches do not execute; no deprecated `use_container_width`; measured rerun targets are met on the reference corpus.
- **Evidence:** Every `st.tabs` group in `dashboard/pages/` now passes `on_change="rerun"` with a stable key and renders only the open tab -- previously five groups on Researchers plus one each on Forecasting and Quality computed all their branches on every rerun, and the Researchers network tab in particular builds a graph and refits a null model. Splitting that page into `_collaboration_section`, `_research_lines_section` and `_bibliometric_laws_section` is what made the laziness expressible; the concentration and trend panels now read the author-year matrix from its own cached loader instead of borrowing it from the table tab, which no longer runs when they are open. Forty-two user-visible strings were still Portuguese or machine-translated (`"Autor"` on nine axes, `"Resumo"`, `"Busca nos chunks"`, `"Mediana"`, `"N/D"`, `"Taxa Multi-Objetivo"`, a download button reading `"Baixar"` that saved `producao_por_autor_ano.csv`), several of which read as broken English rather than as Portuguese: `"Ascented vs. declining terms"`, `"Incline annual participation"`, `"& License type"`, `"is not used of selection or final adjustment"`, and `"in this cut"` where the code meant a filter. `tests/test_dashboard_pages.py` walks the `PAGES` registry and renders all ten pages under both a populated and an empty corpus; it caught a real coupling rather than merely covering the pages -- `optimization_methods_taxonomy` and two burst functions keyed `groupby(...)["id"].count()` off the surrogate primary key purely to count rows, so they broke on any frame assembled without one, and now use `.size()`.
- **Measured suite time:** the under-five-second target does not survive this package and is withdrawn rather than quietly missed. Measured over three consecutive runs on this machine the full suite took 12.6 s, 33.1 s and 38.9 s for 268 tests -- wall-clock timing here is dominated by machine load, so only the shape is trustworthy, not any single figure. The twenty-one page renders are the stable part at 4.9-5.5 s on their own. An earlier reading of 29.7 s was an artifact worth recording: the page fixtures stubbed only the article loaders, so every page still opened a MySQL connection for its run history and quality gates and waited out the timeout. That was both slow and a false green -- the pages were rendering the degraded path rather than the one under test -- and one of them failed outright under load. Stubbing every loader that reaches the database fixed the failure and the time together. Five seconds stays out of reach because each `AppTest.from_string` boots a script runner, and putting the tests behind a marker to protect the old number was rejected: an excluded test is not coverage. A reproducible timing target needs a quiet machine and should be set against CI, not a developer laptop.

## 7. Long term — conditional evidence expansion

### `WP-23` — Reference-year and citation-history ingestion

- **Status:** Annual trajectories: **gate passes** -- 3,055 of 3,099 observed works (98.6%) have a known trajectory, of which 677 are never-cited works whose empty series is a known zero, not a gap. 584 works published before 2012 are left-censored because the provider's series begins in 2012, so longevity and Sleeping Beauty may use only the 2,515 works whose history is complete from publication. Cited-reference years: **gate fails today at 6.4%** of 89,700 reference pairs; the resolver (`enrichment resolve-references`, 50 works per request) is implemented and tested, and is running across the next two quota windows. Closes when `audit citation-years` reports both verdicts as PASS.

- **Objective:** Enable valid literature-age and delayed-recognition analyses.
- **Justification:** Price's index, citation longevity, and Sleeping Beauty metrics cannot be reconstructed from cumulative counts.
- **Dependencies:** `WP-01`, `WP-07`.
- **Deliverable:** Governed cited-reference publication years and annual citation trajectories with coverage/provenance.
- **Completion:** Coverage and validation gates pass; only then may Price, longevity, or Sleeping Beauty panels enter `WP-21` review.

### `WP-24` — Citation graph and verified access data

- **Status:** Both completion gates **pass**: graph integrity (0 self-loops, 0 dangling edges, and 100% agreement between OpenAlex's `referenced_works` and `cites:` indexes over the 3,314 corpus-internal pairs that can be cross-checked) and access validation (all 50 articles the IEEE CSV licenses CC/OAPA are open in OpenAlex; the two sources name the same CC licence in 95.7% of 46 pairs). Backward coverage is 100% -- 324 works report zero references and are known, not missing. Forward coverage is 875 of 3,099 (28.2%), so the disruption index stays unavailable, which the completion rule explicitly allows; the batched forward crawl (233 requests for the rest) runs in the next quota window, after which the gates are re-run over the whole graph and the package closes. OACA stays unavailable regardless: no confounder-controlled model exists.

- **Objective:** Enable disruption and access-association research with the required observables.
- **Justification:** CD disruption requires forward/backward citation relations; OACA requires verified access status and confounder control.
- **Dependencies:** `WP-01`, `WP-07`.
- **Deliverable:** Versioned citation graph; verified access observations; documented sampling/coverage and model protocol.
- **Completion:** Graph integrity and access-validation audits pass; CD/OACA remain unavailable if coverage or confounding control is inadequate.
- **Evidence:** `ingest/openalex.py::fetch_openalex_citing_works` follows the `cites:` cursor to collect forward edges, and `persist_incoming_edges` stores them in the existing `lit_citation_edges` table. No direction column was added: an incoming edge is already one whose `cited_work_id` is ours. What was missing is provenance, so `discovered_via` (additive, nullable) records whether an edge came from a work's own complete `referenced_works` list or from a paginated `cites:` crawl -- absent the distinction, a crawl that ran out of page budget is indistinguishable from a genuinely uncited work, and the disruption index this package exists to enable would be computed on a forward-citation set silently missing its tail. The crawl is capped at five pages of 200 and reports `truncated` when it stops early. `citation_graph_coverage` reports backward and forward coverage separately and defines the usable population as their **intersection**, never the larger of the two, because the disruption index needs both directions for the same work. All of it is tested against injected fixtures; nothing here has been run against the live API, since `.env` carries no `OPENALEX_EMAIL` and a refresh would issue thousands of requests. As first written none of it had a caller outside `tests/`, so the capability existed only in the suite -- the same invariant the dashboard holds to, that no public function is unreachable from a surface, did not hold here. `enrichment refresh-citations` drives the crawl from `lit_external_works` rather than from DOIs, because `cites:` filters on an OpenAlex work id and a DOI the backward pass never resolved has no id to crawl; truncated works are counted in the run stats. `audit citation-graph` prints the coverage gate and says plainly that CD/disruption staying unavailable is the documented outcome, not a defect to fix in code.

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
