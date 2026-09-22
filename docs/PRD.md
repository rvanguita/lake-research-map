# Product and Research Requirements Document — lake-research-map

**Status:** Active specification

**Evidence cutoff:** 2026-09-21

**Implementation baseline:** the active 2026-09-21 Gold version `2974743a…` contains 3,115 articles and 7,552 chunks, all carrying complete embedding metadata; the application has 10 registered dashboard pages and 343 passing automated tests (plus two opt-in MySQL tests skipped by the default run) as measured on 2026-09-22.

**Related documents:** [SDD.md](SDD.md), [ROADMAP.md](ROADMAP.md), [METHODOLOGY.md](METHODOLOGY.md)

This document defines the product and research problem, the decisions the platform must support, and the evidence required to claim that it succeeds. `SDD.md` defines how those requirements are implemented; `ROADMAP.md` sequences changes that are not yet implemented. Counts above are a dated corpus snapshot, not architectural constants.

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
- A retrieval client or LLM agent using Gold chunks to locate evidence while preserving DOI provenance.

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
- CLI and Airflow execution with a read-only analytical dashboard.

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
| `FR-05` | Reconcile chunks without retaining vectors for changed text. | Unchanged chunks retain vectors; changed/removed chunks invalidate cleanly. |
| `FR-06` | Publish semantic outputs only for a complete compatible embedding set. | Coverage and model/dimension checks pass before atomic replacement. |
| `FR-07` | Record a correlated pipeline run and dataset version. | All stage events share a parent run and input/output version identifiers. |
| `FR-08` | Serve Gold as the canonical analytical population. | Dashboard counts agree with curated Gold and approved merges. |
| `FR-09` | Provide auditable screening calibration. | Labels, reviewer agreement, threshold version, metrics, and holdout results persist. |
| `FR-10` | Provide inspectable lexical, dense, and hybrid retrieval. | Results expose parent DOI and component ranks/scores. |
| `FR-11` | Keep dashboard data access read-only. | Mutations occur only through CLI/Airflow workflows. |
| `FR-12` | Degrade safely when a table or optional analysis is unavailable. | Pages show a scoped explanation rather than crashing or inventing results. |

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
| `RISK-08` | Dynamic tabs are not used consistently. | Hidden expensive analyses execute and slow reruns. | Gate heavy tab bodies and add performance acceptance tests. |

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
| Use local BGE embeddings and binary float32 storage. | Appropriate for corpus scale and offline operation; performance claims require benchmarks. | Implemented |
| Keep dashboard pages read-only. | Separates analytical presentation from controlled pipeline mutations. | Implemented |
| Treat contract-valid Gold as the canonical analytical population. | Gold is where approved merges and curated chunks exist; invalid or absent Gold falls back visibly to Silver/Bronze. | Versioned publication implemented; selector-to-pointer binding remains in `WP-04` |
| Keep uploaded screening calibration non-persistent and non-operative. | The current phase can validate the method without allowing a dashboard session to mutate corpus decisions. | Implemented; governed persistence pending |
| Version corpus, enrichment, models, anchors, and labels. | Necessary for reproducible statistical and semantic results. | Approved; implementation pending |
| Add new methods only when they answer a distinct decision question and meet data prerequisites. | Prevents dashboard breadth from exceeding evidential validity. | Active governance rule |
