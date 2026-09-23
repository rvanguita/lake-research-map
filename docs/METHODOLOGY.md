# Analytical Methodology

This document defines what the dashboard can infer from the current corpus, how each model is validated, and which formal indicators are intentionally unavailable.

## Evidence boundaries

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

## Citation count model

`citation_determinants_glm` models cumulative citations as counts. The exposure offset is `log(article_age + 1)`, which accounts for older papers having had longer to accumulate citations. Reference count and team size are standardized, source is an indicator, and rows with missing predictors are excluded. Predictors with no variation in the selected sample are omitted.

A preliminary Poisson GLM estimates dispersion. When Pearson dispersion exceeds 1.5, the reported model uses a negative-binomial variance; otherwise it remains Poisson. That threshold is a heuristic, not a likelihood-based decision, so the AIC of both candidate families is displayed beside it and the panel says when AIC would have chosen the other one. Both variants report HC3 robust confidence intervals, p-values, incidence-rate ratios, sample coverage, and the selected family.

The same panel reports the misspecification evidence rather than leaving it implicit: the missingness profile of each predictor, the design condition number and per-predictor variance inflation factors, the count of observations above the 4/n Cook's-distance cut, and the gap between the observed and model-predicted fraction of zero-citation articles. No zero-inflated model is fitted, so a large positive gap marks the estimates exploratory rather than correcting them. Results describe conditional associations in this corpus and do not establish causal effects; the source indicator in particular also encodes source-specific missingness.

## Citation distributions

Heavy-tail fitting counts zero-citation articles explicitly and then fits above a data-driven `x_min`, chosen by sweeping candidate quantiles and minimising the KS distance subject to at least ten tail observations. Power-law, log-normal and exponential families are compared by AIC, not by KS distance alone, because KS does not penalise the extra free parameter. The reported goodness-of-fit p-value for the power law comes from a parametric bootstrap that refits the exponent on each simulated sample; the log-normal and exponential p-values reuse their own fitted parameters and are therefore optimistic. The panel labels which is which instead of presenting one number. Log-likelihood ratios between families are shown without a Vuong test and are labelled descriptive.

## Forecasts

Publication-volume forecasts compare a persistence baseline, linear trend, and log-linear trend. Selection uses rolling-origin validation over complete historical years. A partial current year may be shown as monitoring evidence but never participates in selection or final fitting.

Every forecast reports its skill against persistence: `1 - MAE(model) / MAE(last complete year repeated)` over the rolling-origin folds. A model that does not beat the naive baseline is labelled as such rather than presented as a projection on equal footing.

Uncertainty bands use the empirical 90th percentile of rolling one-step absolute errors, and the radius expands with the square root of the forecast horizon. Coverage of those bands is measured on folds held out of the radius calibration; scoring them on the same errors that set the radius returns the nominal level by construction and cannot fail. When the series is too short to hold folds back, no coverage is reported at all. The holdout is only three folds on an annual series, so the figure is coarse and the panel states the fold count rather than presenting a bare percentage. Every validation fold is one step ahead, so the multi-year horizon carries no direct empirical coverage evidence.

These are corpus-volume projections, not forecasts of all publications in the field.

Bass diffusion is displayed only when nonlinear least squares converges to finite, admissible parameters and the data provide enough observations. Failed fits remain unavailable; no synthetic fallback curve is presented as an estimate.

## Concept bursts

Concept bursts use a two-state implementation of Kleinberg's model. For every year, the observation is the number of matching documents out of all corpus documents in that year. Dynamic programming chooses between a baseline binomial state and an elevated state. Entering the elevated state incurs a transition penalty. Consecutive elevated years form a burst interval; strength is the interval's improvement in binomial log likelihood over the baseline state.

The concept dictionary is declared in code, so results detect changes for those named concepts rather than discovering unrestricted vocabulary.

## Semantic analyses

Contrastive screening uses the difference between similarity to an in-scope anchor and an off-topic logistics anchor. A zero margin is a semantic decision boundary, not a measured-recall guarantee. Calibration metrics are valid only against reviewed labels.

PCA and t-SNE projections are visualization aids and do not preserve every high-dimensional distance. A projection is labeled by the algorithm actually used; a failed UMAP computation must not silently appear as UMAP. Semantic isolation is a nearest-neighbor distance and must not be described as interdisciplinarity or conceptual novelty without external validation.

The map now states how much of itself is structure. Trustworthiness measures what share of a point's 2D neighbours are also its neighbours in the full 384-dimensional space, and a bootstrap adjusted Rand index over subsamples and seeds measures whether the same articles keep grouping together when the sample changes. Both are shown beside the projection, because silhouette is nearly flat across the candidate cluster counts on this corpus and a flat criterion cannot on its own justify the boundaries drawn.

Theme selection evaluates candidate cluster counts with silhouette score and rejects solutions with any cluster below 2% of the corpus; near-ties favour the smaller model. Because silhouette is nearly flat over that range, it is the tie-break rather than the maximum that decides `k` in practice. Theme labels remain descriptive summaries of the corpus, and their numbering is not stable across runs.

## Trend testing and multiplicity

Keyword trends use Mann-Kendall with Sen slopes on zero-filled annual series. The Benjamini-Hochberg adjustment is computed over every keyword meeting the minimum-prevalence rule, not over the extreme slopes displayed: adjusting a set already selected for being extreme inflates significance instead of controlling it. The table shows the charted terms carrying their family-adjusted values, and the caption states the family size. No serial-dependence correction is applied, and the keyword growth ranking, the descriptive slope chart, and the searched-breakpoint test report no adjustment at all — those remain exploratory.

## Author and network indicators

Author names are canonicalized heuristically. Homonyms can merge and spelling variants can split. The h, g, e, i10, and m indicators are computed only from articles present in this corpus and must be labeled as corpus-scoped indicators rather than full-career metrics.

Co-authorship and keyword networks describe relations within the collected corpus. Community labels, centrality, small-world diagnostics, and concentration measures inherit the corpus selection boundary.

Observed clustering is compared against degree-preserving rewirings of the same graph, reported as a z-score and an empirical p-value. The null matters because a co-authorship graph is clustered simply by virtue of papers having several authors; preserving each author's degree separates collaborative structure from that artefact. Path length and the small-world sigma are computed on the largest connected component while density and the centralities cover the whole graph including fragments, so the two are not on the same population and the panel says so.

Lotka's law is fitted on distinct-DOI counts per canonical author. Like every other person-level indicator here it is corpus-scoped, and it inherits the heuristic-identity caveat: homonyms merge and spelling variants split.

## Retrieval and anomaly audit

Hybrid retrieval combines BM25 and dense embedding ranks through reciprocal rank fusion. The interface exposes component ranks/scores so a reviewer can inspect why an item appeared. No retrieval-quality claim is made without a labeled query relevance set.

Isolation Forest results are audit priorities. They identify multivariate outliers and do not imply data errors, exceptional scientific quality, or fraud.

## Reproducibility

Every published analytical population resolves to a deterministic dataset version derived from the ordered source manifest, search configuration, transformation code, dependency lock, and duplicate-curation decisions. Consumed source bytes are retained content-addressably, while candidate Gold articles, chunks, embeddings, semantics, and duplicate pairs remain isolated until blocking contracts pass. Re-running unchanged evidence reuses the same logical version; rollback reactivates an exact prior Gold snapshot.

Analytical functions are pure Python functions without Streamlit imports. Database access belongs in `dashboard/data.py`, cached normalization and joins in `dashboard/loaders.py`, Plotly construction in `dashboard/charts.py`, and rendering in page controllers. Randomized algorithms use fixed seeds. Tests cover missingness, minimum sample sizes, degenerate inputs, deterministic behavior, source-change propagation, publication isolation, and method-specific invariants.
