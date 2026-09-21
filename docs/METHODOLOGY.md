# Analytical Methodology

This document defines what the dashboard can infer from the current corpus, how each model is validated, and which formal indicators are intentionally unavailable.

## Evidence boundaries

The analytical unit is an article in the curated Gold layer. Publication metadata comes from IEEE Xplore and Elsevier exports; citation and reference counts may be enriched from OpenAlex. Counts are cumulative snapshots, not annual citation histories. The project currently has no cited-reference publication years, forward/backward citation graph, verified open-access status, or longitudinal citation events.

Missing numeric values remain missing. Analytical functions use complete cases and report `n_total`, `n_used`, and coverage whenever a filtered sample can change interpretation. IEEE-only fields such as country, online date, document type, and license must display source coverage and cannot support corpus-wide causal comparisons.

The following indicators are unavailable until their required data is ingested:

- Price's index: requires publication years for every cited reference.
- Sleeping Beauty and beauty coefficient: require annual citation trajectories.
- CD disruption index: requires a forward and backward citation graph.
- Open Access Citation Advantage: requires verified access status and confounder controls.
- Citation half-life or longevity: requires citations indexed by citing year.

The dashboard does not estimate substitutes under those names.

## Citation count model

`citation_determinants_glm` models cumulative citations as counts. The exposure offset is `log(article_age + 1)`, which accounts for older papers having had longer to accumulate citations. Reference count and team size are standardized, source is an indicator, and rows with missing predictors are excluded. Predictors with no variation in the selected sample are omitted.

A preliminary Poisson GLM estimates dispersion. When Pearson dispersion exceeds 1.5, the reported model uses a negative-binomial variance; otherwise it remains Poisson. Both variants report HC3 robust confidence intervals, p-values, incidence-rate ratios, sample coverage, and the selected family. Results describe conditional associations in this corpus and do not establish causal effects.

## Forecasts

Publication-volume forecasts compare a persistence baseline, linear trend, and log-linear trend. Selection uses rolling-origin validation over complete historical years. A partial current year may be shown as monitoring evidence but never participates in selection or final fitting.

Uncertainty bands use the empirical 90th percentile of rolling one-step absolute errors. The radius expands with the square root of the forecast horizon. These are corpus-volume projections, not forecasts of all publications in the field.

Bass diffusion is displayed only when nonlinear least squares converges to finite, admissible parameters and the data provide enough observations. Failed fits remain unavailable; no synthetic fallback curve is presented as an estimate.

## Concept bursts

Concept bursts use a two-state implementation of Kleinberg's model. For every year, the observation is the number of matching documents out of all corpus documents in that year. Dynamic programming chooses between a baseline binomial state and an elevated state. Entering the elevated state incurs a transition penalty. Consecutive elevated years form a burst interval; strength is the interval's improvement in binomial log likelihood over the baseline state.

The concept dictionary is declared in code, so results detect changes for those named concepts rather than discovering unrestricted vocabulary.

## Semantic analyses

Contrastive screening uses the difference between similarity to an in-scope anchor and an off-topic logistics anchor. A zero margin is a semantic decision boundary, not a measured-recall guarantee. Calibration metrics are valid only against reviewed labels.

PCA and t-SNE projections are visualization aids and do not preserve every high-dimensional distance. A projection is labeled by the algorithm actually used; a failed UMAP computation must not silently appear as UMAP. Semantic isolation is a nearest-neighbor distance and must not be described as interdisciplinarity or conceptual novelty without external validation.

Theme selection evaluates candidate cluster counts with silhouette score and rejects solutions with very small clusters; ties favor the smaller model. Theme labels remain descriptive summaries of the corpus.

## Author and network indicators

Author names are canonicalized heuristically. Homonyms can merge and spelling variants can split. The h, g, e, i10, and m indicators are computed only from articles present in this corpus and must be labeled as corpus-scoped indicators rather than full-career metrics.

Co-authorship and keyword networks describe relations within the collected corpus. Community labels, centrality, small-world diagnostics, and concentration measures inherit the corpus selection boundary.

## Retrieval and anomaly audit

Hybrid retrieval combines BM25 and dense embedding ranks through reciprocal rank fusion. The interface exposes component ranks/scores so a reviewer can inspect why an item appeared. No retrieval-quality claim is made without a labeled query relevance set.

Isolation Forest results are audit priorities. They identify multivariate outliers and do not imply data errors, exceptional scientific quality, or fraud.

## Reproducibility

Every published analytical population resolves to a deterministic dataset version derived from the ordered source manifest, search configuration, transformation code, dependency lock, and duplicate-curation decisions. Consumed source bytes are retained content-addressably, while candidate Gold articles, chunks, embeddings, semantics, and duplicate pairs remain isolated until blocking contracts pass. Re-running unchanged evidence reuses the same logical version; rollback reactivates an exact prior Gold snapshot.

Analytical functions are pure Python functions without Streamlit imports. Database access belongs in `dashboard/data.py`, cached normalization and joins in `dashboard/loaders.py`, Plotly construction in `dashboard/charts.py`, and rendering in page controllers. Randomized algorithms use fixed seeds. Tests cover missingness, minimum sample sizes, degenerate inputs, deterministic behavior, source-change propagation, publication isolation, and method-specific invariants.
