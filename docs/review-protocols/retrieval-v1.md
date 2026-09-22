# Review protocol — retrieval relevance (v1)

**Workflow:** `retrieval` · **Labels:** `relevant` · `not_relevant` · `uncertain`

> **Draft.** Operational rules are mine; confirm section 3 before labelling, then
> bump to `v2` if you change anything — a protocol version is immutable once used.

## 1. What you are judging

Each subject is `<query id>::<DOI>`, for example `q03::10.1016/j.est.2026.121555`.
The query texts are versioned in `transform/evidence_samples.py::RETRIEVAL_QUERIES`;
read the one you are judging before you start on its results.

The question is: **would a researcher asking this query want this article back?**

## 2. Why this sample looks the way it does

The 96 judgements pool the top-10 of every retrieval mode — dense, BM25 and their
RRF fusion — into one set. Pooling is what makes the three comparable: judging only
what the current default returns would build the ground truth out of the default's
own hits and guarantee it wins.

So a subject is *a result some mode returned*, not a result the default returned.
You cannot tell which mode produced it, and you should not try.

## 3. Decision rules

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
