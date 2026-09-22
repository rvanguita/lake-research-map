# Evidence — the human review was blocked by tooling, not by reviewer time

**Date:** 2026-09-22 · **Branch:** `feat/roadmap-closure`
**Packages:** `WP-05`, `WP-09`, `WP-10`, `WP-11`, `WP-13` (and `WP-12`'s outstanding human half)

## What was actually blocking

Five packages were recorded as `blocked-awaiting-evidence`, waiting on independent
human labels. The assignments existed — 820 of them across five workflows, on the
active version — and the roadmap read as though the only missing input was
somebody's time.

It was not. `review_workflows.export_assignments` wrote four columns:

```
assignment_id,subject_id,label,rationale
```

and `subject_id` is an internal key. What a reviewer actually received was:

| Workflow | What the queue showed | What the protocol asks |
|---|---|---|
| `screening` | `reject::ieee::bib:03f7a342cbc9:6761637` | should this record have been in the corpus? |
| `pdf` | `10.1002/…::data/.lake_research_map/objects/ae/ae97ce4b…` | is this PDF this article? |
| `retrieval` | `q01::10.1016/j.est.2026.121555` | is this result relevant to the query? |
| `taxonomy` | `Conical / Convex Relaxation (SOCP)::10.1016/…` | does this article use this method? |
| `author` | `a abaide||a da rosa abaide` | are these the same person? |

None of these questions can be answered from the string on the left. The evidence
is on disk: `review/rene-pdf.csv` contained **its header and nothing else** — the
reviewer opened it, found a DOI concatenated with a content-addressed blob path,
and stopped. `review/rene-taxonomy.csv` had all 120 rows and **zero** filled
labels. The only completed files were `codex-assisted-*`, produced by a process
that could query the database for the context the CSV omitted.

So the packages were not waiting on a human decision. They were waiting on a
queue a human could read.

## What changed

`evidence_samples.py` builds every subject id, so it now also owns the inverse:
`describe_subjects()` resolves an id back into the evidence needed to judge it,
and `SUBJECT_COLUMNS` declares the per-workflow columns. `export_assignments`
takes the extra layer sessions and writes those columns between `subject_id` and
`label`, so the fields a reviewer types into stay rightmost in a spreadsheet,
after everything they must read.

Each workflow resolves from the layer that holds its evidence, which is why the
CLI opens four sessions and closes them in a `finally` — builders take sessions
and never open their own:

- **screening** → Bronze, keyed by exactly the `(source, source_id)` pair in the
  subject id. Bronze rather than `lit_rejected` because the rejection audit row
  carries the title but not the abstract.
- **author** → Silver, listing up to three titles per spelling. The names alone
  never settle a homonym; what decides it is what each variant publishes.
- **pdf** → Gold for the article title, Raw for the filename. `lit_pdf_files`
  maps the archived blob back to `Distribution System Planning.pdf`, which is the
  only readable half of the comparison.
- **retrieval** / **taxonomy** → Gold for title and abstract, with the query text
  from `RETRIEVAL_QUERIES` substituted for its id.

Every branch degrades to blank values rather than raising: an export that dies
because one subject no longer resolves is worse than one incomplete row, since
the reviewer can still work the other 29.

## Verification

Suite **322 passed, 2 skipped** (314 before); `ruff check` and
`ruff format --check` clean over 101 files. Eight new tests cover each workflow's
resolution, the column order, a missing-session degradation, and — the property
that matters most — that an enriched file still imports, because the added
columns must not break the path the reviewer's work comes back on.

Ten queues were regenerated against the live active version
`2974743a4b…` for both human reviewers, replacing files that held no work:

| Workflow | Subjects per reviewer |
|---|---|
| `screening` | 30 |
| `author` | 40 |
| `pdf` | 16 |
| `retrieval` | 96 |
| `taxonomy` | 120 |

Spot-checked against the live database: the screening queue's first row now
carries *Regional electricity load profile subclasses for distribution network
planning*, ICUE 2013, with its abstract.

## Residual risk

This does not produce a single label, and no package advances to complete. It
removes the reason none could be produced. `lit_model_approvals` remains empty,
so no threshold is approved and automatic exclusion stays disabled — which is
`WP-10`'s documented behaviour, not a gap.

The 216 existing labels are all from the reviewer id `codex-assisted` and remain
explicitly AI-assisted drafts. They are not independent human evidence and must
not be counted as the second reviewer in any agreement statistic.
