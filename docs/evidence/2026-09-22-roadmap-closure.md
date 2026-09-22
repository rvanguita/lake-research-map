# Evidence — closing what WP-10, WP-24, WP-09, WP-07 and WP-08 could still close

**Date:** 2026-09-22 · **Branch:** `feat/roadmap-closure`

Five packages were worked in the order `WP-10 → WP-24 → WP-09 → WP-07 → WP-08`.
Four of them were recorded as waiting on evidence that only a human or a
credentialed crawl can produce. That turned out to be true of the *measurements*
and not of the *mechanisms*: in each package the code that consumes the missing
evidence could not have consumed it even if it had arrived. This note records
what was closed, what was measured, and what genuinely still waits.

## Measured baseline

- Suite before: **280 passed, 1 skipped**. After: **303 passed, 1 skipped** (+23).
- `uv run ruff check` and `uv run ruff format --check` pass.
- MySQL 8.4.11 reachable; active dataset version
  `2974743a4b264bcd2dec6b6749360e9c7af6b2dd3da827a8a079f2c21b646e4e`.
- `audit citation-graph` on the live database: 0 OpenAlex works observed, so
  citation coverage is 0 *by absence rather than by measurement* — which is now
  a sentence the tool prints rather than an inference a reader has to make.
- `reviews status --workflow screening`: 0 assignments, 0 labels on the active
  version. The screening round has not been started, which is the honest state.

## WP-10 — the calibration could not read a stored label

`reviews import` wrote reviewer decisions to `lit_review_labels`, and
`calibrate_screening_threshold` accepted only a CSV uploaded in the browser.
The two halves of the package never touched: a durable label could not reach
the threshold it exists to validate, and the in-memory panel's evidence
disappeared when the tab closed.

- `review_workflows.collect_labels` reads the durable tables into the long form
  the calibrator already speaks. Only the newest revision per assignment is
  returned — the table is append-only, so a corrected decision is a new row and
  every earlier revision is history, not a second opinion. An adjudication
  arrives as an extra row under the reserved reviewer `adjudicated`, which is
  the marker `resolve_review_consensus` already looks for, so a final decision
  outranks a disagreement without erasing the raw labels behind it.
- `screening_calibration.label_set_digest` computes the `label_set_sha256` that
  `approve_model` has always demanded and that **nothing computed**. The column
  was there to be filled in by hand, and a hand-typed hash binds an approval to
  nothing. The digest covers the raw reviewer rows rather than the resolved
  consensus, because two different label sets can resolve identically and an
  approval that cannot tell them apart has not recorded its evidence.
- `screening_model_digest` identifies the rule being approved: embedding model,
  both anchors, the cut, and the sensitivity target. Change any one and the
  approved sensitivity stops describing what the dashboard would do.
- `reviews calibrate` runs the whole chain off the database and prints both
  digests plus the exact `reviews approve` invocation. It applies nothing. A
  threshold that excludes work is not allowed to select itself.

Residual risk: none of this produces a number without labels. With none stored,
the command refuses and says why.

## WP-24 — the capability existed only in the test suite

`fetch_openalex_citing_works`, `persist_incoming_edges` and
`citation_graph_coverage` were implemented and fixture-tested, and **no caller
existed anywhere outside `tests/`**. The roadmap's own invariant — no public
function unreachable from a surface — did not hold here.

- `enrichment refresh-citations` drives the crawl from `lit_external_works`
  rather than from DOIs, because `cites:` filters on an OpenAlex work id: a DOI
  the backward pass never resolved has no id to crawl. Truncated works are
  counted in the run stats, so an exhausted page budget stays distinguishable
  from a genuinely uncited work.
- `audit citation-graph` reports backward and forward coverage separately and
  the usable population as their intersection, and says plainly when the gate
  fails that CD/disruption staying unavailable is the documented outcome rather
  than a defect to fix in code.

Residual risk: coverage is still unmeasured, because the crawl needs
`OPENALEX_EMAIL` and thousands of live requests. What changed is that measuring
it is now one command instead of unwritten code.

## WP-09 — the agreement report had no reader outside the browser

- `reviews status` reports progress, pairwise agreement with Cohen's kappa, the
  label-set digest, and the disagreement queue, from the durable tables.
- `assignment_progress` breaks progress out per reviewer. A round where one
  reviewer finished and the other never started produces plenty of labels and
  zero meaningful agreement, and a single completion percentage hides exactly
  that case.
- A screening subject is not always a DOI: `evidence rejections` emits
  `reject::<source>::<id>` for records dropped before they ever had a margin.
  Those cannot join a threshold evaluation, and they are now reported as
  `unknown_doi` rather than vanishing in the join — a round whose labels mostly
  disappear there otherwise looks identical to a round nobody labelled.

Residual risk: unchanged and not closable here. The protocols in
`docs/review-protocols/` are drafts whose section 3 is a genuine research
decision, and independent reviewer labels are human work.

## WP-07 — the refresh demanded a credential that does not exist

`_run_enrichment_command` required **both** `OPENALEX_API_KEY` and
`OPENALEX_EMAIL`. OpenAlex issues no API key for the public corpus — the
module's own docstring says so — so every refresh was unreachable by
construction, whatever the operator put in `.env`. This is the same shape of
defect as the database roles WP-08 removed on 2026-09-22: a credential gate
nobody could satisfy, failing in a way that pointed at the wrong thing.

The gate now requires `OPENALEX_EMAIL` alone, with the reason stated in the
error, and treats the key as optional. Relatedly, the `User-Agent` carried a
fixed `mailto:researcher@example.com`. An address nobody reads is worse than no
address — it is what OpenAlex would contact about a misbehaving crawl — so the
contact now comes from the environment and the header omits it when there is
none.

Residual risk: the corpus refresh itself still needs the operator to set
`OPENALEX_EMAIL` and accept a live crawl. That is a decision, not a defect.

## WP-08 — recovery depended on someone remembering to type a command

`maintenance recover-stale` was the only thing that ever cleared a killed run.
Until someone ran it, `lit_pipeline_executions` claimed a run was in progress
that had not existed for days — and the dashboard's status reads those rows.

`_pipeline_lock` now yields whether it actually acquired the MySQL advisory
lock, and `run()` sweeps abandoned executions when it did. Holding that lock is
proof that no other writer is live, which is what makes it safe to treat every
surviving `running` row as abandoned instead of waiting out an age cutoff.
Under SQLite the flag is `False` and no sweep happens, because nothing was
proven. The sweep is housekeeping: if it fails it is logged and the run
proceeds. `recover_abandoned_executions` is shared with the manual command,
which keeps its `--older-than-minutes` cutoff.

The status line also claimed the retry/timeout policy was open. It was not:
`STAGE_POLICY` in `airflow/dags/lake_research_map_dags.py` has carried
per-stage `retries` and `execution_timeout` since before this cycle. Corrected
rather than re-delivered.

## Follow-up decisions, for the owner

1. Set `OPENALEX_EMAIL` and authorize a live crawl, or record that WP-07's
   refresh and WP-24's coverage stay unmeasured by choice.
2. Replace section 3 of `docs/review-protocols/screening-v1.md` with the
   review's real inclusion criteria and bump to `v2`; a protocol version is
   immutable once used.
3. Run a screening round (`evidence rejections` → `reviews setup` → `export` →
   independent labelling → `import` → `status` → `calibrate`), then decide the
   approval. Automatic exclusion stays disabled until that decision exists.
