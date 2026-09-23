# Review protocol — no-DOI rejection audit (v2)

**Workflow:** `screening` · **Labels:** `include` · `exclude` · `uncertain`

> **Approved for use on 2026-09-22.** This version is immutable once persisted.
> Reviewers must work independently and must not inspect the other reviewer's
> decision before submitting their own.

## 1. Review question

Each subject is `reject::<source>::<source id>`: a publisher record excluded at
Silver because no DOI could be normalized. Decide whether the record would have
belonged in the review independently of its missing DOI.

The review scope is research on planning electric-power distribution systems or
distribution networks. A planning contribution makes a medium- or long-term
decision about investments, capacity, topology, location, sizing, reinforcement,
expansion, replacement, or staged development of distribution assets or resources.

## 2. Evidence to inspect

Use the title, abstract, publication type, venue, keywords, and publisher metadata
available for the rejected record. Do not infer relevance from the search query,
publisher, venue, authors, or neighboring records alone. A missing DOI is the
reason for this audit and is never itself a reason to exclude the record.

## 3. Decision rules

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

## 4. Precedence and rationale

Apply publication-type exclusions first, then physical-system scope, then the
planning-decision requirement. When a record contains both eligible and ineligible
material, label `include` if electric-distribution planning is a substantive study
objective or evaluated application rather than background context.

Every `include` and `uncertain` decision requires a concise rationale quoting or
paraphrasing the decisive metadata. For `exclude`, record the first applicable
exclusion category. Reviewers must not attempt to recover or merge the record;
adjudication and any later remediation are separate steps.

## 5. Interpretation

The audit estimates selection bias caused by DOI-based exclusion. It does not
retroactively add records to Gold. Disagreements preserve both raw labels and are
resolved through the versioned adjudication workflow.
