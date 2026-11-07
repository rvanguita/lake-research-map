"""The six medallion stages, `run_all` and `run` (no orchestrator, ADR-09).

The command line is `cli.main` (the `lake-research-map` console script); this
module holds only what a stage computes, and imports no application. Run
bookkeeping, the lock and recovery are `runs.py`.

Usage (`uv run lake-research-map <group> --help` lists each group's actions):
    --stage raw|bronze|silver|gold|embed|semantic|all   run one stage, or the chain (default: all)
    duplicates   list | merge | keep | undo -- human decisions on near-duplicate pairs
    versions     list | activate -- inspect dataset versions, reactivate a published snapshot
    reviews      setup | export | import | adjudicate | approve | status | calibrate -- human evidence
    enrichment   refresh-openalex | refresh-citations | crossref-references | resolve-references
                 -- OpenAlex / Crossref metadata crawls
    evidence     sample review subjects into a CSV (read-only against the corpus)
    audit        reference-corpus | citation-graph | citation-years -- read-only acceptance audits
    maintenance  recover-stale -- fail executions that stopped heartbeating
    vectors      status | backfill | rebuild | sync-alias -- inspect and repair the Qdrant store
    search       "query" -- article-level ranking (--mode, --scope, --rerank, filters)
    screen       "subject" -- score every article against a subject and its off-subject reading
    mcp          the retrieval operations as MCP tools over stdio
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from db.bootstrap import bootstrap
from db.vector_store import get_vector_store
from ingest.raw_bib import load_bib_entries
from ingest.raw_config import load_configs
from ingest.raw_csv import load_ieee_csv
from ingest.raw_pdfs import load_pdf_inventory
from ingest.reconciliation import (
    active_manifest,
    persist_snapshot,
    remove_absent_sources,
)
from ingest.snapshots import ScannedSource, build_fingerprint, scan_sources
from quality import (
    ContractViolation,
    assert_contract,
    bronze_contract,
    embed_contract,
    gold_contract,
    raw_contract,
    semantic_contract,
    silver_contract,
)
from runs import (
    STAGE_ORDER,
    _begin_stage,
    _complete_skipped_stage,
    _ensure_execution,
    _finalize_execution,
    _finish_stage,
    _handle_stage_failure,
    _is_unchanged_full_run,
    _layer_sessions,
    _pipeline_lock,
    _record_results,
    _recover_before_run,
    _working_version,
)
from transform.bronze_articles import build_bronze_articles
from transform.publication import materialize_version, publish_vector_alias
from transform.silver_articles import build_silver_articles
from transform.versioned_gold import (
    build_dataset_embeddings,
    build_dataset_gold,
    build_dataset_semantics,
)

STAGES = (*STAGE_ORDER, "all")


def _curation_records(session) -> list[dict]:
    from db.gold_models import DuplicateOverride

    return [
        {
            "doi_a": row.doi_a,
            "doi_b": row.doi_b,
            "decision": row.decision,
            "canonical_doi": row.canonical_doi,
            "reason": row.reason,
        }
        for row in session.scalars(
            select(DuplicateOverride).order_by(DuplicateOverride.doi_a, DuplicateOverride.doi_b)
        )
    ]


def _append_effective_sources(raw_session, scanned: list[ScannedSource]) -> list[ScannedSource]:
    """Retain archived active paths that are absent from an additive download batch."""
    from db.raw_models import SourceBlob, SourceFile, SourceRevision

    effective = {source.path: source for source in scanned}
    for row in raw_session.scalars(select(SourceFile)).all():
        if row.path in effective:
            continue
        revision = (
            raw_session.get(SourceRevision, row.source_revision_id)
            if row.source_revision_id
            else raw_session.scalar(
                select(SourceRevision).where(
                    SourceRevision.path == row.path, SourceRevision.sha256 == row.sha256
                )
            )
        )
        blob = raw_session.get(SourceBlob, row.sha256)
        if revision is None or blob is None:
            raise RuntimeError(
                f"append policy cannot retain {row.path!r}: archived revision is incomplete"
            )
        effective[row.path] = ScannedSource(
            path=row.path,
            source=row.source,
            kind=row.kind,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            mtime=row.mtime,
            revision_id=revision.revision_id,
            archive_path=blob.archive_path,
        )
    return sorted(effective.values(), key=lambda source: source.path)


def _enrichment_records() -> list[list]:
    """The observation state bronze will read, as a stable fingerprint input."""
    from ingest.enrichment import load_enrichment_observations

    with _layer_sessions("bronze") as (session,):
        try:
            observed = load_enrichment_observations(session)
        except Exception:  # pragma: no cover - a pre-enrichment database
            return []
    return [
        [doi, values.get("citation_count"), values.get("reference_count")]
        for doi, values in sorted(observed.items())
    ]


def _prepare_raw_version(execution_id: str, workflow: str, trigger: str, source_policy: str):
    from db.gold_models import DatasetVersion, PublicationState

    sources = scan_sources()
    with _layer_sessions("raw") as (raw_session,):
        if source_policy == "append":
            sources = _append_effective_sources(raw_session, sources)
    with _layer_sessions("gold") as (gold_session,):
        fingerprint = build_fingerprint(
            sources,
            curation_records=_curation_records(gold_session),
            enrichment_records=_enrichment_records(),
        )
        state = gold_session.get(PublicationState, 1)
        if state is None:
            state = PublicationState(id=1)
            gold_session.add(state)
        version = gold_session.get(DatasetVersion, fingerprint.version_id)
        if version is None:
            version = DatasetVersion(
                version_id=fingerprint.version_id,
                parent_version_id=state.active_version_id,
                source_manifest_sha256=fingerprint.source_manifest_sha256,
                config_sha256=fingerprint.config_sha256,
                code_revision=fingerprint.code_revision,
                code_sha256=fingerprint.code_sha256,
                curation_sha256=fingerprint.curation_sha256,
                status="candidate",
            )
            gold_session.add(version)
        elif version.status == "failed":
            version.status = "candidate"
            version.failed_at = None
            version.failure_reason = None
        state.working_version_id = fingerprint.version_id
        state.execution_id = execution_id
        _ensure_execution(
            gold_session,
            execution_id,
            workflow=workflow,
            trigger=trigger,
            version_id=fingerprint.version_id,
        )
        gold_session.commit()
        return sources, fingerprint.version_id, version.status == "active"


def run_raw(
    *,
    execution_id: str,
    workflow: str = "raw",
    trigger: str = "cli",
    source_policy: str = "append",
) -> dict:
    sources, version_id, already_active = _prepare_raw_version(
        execution_id, workflow, trigger, source_policy
    )
    with _layer_sessions("gold", "raw") as (gold_session, raw_session):
        run_id, started = _begin_stage(gold_session, "raw", execution_id, version_id)
        try:
            if already_active and workflow == "all":
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "raw"
                )
            previous = active_manifest(raw_session)
            removed = (
                remove_absent_sources(raw_session, {source.path for source in sources})
                if source_policy == "snapshot"
                else 0
            )
            stats = {
                "source_policy": source_policy,
                "config": load_configs(raw_session),
                "ieee_csv_rows": load_ieee_csv(raw_session),
                "bib_entries": load_bib_entries(raw_session),
                "pdf_files": load_pdf_inventory(raw_session),
            }
            changes = persist_snapshot(
                raw_session,
                sources,
                version_id=version_id,
                execution_id=execution_id,
                previous=previous,
            )
            stats.update(changes)
            stats["removed_active_paths"] = removed
            results = raw_contract(raw_session, version_id)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="raw",
                results=results,
            )
            gold_session.commit()
            assert_contract("raw", results)
            raw_session.commit()
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            if workflow != "all":
                _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            print(f"[raw] {stats}")
            return stats
        except Exception as exc:
            raw_session.rollback()
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def run_bronze(*, execution_id: str, workflow: str = "bronze", trigger: str = "cli") -> dict:
    with _layer_sessions("gold", "raw", "bronze") as (gold_session, raw_session, bronze_session):
        version_id = _working_version(gold_session, execution_id)
        _ensure_execution(
            gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
        )
        gold_session.commit()
        run_id, started = _begin_stage(gold_session, "bronze", execution_id, version_id)
        try:
            if _is_unchanged_full_run(gold_session, version_id, workflow):
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "bronze"
                )
            stats = build_bronze_articles(raw_session, bronze_session, version_id)
            results = bronze_contract(raw_session, bronze_session, version_id)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="bronze",
                results=results,
            )
            gold_session.commit()
            assert_contract("bronze", results)
            bronze_session.commit()
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            if workflow != "all":
                _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            print(f"[bronze] {stats}")
            return stats
        except Exception as exc:
            bronze_session.rollback()
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def run_silver(*, execution_id: str, workflow: str = "silver", trigger: str = "cli") -> dict:
    with _layer_sessions("gold", "raw", "bronze", "silver") as (
        gold_session,
        raw_session,
        bronze_session,
        silver_session,
    ):
        version_id = _working_version(gold_session, execution_id)
        _ensure_execution(
            gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
        )
        gold_session.commit()
        run_id, started = _begin_stage(gold_session, "silver", execution_id, version_id)
        try:
            if _is_unchanged_full_run(gold_session, version_id, workflow):
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "silver"
                )
            stats = build_silver_articles(bronze_session, silver_session, raw_session, version_id)
            results = silver_contract(bronze_session, silver_session, raw_session, version_id)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="silver",
                results=results,
            )
            gold_session.commit()
            assert_contract("silver", results)
            silver_session.commit()
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            if workflow != "all":
                _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            print(f"[silver] {stats}")
            return stats
        except Exception as exc:
            silver_session.rollback()
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def run_gold(*, execution_id: str, workflow: str = "gold", trigger: str = "cli") -> dict:
    with _layer_sessions("gold", "silver") as (gold_session, silver_session):
        version_id = _working_version(gold_session, execution_id)
        _ensure_execution(
            gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
        )
        gold_session.commit()
        run_id, started = _begin_stage(gold_session, "gold", execution_id, version_id)
        try:
            if _is_unchanged_full_run(gold_session, version_id, workflow):
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "gold"
                )
            stats = build_dataset_gold(silver_session, gold_session, version_id)
            results = gold_contract(gold_session, version_id)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="gold",
                results=results,
            )
            gold_session.commit()
            assert_contract("gold", results)
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            if workflow != "all":
                _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            print(f"[gold] {stats}")
            return stats
        except Exception as exc:
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def run_embed(*, execution_id: str, workflow: str = "embed", trigger: str = "cli") -> dict:
    with _layer_sessions("gold") as (gold_session,):
        version_id = _working_version(gold_session, execution_id)
        _ensure_execution(
            gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
        )
        gold_session.commit()
        run_id, started = _begin_stage(gold_session, "embed", execution_id, version_id)
        try:
            if _is_unchanged_full_run(gold_session, version_id, workflow):
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "embed"
                )
            gold_session.info["execution_id"] = execution_id
            # Resolved after the skip check: an unchanged run never needs Qdrant.
            store = get_vector_store()
            stats = build_dataset_embeddings(gold_session, version_id, store=store)
            results = embed_contract(gold_session, version_id, store)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="embed",
                results=results,
            )
            gold_session.commit()
            assert_contract("embed", results)
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            if workflow != "all":
                _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            print(f"[embed] {stats}")
            return stats
        except Exception as exc:
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def _version_ready(session, version_id: str) -> bool:
    from db.gold_models import PipelineRun, QualityResult

    for stage in STAGES[:-1]:
        run_id = session.scalar(
            select(PipelineRun.id)
            .where(
                PipelineRun.dataset_version_id == version_id,
                PipelineRun.stage == stage,
                PipelineRun.status == "success",
            )
            .order_by(PipelineRun.id.desc())
            .limit(1)
        )
        if run_id is None:
            return False
        blocking_failure = session.scalar(
            select(func.count())
            .select_from(QualityResult)
            .where(
                QualityResult.stage_run_id == run_id,
                QualityResult.severity == "error",
                QualityResult.passed.is_(False),
            )
        )
        if blocking_failure:
            return False
    return True


def run_semantic(*, execution_id: str, workflow: str = "semantic", trigger: str = "cli") -> dict:
    with _layer_sessions("gold") as (gold_session,):
        version_id = _working_version(gold_session, execution_id)
        _ensure_execution(
            gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
        )
        gold_session.commit()
        run_id, started = _begin_stage(gold_session, "semantic", execution_id, version_id)
        try:
            if _is_unchanged_full_run(gold_session, version_id, workflow):
                return _complete_skipped_stage(
                    gold_session, run_id, started, execution_id, version_id, "semantic"
                )
            store = get_vector_store()
            stats = build_dataset_semantics(gold_session, version_id, store=store)
            results = semantic_contract(gold_session, version_id)
            _record_results(
                gold_session,
                execution_id=execution_id,
                run_id=run_id,
                version_id=version_id,
                stage="semantic",
                results=results,
            )
            gold_session.commit()
            assert_contract("semantic", results)
            _finish_stage(gold_session, run_id, started, stats=stats, status="success")
            gold_session.flush()
            if not _version_ready(gold_session, version_id):
                raise ContractViolation("publication", [])
            materialize_version(gold_session, version_id, execution_id, store=store)
            _finalize_execution(gold_session, execution_id, "success")
            gold_session.commit()
            publish_vector_alias(store, version_id)
            print(f"[semantic] {stats}")
            return stats
        except Exception as exc:
            _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
            raise


def run_all(
    *,
    execution_id: str | None = None,
    trigger: str = "cli",
    source_policy: str = "append",
) -> dict:
    """Every stage in order under one execution id.

    Called by `run("all")`, which already holds the pipeline lock and has run
    `bootstrap()`; bootstrapping again here only repeated the schema checks.
    """
    execution_id = execution_id or str(uuid.uuid4())
    return {
        "raw": run_raw(
            execution_id=execution_id,
            workflow="all",
            trigger=trigger,
            source_policy=source_policy,
        ),
        "bronze": run_bronze(execution_id=execution_id, workflow="all", trigger=trigger),
        "silver": run_silver(execution_id=execution_id, workflow="all", trigger=trigger),
        "gold": run_gold(execution_id=execution_id, workflow="all", trigger=trigger),
        "embed": run_embed(execution_id=execution_id, workflow="all", trigger=trigger),
        "semantic": run_semantic(execution_id=execution_id, workflow="all", trigger=trigger),
    }


def run(
    stage: str,
    *,
    execution_id: str | None = None,
    trigger: str = "cli",
    workflow: str | None = None,
    source_policy: str = "append",
) -> None:
    execution_id = execution_id or str(uuid.uuid4())
    workflow = workflow or stage
    with _pipeline_lock(execution_id) as exclusive:
        bootstrap()
        if exclusive:
            _recover_before_run(execution_id)
        if stage == "all":
            run_all(execution_id=execution_id, trigger=trigger, source_policy=source_policy)
            return
        runners = {
            "raw": run_raw,
            "bronze": run_bronze,
            "silver": run_silver,
            "gold": run_gold,
            "embed": run_embed,
            "semantic": run_semantic,
        }
        if stage == "raw":
            run_raw(
                execution_id=execution_id,
                workflow=workflow,
                trigger=trigger,
                source_policy=source_policy,
            )
        else:
            runners[stage](execution_id=execution_id, workflow=workflow, trigger=trigger)
