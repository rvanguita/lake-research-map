"""Executable, persisted data contracts for medallion publication gates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.gold_models import (
    DatasetArticle,
    DatasetChunk,
    DatasetDuplicatePair,
    DatasetSemantics,
    QualityResult,
)
from lake_research_map.db.raw_models import (
    BibEntry,
    DatasetSourceFile,
    IeeeCsvRow,
    PdfFile,
    SourceFile,
)
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.db.silver_models import RejectedArticle
from lake_research_map.transform.bronze_articles import normalize_doi
from lake_research_map.transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION
from lake_research_map.transform.publication_categories import PUBLICATION_CATEGORIES

EMBED_DIMENSION = 384


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    severity: str
    passed: bool
    observed: Any
    expected: Any
    details: dict[str, Any] = field(default_factory=dict)


class ContractViolation(RuntimeError):
    def __init__(self, stage: str, failures: list[CheckResult]):
        self.stage = stage
        self.failures = failures
        super().__init__(
            f"{stage} contract failed: " + ", ".join(result.check_id for result in failures)
        )


def _result(
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    *,
    severity: str = "error",
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(check_id, severity, passed, observed, expected, details or {})


def raw_contract(session: Session, version_id: str) -> list[CheckResult]:
    manifest = session.scalars(select(SourceFile)).all()
    version_rows = session.scalars(
        select(DatasetSourceFile).where(DatasetSourceFile.dataset_version_id == version_id)
    ).all()
    paths = {row.path for row in manifest}
    child_paths = {
        *(row.source_file for row in session.scalars(select(BibEntry)).all()),
        *(row.source_file for row in session.scalars(select(IeeeCsvRow)).all()),
        *(row.path for row in session.scalars(select(PdfFile)).all()),
    }
    uniform = {row.dataset_version_id for row in manifest}
    return [
        _result("raw.non_empty_manifest", bool(manifest), len(manifest), "> 0"),
        _result(
            "raw.snapshot_coverage",
            len(version_rows) == len(manifest),
            len(version_rows),
            len(manifest),
        ),
        _result(
            "raw.version_uniformity",
            uniform == {version_id},
            sorted(str(v) for v in uniform),
            [version_id],
        ),
        _result(
            "raw.child_source_references",
            child_paths <= paths,
            len(child_paths - paths),
            0,
            details={"orphan_paths": sorted(child_paths - paths)[:20]},
        ),
    ]


def bronze_contract(
    raw_session: Session, bronze_session: Session, version_id: str
) -> list[CheckResult]:
    rows = bronze_session.scalars(select(BronzeArticle)).all()
    keys = [(row.source, row.source_id) for row in rows]
    raw_bib_ids = set(raw_session.scalars(select(BibEntry.id)).all())
    raw_csv_ids = set(raw_session.scalars(select(IeeeCsvRow.id)).all())
    broken_refs = [
        row.id
        for row in rows
        if (row.raw_bib_id is not None and row.raw_bib_id not in raw_bib_ids)
        or (row.raw_csv_id is not None and row.raw_csv_id not in raw_csv_ids)
    ]
    versions = {row.dataset_version_id for row in rows}
    invalid_categories = [
        row.id
        for row in rows
        if row.publication_category not in PUBLICATION_CATEGORIES
        or not row.publication_category_basis
    ]
    return [
        _result("bronze.non_empty", bool(rows), len(rows), "> 0"),
        _result(
            "bronze.natural_key_unique", len(keys) == len(set(keys)), len(keys) - len(set(keys)), 0
        ),
        _result(
            "bronze.raw_references",
            not broken_refs,
            len(broken_refs),
            0,
            details={"row_ids": broken_refs[:20]},
        ),
        _result(
            "bronze.version_uniformity",
            versions == {version_id},
            sorted(str(v) for v in versions),
            [version_id],
        ),
        _result(
            "bronze.publication_categories",
            not invalid_categories,
            len(invalid_categories),
            0,
            details={"row_ids": invalid_categories[:20]},
        ),
    ]


def silver_contract(
    bronze_session: Session, silver_session: Session, raw_session: Session, version_id: str
) -> list[CheckResult]:
    bronze_rows = bronze_session.scalars(select(BronzeArticle)).all()
    rows = silver_session.scalars(select(SilverArticle)).all()
    rejected = silver_session.scalars(select(RejectedArticle)).all()
    dois = [row.doi for row in rows]
    represented = {int(value) for row in rows for value in (row.bronze_ids or [])}
    rejected_ids = {row.bronze_id for row in rejected}
    expected_ids = {row.id for row in bronze_rows}
    pdf_paths = {row.archive_path or row.path for row in raw_session.scalars(select(PdfFile)).all()}
    invalid_pdf = [row.doi for row in rows if row.has_pdf and row.pdf_path not in pdf_paths]
    normalized = all(doi and normalize_doi(doi) == doi for doi in dois)
    versions = {row.dataset_version_id for row in rows} | {
        row.dataset_version_id for row in rejected
    }
    invalid_categories = [
        row.doi
        for row in rows
        if row.publication_category not in PUBLICATION_CATEGORIES
        or not row.publication_category_basis
    ]
    return [
        _result("silver.doi_unique", len(dois) == len(set(dois)), len(dois) - len(set(dois)), 0),
        _result(
            "silver.doi_normalized", normalized, sum(normalize_doi(doi) != doi for doi in dois), 0
        ),
        _result(
            "silver.bronze_partition",
            represented.isdisjoint(rejected_ids) and represented | rejected_ids == expected_ids,
            {"represented": len(represented), "rejected": len(rejected_ids)},
            {"bronze": len(expected_ids), "overlap": 0},
        ),
        _result(
            "silver.pdf_references",
            not invalid_pdf,
            len(invalid_pdf),
            0,
            details={"dois": invalid_pdf[:20]},
        ),
        _result(
            "silver.version_uniformity",
            versions <= {version_id},
            sorted(str(v) for v in versions),
            [version_id],
        ),
        _result(
            "silver.publication_categories",
            not invalid_categories,
            len(invalid_categories),
            0,
            details={"dois": invalid_categories[:20]},
        ),
    ]


def gold_contract(session: Session, version_id: str) -> list[CheckResult]:
    articles = session.scalars(
        select(DatasetArticle).where(DatasetArticle.dataset_version_id == version_id)
    ).all()
    chunks = session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
    ).all()
    dois = [row.doi for row in articles]
    doi_set = set(dois)
    keys = [(row.doi, row.chunk_type, row.seq) for row in chunks]
    orphans = [row.id for row in chunks if row.doi not in doi_set]
    bad_lengths = [row.id for row in chunks if row.char_len != len(row.text)]
    invalid_categories = [
        row.doi
        for row in articles
        if row.publication_category not in PUBLICATION_CATEGORIES
        or not row.publication_category_basis
    ]
    return [
        _result("gold.non_empty", bool(articles), len(articles), "> 0"),
        _result("gold.doi_unique", len(dois) == len(doi_set), len(dois) - len(doi_set), 0),
        _result(
            "gold.chunk_key_unique", len(keys) == len(set(keys)), len(keys) - len(set(keys)), 0
        ),
        _result(
            "gold.chunk_parent_integrity",
            not orphans,
            len(orphans),
            0,
            details={"chunk_ids": orphans[:20]},
        ),
        _result(
            "gold.chunk_lengths",
            not bad_lengths,
            len(bad_lengths),
            0,
            details={"chunk_ids": bad_lengths[:20]},
        ),
        _result(
            "gold.publication_categories",
            not invalid_categories,
            len(invalid_categories),
            0,
            details={"dois": invalid_categories[:20]},
        ),
    ]


def embed_contract(session: Session, version_id: str) -> list[CheckResult]:
    chunks = session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
    ).all()
    missing = []
    incompatible = []
    stale = []
    non_finite = []
    for row in chunks:
        if row.embedding_bin is None:
            missing.append(row.id)
            continue
        if (
            len(row.embedding_bin) != EMBED_DIMENSION * 4
            or row.embed_model != EMBED_MODEL_NAME
            or row.embed_revision != EMBED_MODEL_REVISION
            or row.embedding_dim != EMBED_DIMENSION
            or row.embedding_dtype != "float32"
            or row.embedding_normalized is not True
        ):
            incompatible.append(row.id)
            continue
        expected_hash = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
        if row.text_sha256 != expected_hash:
            stale.append(row.id)
            continue
        if not np.isfinite(np.frombuffer(row.embedding_bin, dtype=np.float32)).all():
            non_finite.append(row.id)
    return [
        _result(
            "embed.complete", not missing, len(missing), 0, details={"chunk_ids": missing[:20]}
        ),
        _result(
            "embed.compatible",
            not incompatible,
            len(incompatible),
            0,
            details={"chunk_ids": incompatible[:20]},
        ),
        _result(
            "embed.finite",
            not non_finite,
            len(non_finite),
            0,
            details={"chunk_ids": non_finite[:20]},
        ),
        _result(
            "embed.text_hash",
            not stale,
            len(stale),
            0,
            details={"chunk_ids": stale[:20]},
        ),
    ]


def semantic_contract(session: Session, version_id: str) -> list[CheckResult]:
    abstracts = session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == version_id)
        .where(DatasetChunk.chunk_type == "abstract")
    ).all()
    rows = session.scalars(
        select(DatasetSemantics).where(DatasetSemantics.dataset_version_id == version_id)
    ).all()
    pairs = session.scalars(
        select(DatasetDuplicatePair).where(DatasetDuplicatePair.dataset_version_id == version_id)
    ).all()
    semantic_dois = [row.doi for row in rows]
    expected_dois = {row.doi for row in abstracts}
    finite = all(
        all(math.isfinite(value) for value in (row.relevance_score, row.map_x, row.map_y))
        and (row.offtopic_score is None or math.isfinite(row.offtopic_score))
        for row in rows
    )
    ranges = all(
        -1 <= row.relevance_score <= 1
        and (row.offtopic_score is None or -1 <= row.offtopic_score <= 1)
        for row in rows
    )
    pair_keys = [(row.doi_a, row.doi_b) for row in pairs]
    invalid_pairs = [key for key in pair_keys if key[0] >= key[1] or not set(key) <= expected_dois]
    return [
        _result(
            "semantic.coverage",
            set(semantic_dois) == expected_dois,
            len(semantic_dois),
            len(expected_dois),
        ),
        _result(
            "semantic.doi_unique",
            len(semantic_dois) == len(set(semantic_dois)),
            len(semantic_dois) - len(set(semantic_dois)),
            0,
        ),
        _result("semantic.finite", finite, finite, True),
        _result("semantic.score_ranges", ranges, ranges, True),
        _result(
            "semantic.duplicate_pairs",
            not invalid_pairs and len(pair_keys) == len(set(pair_keys)),
            len(invalid_pairs) + len(pair_keys) - len(set(pair_keys)),
            0,
        ),
    ]


def publication_contract(session: Session, version_id: str) -> list[CheckResult]:
    """Return the blocking contracts required before Gold materialization.

    The pipeline records stage-specific results before calling the publisher,
    but `materialize_version()` is also a public service boundary used by the
    CLI reactivation command and recovery tooling. Re-evaluating the candidate
    here prevents a caller from bypassing the publication gate by invoking the
    materializer directly.
    """
    return [
        *gold_contract(session, version_id),
        *embed_contract(session, version_id),
        *semantic_contract(session, version_id),
    ]


def persist_results(
    session: Session,
    *,
    execution_id: str,
    stage_run_id: int,
    dataset_version_id: str,
    stage: str,
    results: list[CheckResult],
) -> None:
    session.add_all(
        [
            QualityResult(
                execution_id=execution_id,
                stage_run_id=stage_run_id,
                dataset_version_id=dataset_version_id,
                stage=stage,
                check_id=result.check_id,
                severity=result.severity,
                passed=result.passed,
                observed=result.observed,
                expected=result.expected,
                details=result.details,
            )
            for result in results
        ]
    )
    session.flush()


def assert_contract(stage: str, results: list[CheckResult]) -> None:
    failures = [result for result in results if result.severity == "error" and not result.passed]
    if failures:
        raise ContractViolation(stage, failures)
