"""Build, validate, publish, and reactivate immutable Gold snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import (
    Article,
    Chunk,
    DatasetArticle,
    DatasetChunk,
    DatasetDuplicatePair,
    DatasetSemantics,
    DatasetVersion,
    DuplicateOverride,
    DuplicatePair,
    PublicationState,
    Semantics,
)
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.duplicate_resolution import active_merge_plan
from lake_research_map.transform.embeddings import EMBED_BATCH_SIZE, EMBED_MODEL_NAME
from lake_research_map.transform.gold_articles import _desired_chunks, _merge_silver_group
from lake_research_map.transform.semantics import (
    ANCHOR_TEXT,
    OFF_ANCHOR_TEXT,
    _embed_anchors,
    discover_themes,
    near_duplicate_pairs,
    project_2d,
    reduced_space,
    relevance_scores,
    unresolved_duplicate_pairs,
)


def _state(session: Session) -> PublicationState:
    state = session.get(PublicationState, 1)
    if state is None:
        state = PublicationState(id=1)
        session.add(state)
        session.flush()
    return state


def _assert_mutable_candidate(session: Session, dataset_version_id: str) -> None:
    """Reject writes to a published snapshot.

    A curation or source change must first produce a new fingerprint through
    the Raw stage.  Rebuilding derived rows in-place would make the version ID
    stop describing immutable evidence and would also corrupt exact rollback.
    """
    version = session.get(DatasetVersion, dataset_version_id)
    if version is not None and version.status == "active":
        raise ValueError(
            f"dataset version {dataset_version_id} is already published; "
            "run the complete pipeline to create a new candidate version"
        )


def build_dataset_gold(
    silver_session: Session, gold_session: Session, dataset_version_id: str
) -> dict[str, int]:
    """Build article/chunk rows for a candidate without touching published tables."""
    _assert_mutable_candidate(gold_session, dataset_version_id)
    silver_rows = silver_session.scalars(select(SilverArticle)).all()
    by_doi = {row.doi: row for row in silver_rows}
    merge_plan = active_merge_plan(gold_session, set(by_doi))
    merged_duplicates = {doi for rows in merge_plan.groups.values() for doi in rows}

    state = _state(gold_session)
    active_version = state.active_version_id
    stored: dict[tuple[str, str, int], DatasetChunk | Chunk] = {}
    if active_version:
        active_chunks = gold_session.scalars(
            select(DatasetChunk).where(DatasetChunk.dataset_version_id == active_version)
        ).all()
        stored = {(row.doi, row.chunk_type, row.seq): row for row in active_chunks}
    elif gold_session.scalar(select(func.count()).select_from(Chunk)):
        live_chunks = gold_session.scalars(select(Chunk)).all()
        stored = {(row.doi, row.chunk_type, row.seq): row for row in live_chunks}

    gold_session.execute(
        delete(DatasetDuplicatePair).where(
            DatasetDuplicatePair.dataset_version_id == dataset_version_id
        )
    )
    gold_session.execute(
        delete(DatasetSemantics).where(DatasetSemantics.dataset_version_id == dataset_version_id)
    )
    gold_session.execute(
        delete(DatasetChunk).where(DatasetChunk.dataset_version_id == dataset_version_id)
    )
    gold_session.execute(
        delete(DatasetArticle).where(DatasetArticle.dataset_version_id == dataset_version_id)
    )

    chunk_counts: Counter[str] = Counter()
    reused = new = changed = 0
    for silver_row in silver_rows:
        if silver_row.doi in merged_duplicates:
            continue
        duplicate_rows = [by_doi[doi] for doi in merge_plan.groups.get(silver_row.doi, ())]
        row = _merge_silver_group(silver_row, duplicate_rows) if duplicate_rows else silver_row
        gold_session.add(
            DatasetArticle(
                dataset_version_id=dataset_version_id,
                doi=row.doi,
                sources=row.sources or [],
                title=row.title,
                authors=row.authors or [],
                year=row.year,
                venue=row.venue,
                keywords=row.keywords or [],
                abstract=row.abstract,
                citation_count=row.citation_count,
                reference_count=row.reference_count,
                countries=row.countries or [],
                online_date=row.online_date,
                document_type=row.document_type,
                license=row.license,
                url=row.url,
                has_pdf=row.has_pdf,
                pdf_path=row.pdf_path,
                silver_id=row.id,
                is_non_article=getattr(row, "is_non_article", False),
            )
        )
        for desired in _desired_chunks(row):
            key = (row.doi, desired["chunk_type"], desired["seq"])
            previous = stored.get(key)
            same_text = previous is not None and previous.text == desired["text"]
            if same_text:
                reused += 1
            elif previous is None:
                new += 1
            else:
                changed += 1
            gold_session.add(
                DatasetChunk(
                    dataset_version_id=dataset_version_id,
                    doi=row.doi,
                    seq=desired["seq"],
                    chunk_type=desired["chunk_type"],
                    text=desired["text"],
                    char_len=len(desired["text"]),
                    embedding=previous.embedding if same_text else None,
                    embedding_bin=previous.embedding_bin if same_text else None,
                    embed_model=previous.embed_model if same_text else None,
                )
            )
            chunk_counts[desired["chunk_type"]] += 1
    gold_session.flush()
    return {
        "articles": len(silver_rows) - len(merged_duplicates),
        "abstract_chunks": chunk_counts["abstract"],
        "fulltext_chunks": chunk_counts["fulltext"],
        "chunks_reused": reused,
        "chunks_new": new,
        "chunks_invalidated": changed,
        "duplicate_records_merged": len(merged_duplicates),
        "duplicate_overrides_inactive": merge_plan.inactive_overrides,
    }


def build_dataset_embeddings(gold_session: Session, dataset_version_id: str) -> dict[str, int]:
    _assert_mutable_candidate(gold_session, dataset_version_id)
    rows = gold_session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == dataset_version_id)
        .order_by(DatasetChunk.id)
    ).all()
    pending = [
        row for row in rows if row.embedding_bin is None or row.embed_model != EMBED_MODEL_NAME
    ]
    if not pending:
        return {"embedded": 0, "already_embedded": len(rows), "total_chunks": len(rows)}

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        vectors = model.embed([row.text for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            array = np.asarray(vector, dtype=np.float32)
            row.embedding = array.tolist()
            row.embedding_bin = array.tobytes()
            row.embed_model = EMBED_MODEL_NAME
        gold_session.flush()
    return {
        "embedded": len(pending),
        "already_embedded": len(rows) - len(pending),
        "total_chunks": len(rows),
    }


def build_dataset_semantics(
    gold_session: Session,
    dataset_version_id: str,
    *,
    anchor_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, int | float]:
    _assert_mutable_candidate(gold_session, dataset_version_id)
    chunks = gold_session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == dataset_version_id)
        .where(DatasetChunk.chunk_type == "abstract")
        .order_by(DatasetChunk.doi)
    ).all()
    if not chunks:
        raise ValueError("semantic stage requires at least one abstract chunk")
    incomplete = [row.doi for row in chunks if row.embedding_bin is None]
    if incomplete:
        raise ValueError(
            f"semantic stage requires complete abstract embeddings ({len(incomplete)} missing)"
        )

    matrix = np.array(
        [np.frombuffer(row.embedding_bin, dtype=np.float32) for row in chunks], dtype="float32"
    )
    dois = [row.doi for row in chunks]
    texts = [row.text for row in chunks]
    anchors = (
        np.array(anchor_vectors, dtype="float32")
        if anchor_vectors is not None
        else _embed_anchors([ANCHOR_TEXT, OFF_ANCHOR_TEXT])
    )
    scores = relevance_scores(matrix, anchors[0])
    offtopic = relevance_scores(matrix, anchors[1])
    reduced = reduced_space(matrix)
    labels, theme_labels = discover_themes(reduced, texts)
    coords = project_2d(reduced)
    pairs = near_duplicate_pairs(matrix, dois)
    reviewed = {
        (row.doi_a, row.doi_b) for row in gold_session.scalars(select(DuplicateOverride)).all()
    }
    pairs = unresolved_duplicate_pairs(pairs, reviewed)

    gold_session.execute(
        delete(DatasetSemantics).where(DatasetSemantics.dataset_version_id == dataset_version_id)
    )
    gold_session.execute(
        delete(DatasetDuplicatePair).where(
            DatasetDuplicatePair.dataset_version_id == dataset_version_id
        )
    )
    gold_session.add_all(
        [
            DatasetSemantics(
                dataset_version_id=dataset_version_id,
                doi=doi,
                relevance_score=float(scores[index]),
                offtopic_score=float(offtopic[index]),
                theme_id=int(labels[index]),
                theme_label=theme_labels[int(labels[index])],
                map_x=float(coords[index][0]),
                map_y=float(coords[index][1]),
                embed_model=EMBED_MODEL_NAME,
            )
            for index, doi in enumerate(dois)
        ]
    )
    gold_session.add_all(
        [
            DatasetDuplicatePair(
                dataset_version_id=dataset_version_id,
                doi_a=min(doi_a, doi_b),
                doi_b=max(doi_a, doi_b),
                similarity=similarity,
            )
            for doi_a, doi_b, similarity in pairs
        ]
    )
    gold_session.flush()
    return {
        "articles": len(dois),
        "themes": len(theme_labels),
        "duplicate_pairs": len(pairs),
        "embedding_coverage": 1.0,
        "median_relevance": round(float(np.median(scores)), 4),
        "median_margin": round(float(np.median(scores - offtopic)), 4),
        "offtopic_articles": int((scores - offtopic < 0).sum()),
    }


def _copy_live_to_dataset(session: Session, version_id: str) -> None:
    for row in session.scalars(select(Article)).all():
        session.add(
            DatasetArticle(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Article.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(Chunk)).all():
        session.add(
            DatasetChunk(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Chunk.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(Semantics)).all():
        session.add(
            DatasetSemantics(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Semantics.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(DuplicatePair)).all():
        session.add(
            DatasetDuplicatePair(
                dataset_version_id=version_id,
                doi_a=min(row.doi_a, row.doi_b),
                doi_b=max(row.doi_a, row.doi_b),
                similarity=row.similarity,
                created_at=row.created_at,
            )
        )


def adopt_legacy_publication(session: Session) -> str | None:
    state = _state(session)
    if state.active_version_id or not session.scalar(select(func.count()).select_from(Article)):
        return state.active_version_id
    payload = {
        "articles": [row[0] for row in session.execute(select(Article.doi).order_by(Article.doi))],
        "chunks": [
            tuple(row)
            for row in session.execute(
                select(Chunk.doi, Chunk.chunk_type, Chunk.seq, Chunk.text).order_by(
                    Chunk.doi, Chunk.chunk_type, Chunk.seq
                )
            )
        ],
    }
    version_id = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if session.get(DatasetVersion, version_id) is None:
        session.add(
            DatasetVersion(
                version_id=version_id,
                status="active",
                stats={"kind": "legacy_import", "provenance_complete": False},
            )
        )
        _copy_live_to_dataset(session, version_id)
    state.active_version_id = version_id
    session.flush()
    return version_id


def materialize_version(session: Session, version_id: str, execution_id: str) -> None:
    """Atomically replace live Gold projections with a validated snapshot."""
    version = session.get(DatasetVersion, version_id)
    if version is None:
        raise ValueError(f"unknown dataset version {version_id}")
    articles = session.scalars(
        select(DatasetArticle).where(DatasetArticle.dataset_version_id == version_id)
    ).all()
    if not articles:
        raise ValueError(f"dataset version {version_id} has no Gold article snapshot")

    state = _state(session)
    if state.active_version_id is None:
        adopt_legacy_publication(session)
    previous_id = state.active_version_id

    session.execute(delete(Semantics))
    session.execute(delete(DuplicatePair))
    session.execute(delete(Chunk))
    session.execute(delete(Article))
    session.add_all(
        [
            Article(
                **{
                    column.name: getattr(row, column.name)
                    for column in Article.__table__.columns
                    if column.name != "id"
                }
            )
            for row in articles
        ]
    )
    chunks = session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            Chunk(
                **{
                    column.name: getattr(row, column.name)
                    for column in Chunk.__table__.columns
                    if column.name != "id"
                }
            )
            for row in chunks
        ]
    )
    semantic_rows = session.scalars(
        select(DatasetSemantics).where(DatasetSemantics.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            Semantics(
                **{
                    column.name: getattr(row, column.name)
                    for column in Semantics.__table__.columns
                    if column.name != "id"
                }
            )
            for row in semantic_rows
        ]
    )
    pair_rows = session.scalars(
        select(DatasetDuplicatePair).where(DatasetDuplicatePair.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            DuplicatePair(
                doi_a=row.doi_a,
                doi_b=row.doi_b,
                similarity=row.similarity,
                created_at=row.created_at,
            )
            for row in pair_rows
        ]
    )
    if previous_id and previous_id != version_id:
        previous = session.get(DatasetVersion, previous_id)
        if previous is not None:
            previous.status = "superseded"
    version.status = "active"
    from lake_research_map.db.time import naive_utc_now

    version.published_at = naive_utc_now()
    version.failed_at = None
    version.failure_reason = None
    state.active_version_id = version_id
    state.working_version_id = version_id
    state.execution_id = execution_id
    session.flush()
