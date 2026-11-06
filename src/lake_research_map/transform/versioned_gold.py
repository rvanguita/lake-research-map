"""Build, validate, publish, and reactivate immutable Gold snapshots.

Chunk text, metadata and embedding provenance live in the `lit_dataset_*`
tables; the vectors themselves live in one Qdrant collection per version
(ADR-08, `db/vector_store.py`). The publication pointer in Gold stays the only
thing a publication commits -- the collection of a version is complete and
contract-checked before the pointer can move to it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

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
    PipelineExecution,
    PublicationState,
    SemanticRun,
    Semantics,
)
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.db.vector_store import (
    ACTIVE_ALIAS,
    QdrantVectorStore,
    VectorPoint,
    collection_name,
    point_id,
)
from lake_research_map.transform.duplicate_resolution import active_merge_plan
from lake_research_map.transform.embeddings import (
    EMBED_BATCH_SIZE,
    EMBED_DIMENSION,
    EMBED_MODEL_NAME,
    EMBED_MODEL_REVISION,
    embedding_model,
)
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
from lake_research_map.transform.vector_payloads import chunk_payload, semantic_payload

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], Iterable]


def _state(session: Session) -> PublicationState:
    state = session.get(PublicationState, 1)
    if state is None:
        state = PublicationState(id=1)
        session.add(state)
        session.flush()
    return state


def active_version_id(session: Session) -> str | None:
    """The published version, read without creating the state row `_state` adds."""
    state = session.get(PublicationState, 1)
    return state.active_version_id if state is not None else None


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
    # `stored` only classifies each chunk as reused / new / changed text for the
    # run stats. Vectors are no longer carried here: they live in Qdrant, and the
    # embed stage reuses any stored vector with the same text hash and model
    # revision -- from this candidate's own collection after a killed run, or
    # from the active version's -- so a rebuild still never re-embeds unchanged
    # text.
    candidate_chunks = gold_session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == dataset_version_id)
    ).all()
    stored.update({(row.doi, row.chunk_type, row.seq): row for row in candidate_chunks})
    if active_version:
        active_chunks = gold_session.scalars(
            select(DatasetChunk).where(DatasetChunk.dataset_version_id == active_version)
        ).all()
        for row in active_chunks:
            stored.setdefault((row.doi, row.chunk_type, row.seq), row)
    elif gold_session.scalar(select(func.count()).select_from(Chunk)):
        # Fill gaps only. Replacing `stored` here threw away vectors a killed
        # first run had already committed to the candidate -- the exact case
        # the comment above promises to protect -- because the legacy rows
        # carry no revision and so send every chunk back to be re-embedded.
        for row in gold_session.scalars(select(Chunk)).all():
            stored.setdefault((row.doi, row.chunk_type, row.seq), row)

    gold_session.execute(
        delete(DatasetDuplicatePair).where(
            DatasetDuplicatePair.dataset_version_id == dataset_version_id
        )
    )
    gold_session.execute(
        delete(SemanticRun).where(SemanticRun.dataset_version_id == dataset_version_id)
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
                publication_category=row.publication_category,
                publication_category_basis=row.publication_category_basis,
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
                    text_sha256=hashlib.sha256(desired["text"].encode("utf-8")).hexdigest(),
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


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_embed_fn() -> EmbedFn:
    model = embedding_model()
    return lambda texts: model.embed(texts)


def _legacy_vectors(gold_session: Session, version_id: str) -> dict[str, np.ndarray]:
    """Reusable vectors a version published before ADR-08 still holds as MySQL BLOBs.

    One read per version, through the `dataset_version_id` index. Looking the
    BLOBs up by `text_sha256` instead scans the whole table on the deployed
    schema -- that column arrived through an additive migration, which adds no
    index -- and cost 7 s per 64 hashes when measured on 2026-11-06. Only rows
    whose provenance names the pinned model revision qualify, the rule the embed
    contract applies, so a BLOB of unknown origin is never promoted into Qdrant.
    """
    rows = gold_session.execute(
        select(DatasetChunk.text_sha256, DatasetChunk.embedding_bin)
        .where(DatasetChunk.dataset_version_id == version_id)
        .where(DatasetChunk.embedding_bin.is_not(None))
        .where(DatasetChunk.text_sha256.is_not(None))
        .where(DatasetChunk.embed_model == EMBED_MODEL_NAME)
        .where(DatasetChunk.embed_revision == EMBED_MODEL_REVISION)
    ).all()
    found: dict[str, np.ndarray] = {}
    for digest, blob in rows:
        if digest not in found and len(blob) == EMBED_DIMENSION * 4:
            found[digest] = np.frombuffer(blob, dtype=np.float32).copy()
    return found


def _reusable_vectors(
    gold_session: Session,
    store: QdrantVectorStore,
    wanted: set[str],
    *,
    collections: list[str],
    legacy_versions: list[str],
) -> dict[str, np.ndarray]:
    """Resolve every reusable vector up front, before any batch runs.

    A vector is a function of (text, model, revision), so any stored point or
    qualifying BLOB with the same text hash holds what the model would return.
    Resolving them in bulk -- a few hundred hashes per Qdrant request, one
    MySQL read per legacy version -- keeps the batch loop down to the upsert,
    the Gold commit and the texts nothing could supply.
    """
    found: dict[str, np.ndarray] = {}
    for version in collections:
        missing = wanted - found.keys()
        if not missing:
            return found
        found.update(
            store.find_by_text_hash(
                version, sorted(missing), model=EMBED_MODEL_NAME, revision=EMBED_MODEL_REVISION
            )
        )
    for version in legacy_versions:
        missing = wanted - found.keys()
        if not missing:
            return found
        legacy = _legacy_vectors(gold_session, version)
        found.update({digest: legacy[digest] for digest in missing if digest in legacy})
    return found


def fill_vector_collection(
    gold_session: Session,
    store: QdrantVectorStore,
    dataset_version_id: str,
    *,
    embed_fn: EmbedFn | None = None,
    allow_compute: bool = True,
    write_provenance: bool = False,
    execution_id: str | None = None,
) -> dict[str, int | str]:
    """Make the version's collection hold exactly one valid point per Gold chunk.

    A chunk is pending when its point is missing or carries another text hash
    or model revision (and, with `write_provenance`, when its Gold row lacks the
    provenance). A pending chunk's vector is copied from any point with the
    same text hash and revision -- this collection first, so a killed run
    resumes, then the active version's -- or from a pre-ADR-08 BLOB, and only
    text no store has seen goes to the model. Each batch is upserted (and
    waited for) before Gold records its provenance and commits: a run killed
    between the two re-upserts the same deterministic ids on retry instead of
    losing or duplicating work. Points whose chunk no longer exists are
    deleted, so the collection never answers with a chunk the version lacks.

    `write_provenance=False` is the repair path (`vectors backfill`/`rebuild`):
    it only writes Qdrant, never the Gold rows of a published snapshot.
    """
    rows = gold_session.execute(
        select(
            DatasetChunk.id,
            DatasetChunk.doi,
            DatasetChunk.chunk_type,
            DatasetChunk.seq,
            DatasetChunk.text_sha256,
            DatasetChunk.embed_model,
            DatasetChunk.embed_revision,
            DatasetChunk.embedded_at,
        )
        .where(DatasetChunk.dataset_version_id == dataset_version_id)
        .order_by(DatasetChunk.id)
    ).all()
    total_chunks = len(rows)
    stats: dict[str, int | str] = {
        "embedded": 0,
        "vectors_copied": 0,
        "already_embedded": 0,
        "unresolved": 0,
        "orphans_removed": 0,
        "total_chunks": total_chunks,
        "collection": collection_name(dataset_version_id),
    }
    if not rows:
        return stats
    store.ensure_collection(dataset_version_id, EMBED_DIMENSION)

    ids = {row.id: point_id(row.doi, row.chunk_type, row.seq) for row in rows}
    present = store.fetch(
        dataset_version_id,
        list(ids.values()),
        payload_fields=["text_sha256", "embed_model", "embed_revision"],
    )

    def settled(row) -> bool:
        point = present.get(ids[row.id])
        if (
            point is None
            or row.text_sha256 is None
            or point.payload.get("text_sha256") != row.text_sha256
            or point.payload.get("embed_model") != EMBED_MODEL_NAME
            or point.payload.get("embed_revision") != EMBED_MODEL_REVISION
        ):
            return False
        if not write_provenance:
            return True
        return (
            row.embed_model == EMBED_MODEL_NAME
            and row.embed_revision == EMBED_MODEL_REVISION
            and row.embedded_at is not None
        )

    pending_ids = [row.id for row in rows if not settled(row)]
    stats["already_embedded"] = total_chunks - len(pending_ids)

    expected = set(ids.values())
    stored_ids = {point.id for point in store.scroll(dataset_version_id, payload_fields=False)}
    orphans = sorted(stored_ids - expected)
    if orphans:
        store.delete_points(dataset_version_id, orphans)
        stats["orphans_removed"] = len(orphans)
    if not pending_ids:
        return stats

    # Plain result rows, not ORM instances: each batch commits and expunges the
    # session, which would detach ORM objects loaded once up here.
    articles = {
        row.doi: row
        for row in gold_session.execute(
            select(
                DatasetArticle.doi,
                DatasetArticle.sources,
                DatasetArticle.title,
                DatasetArticle.year,
                DatasetArticle.venue,
                DatasetArticle.publication_category,
                DatasetArticle.is_non_article,
                DatasetArticle.has_pdf,
                DatasetArticle.abstract,
                DatasetArticle.url,
            ).where(DatasetArticle.dataset_version_id == dataset_version_id)
        ).all()
    }
    semantic_rows = {
        row.doi: row
        for row in gold_session.execute(
            select(
                DatasetSemantics.doi,
                DatasetSemantics.relevance_score,
                DatasetSemantics.offtopic_score,
                DatasetSemantics.theme_id,
                DatasetSemantics.theme_label,
            ).where(DatasetSemantics.dataset_version_id == dataset_version_id)
        ).all()
    }
    active = active_version_id(gold_session)
    reuse_from = [dataset_version_id]
    if active and active != dataset_version_id and store.has_collection(active):
        reuse_from.append(active)
    pending = set(pending_ids)
    reusable = _reusable_vectors(
        gold_session,
        store,
        {row.text_sha256 for row in rows if row.id in pending and row.text_sha256},
        collections=reuse_from,
        legacy_versions=list(dict.fromkeys(v for v in (dataset_version_id, active) if v)),
    )
    embed = embed_fn
    done = 0
    for start in range(0, len(pending_ids), EMBED_BATCH_SIZE):
        batch_ids = pending_ids[start : start + EMBED_BATCH_SIZE]
        by_id = {
            row.id: row
            for row in gold_session.scalars(
                select(DatasetChunk).where(DatasetChunk.id.in_(batch_ids))
            ).all()
        }
        batch = [by_id[row_id] for row_id in batch_ids if row_id in by_id]
        # The hash of the text itself, not the stored one: a stale stored hash
        # must miss the reuse map and be re-embedded, never reuse a wrong vector.
        digests = {row.id: _text_sha256(row.text) for row in batch}
        vectors = {digest: reusable[digest] for digest in digests.values() if digest in reusable}
        copied = set(vectors)
        to_embed = [row for row in batch if digests[row.id] not in vectors]
        if to_embed and allow_compute:
            if embed is None:
                embed = _default_embed_fn()
            # Identical texts within a batch are embedded once.
            unique = list(dict.fromkeys(digests[row.id] for row in to_embed))
            text_by_digest = {digests[row.id]: row.text for row in to_embed}
            computed = embed([text_by_digest[digest] for digest in unique])
            for digest, vector in zip(unique, computed, strict=True):
                vectors[digest] = np.asarray(vector, dtype=np.float32)
            stats["embedded"] += len(to_embed)
        stats["vectors_copied"] += sum(1 for row in batch if digests[row.id] in copied)

        now = datetime.now(UTC).replace(tzinfo=None)
        points = []
        for row in batch:
            vector = vectors.get(digests[row.id])
            if vector is None:
                stats["unresolved"] += 1
                continue
            points.append(
                VectorPoint(
                    id=ids[row.id],
                    vector=vector,
                    payload=chunk_payload(
                        version_id=dataset_version_id,
                        doi=row.doi,
                        chunk_type=row.chunk_type,
                        seq=row.seq,
                        text=row.text,
                        text_sha256=digests[row.id],
                        embedded_at=(row.embedded_at or now) if not write_provenance else now,
                        article=articles.get(row.doi),
                        semantics=semantic_rows.get(row.doi),
                    ),
                )
            )
            if write_provenance:
                row.text_sha256 = digests[row.id]
                row.embed_model = EMBED_MODEL_NAME
                row.embed_revision = EMBED_MODEL_REVISION
                row.embedding_dim = int(vector.size)
                row.embedding_dtype = str(vector.dtype)
                row.embedding_normalized = bool(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3))
                row.embedded_at = now
        store.upsert(dataset_version_id, points)
        heartbeat_execution_id = execution_id or gold_session.info.get("execution_id")
        if heartbeat_execution_id:
            execution = gold_session.get(PipelineExecution, heartbeat_execution_id)
            if execution is not None:
                execution.heartbeat_at = datetime.now(UTC).replace(tzinfo=None)
        gold_session.flush()
        gold_session.commit()
        gold_session.expunge_all()
        done += len(batch)
        logger.info("vector progress: %d/%d chunks", done, len(pending_ids))
    return stats


def build_dataset_embeddings(
    gold_session: Session,
    dataset_version_id: str,
    *,
    store: QdrantVectorStore,
    embed_fn: EmbedFn | None = None,
    execution_id: str | None = None,
) -> dict[str, int | str]:
    """Embed a candidate incrementally into its Qdrant collection.

    Only one batch of ORM objects and vectors is kept in memory, and each
    batch is committed before the next starts, so a rerun skips work that
    already reached the store even if the original process was terminated.
    """
    _assert_mutable_candidate(gold_session, dataset_version_id)
    return fill_vector_collection(
        gold_session,
        store,
        dataset_version_id,
        embed_fn=embed_fn,
        write_provenance=True,
        execution_id=execution_id,
    )


def build_dataset_semantics(
    gold_session: Session,
    dataset_version_id: str,
    *,
    store: QdrantVectorStore,
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
    keys = [point_id(row.doi, row.chunk_type, row.seq) for row in chunks]
    points = (
        store.fetch(dataset_version_id, keys, with_vectors=True, payload_fields=False)
        if store.has_collection(dataset_version_id)
        else {}
    )
    incomplete = [
        row.doi
        for row, key in zip(chunks, keys, strict=True)
        if key not in points or points[key].vector is None
    ]
    if incomplete:
        raise ValueError(
            f"semantic stage requires complete abstract embeddings ({len(incomplete)} missing)"
        )

    matrix = np.array([points[key].vector for key in keys], dtype="float32")
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
    from lake_research_map.dashboard.analytics import semantic_stability_diagnostics
    from lake_research_map.transform.semantics import theme_sweep

    stability = semantic_stability_diagnostics(reduced, labels, coords)
    if stability.get("valid"):
        stability["k_sweep"] = theme_sweep(reduced)
    pairs = near_duplicate_pairs(matrix, dois)
    reviewed = {
        (row.doi_a, row.doi_b) for row in gold_session.scalars(select(DuplicateOverride)).all()
    }
    pairs = unresolved_duplicate_pairs(pairs, reviewed)

    gold_session.execute(
        delete(DatasetSemantics).where(DatasetSemantics.dataset_version_id == dataset_version_id)
    )
    gold_session.execute(
        delete(SemanticRun).where(SemanticRun.dataset_version_id == dataset_version_id)
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
    gold_session.add(
        SemanticRun(
            run_id=uuid.uuid4().hex,
            dataset_version_id=dataset_version_id,
            embed_model=EMBED_MODEL_NAME,
            embed_revision=EMBED_MODEL_REVISION,
            embedding_dim=int(matrix.shape[1]),
            article_count=len(dois),
            parameters={
                "anchor_sha256": hashlib.sha256(ANCHOR_TEXT.encode("utf-8")).hexdigest(),
                "off_anchor_sha256": hashlib.sha256(OFF_ANCHOR_TEXT.encode("utf-8")).hexdigest(),
            },
            stability=stability,
        )
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
    _write_semantic_payloads(gold_session, store, dataset_version_id)
    return {
        "articles": len(dois),
        "themes": len(theme_labels),
        "duplicate_pairs": len(pairs),
        "embedding_coverage": 1.0,
        "median_relevance": round(float(np.median(scores)), 4),
        "median_margin": round(float(np.median(scores - offtopic)), 4),
        "offtopic_articles": int((scores - offtopic < 0).sum()),
    }


def _write_semantic_payloads(
    gold_session: Session, store: QdrantVectorStore, dataset_version_id: str
) -> None:
    """Copy each article's screening signals onto every point of that article.

    Fulltext points carry them too, so a client filtering by `relevance_margin`
    gets the same articles whichever chunk type it searches.
    """
    signals = {
        row.doi: semantic_payload(row)
        for row in gold_session.scalars(
            select(DatasetSemantics).where(
                DatasetSemantics.dataset_version_id == dataset_version_id
            )
        ).all()
    }
    points_by_doi: dict[str, list[str]] = defaultdict(list)
    for doi, chunk_type, seq in gold_session.execute(
        select(DatasetChunk.doi, DatasetChunk.chunk_type, DatasetChunk.seq).where(
            DatasetChunk.dataset_version_id == dataset_version_id
        )
    ):
        points_by_doi[doi].append(point_id(doi, chunk_type, seq))
    store.set_payloads(
        dataset_version_id,
        [(points_by_doi[doi], payload) for doi, payload in signals.items() if doi in points_by_doi],
    )


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


def materialize_version(
    session: Session, version_id: str, execution_id: str, *, store: QdrantVectorStore
) -> None:
    """Atomically replace live Gold projections with a validated snapshot.

    The version's Qdrant collection is part of the publication contract, so a
    pointer can never move to a version whose vectors are missing or stale.
    Moving the external alias is left to the caller, after its commit
    (`publish_vector_alias`): the alias must never point at a version Gold has
    not published.
    """
    from lake_research_map.quality import assert_contract, publication_contract

    version = session.get(DatasetVersion, version_id)
    if version is None:
        raise ValueError(f"unknown dataset version {version_id}")
    assert_contract("publication", publication_contract(session, version_id, store))
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


def publish_vector_alias(store: QdrantVectorStore, version_id: str) -> bool:
    """Point the external alias at a version Gold has just published.

    Called after the publication commit. A failure is logged, never raised:
    the publication already happened, internal readers resolve collections
    through the Gold pointer rather than the alias, and `vectors sync-alias`
    repairs it.
    """
    try:
        store.point_alias(version_id)
    except Exception:
        logger.exception(
            "alias %s was not moved to version %s; run `lake-research-map vectors sync-alias`",
            ACTIVE_ALIAS,
            version_id,
        )
        return False
    return True
