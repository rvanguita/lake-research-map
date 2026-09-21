"""Fill in lit_gold.chunks.embedding -- the step gold_articles.py deliberately
left for later.

Runs entirely locally via `fastembed` (ONNX runtime, no PyTorch) so this
stage has no API key and no GPU requirement. Idempotent: only chunks with
`embedding IS NULL` are processed, so re-running after a fresh `--stage gold`
picks up just the new chunks, and running it twice in a row is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import Chunk

# BAAI/bge-small-en-v1.5: 384-dim, ~512-token context. fastembed truncates
# longer inputs automatically -- consistent with the 1500-char chunking cap
# in gold_articles.py, which already keeps chunks close to that context size.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 256

ProgressCallback = Callable[[int, int], None]


def build_embeddings(gold_session: Session, on_progress: ProgressCallback | None = None) -> dict:
    """Embed every chunk that doesn't have a vector yet.

    `on_progress(processed, total)` is called after each batch commit -- used
    by the dashboard's progress bar; the CLI/Airflow path just leaves it None.
    """
    total_chunks = gold_session.query(Chunk).count()

    pending_ids = gold_session.scalars(select(Chunk.id).where(Chunk.embedding.is_(None))).all()
    already_embedded = total_chunks - len(pending_ids)

    if not pending_ids:
        return {"embedded": 0, "already_embedded": already_embedded, "total_chunks": total_chunks}

    # Imported here (not at module load) so importing this module -- e.g. from
    # pipeline.py's top-level imports -- doesn't pay fastembed's own import
    # cost or trigger a model-file check for stages that never call this.
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME)

    embedded = 0
    total_pending = len(pending_ids)
    for start in range(0, total_pending, EMBED_BATCH_SIZE):
        batch_ids = pending_ids[start : start + EMBED_BATCH_SIZE]
        chunks = gold_session.scalars(select(Chunk).where(Chunk.id.in_(batch_ids))).all()
        # Preserve batch_ids order so texts and chunks line up 1:1 for zip below.
        chunks_by_id = {c.id: c for c in chunks}
        ordered_chunks = [chunks_by_id[i] for i in batch_ids if i in chunks_by_id]
        texts = [c.text for c in ordered_chunks]

        vectors = model.embed(texts)
        for chunk, vector in zip(ordered_chunks, vectors, strict=True):
            chunk.embedding = vector.tolist()
            chunk.embedding_bin = np.array(vector, dtype=np.float32).tobytes()
            chunk.embed_model = EMBED_MODEL_NAME
        gold_session.flush()

        embedded += len(ordered_chunks)
        if on_progress:
            on_progress(embedded, total_pending)

    return {
        "embedded": embedded,
        "already_embedded": already_embedded,
        "total_chunks": total_chunks,
    }
