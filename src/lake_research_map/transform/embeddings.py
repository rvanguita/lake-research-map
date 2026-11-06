"""Embedding model contract shared by the pipeline and its readers.

Embedding runs entirely locally via `fastembed` (ONNX runtime, no PyTorch), so
the stage needs no API key and no GPU. The work itself lives in
`transform/versioned_gold.py::fill_vector_collection`, which writes each
candidate version's vectors to its own Qdrant collection (ADR-08) and records
their provenance on the Gold chunk rows; this module holds the model identity,
the batch and thread bounds, the revision check those writers and `quality.py`
agree on, and the one place the model is loaded. A previous `build_embeddings`
wrote the *live* `lit_chunks` table directly from a dashboard button --
publication rebuilds that table from the candidate, so those vectors were
discarded on the next publish and never passed `embed_contract`. It was removed
rather than fixed: there is no correct way to mutate a published version in
place.
"""

from __future__ import annotations

import hashlib
import os
from functools import cache
from pathlib import Path

# BAAI/bge-small-en-v1.5: 384-dim, ~512-token context. fastembed truncates
# longer inputs automatically -- consistent with the 1500-char chunking cap
# in gold_articles.py, which already keeps chunks close to that context size.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIMENSION = 384
EMBED_MODEL_REVISION = os.environ.get(
    "LAKE_RESEARCH_MAP_EMBED_REVISION",
    "52398278842ec682c6f32300af41344b1c0b0bb2",
)
# The files whose bytes decide which vector a text gets. A Hub snapshot that
# changes anything else (README, .gitattributes) embeds identically.
MODEL_DEFINING_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# `model_content_digest` of the pinned snapshot's MODEL_DEFINING_FILES.
EMBED_MODEL_CONTENT_SHA256 = os.environ.get(
    "LAKE_RESEARCH_MAP_EMBED_CONTENT_SHA256",
    "9002f600edcf32abe11727ca9e91ffcc7fafee6cb6604679d96663b052c53988",
)
# Keep the default deliberately conservative: embedding is CPU and memory
# intensive, and a killed terminal must not lose an entire in-memory batch.
# Operators can increase these values for a larger machine.
EMBED_BATCH_SIZE = max(1, int(os.environ.get("LAKE_RESEARCH_MAP_EMBED_BATCH_SIZE", "64")))
EMBED_THREADS = max(1, int(os.environ.get("LAKE_RESEARCH_MAP_EMBED_THREADS", "2")))


def model_content_digest(model_dir) -> str | None:
    """SHA-256 over the model-defining files of a snapshot directory, or None."""
    root = Path(model_dir)
    digest = hashlib.sha256()
    for name in MODEL_DEFINING_FILES:
        path = root / name
        if not path.is_file():
            return None
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                file_hash.update(block)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.digest())
    return digest.hexdigest()


def resolved_model_revision(model) -> str:
    """Return and enforce the Hugging Face snapshot used by fastembed.

    The pin names the snapshot the stored vectors were produced with. Another
    snapshot is accepted only when its model-defining files are byte-identical
    to the pinned ones, and the pinned revision is returned so provenance keeps
    naming the vectors' real origin. That case is not hypothetical: the Hub
    repository received a README-only commit on 2026-09-24, and any fresh
    download resolves to it -- with fastembed's default cache in tmpfs `/tmp`,
    every reboot is a fresh download -- so a commit-hash-only check refused a
    model that embeds exactly as the pinned one does. A change to the weights
    or the tokenizer still fails here.
    """
    model_dir = getattr(getattr(model, "model", None), "_model_dir", None)
    revision = getattr(model_dir, "name", None)
    if revision == EMBED_MODEL_REVISION:
        return revision
    if model_dir is not None and model_content_digest(model_dir) == EMBED_MODEL_CONTENT_SHA256:
        return EMBED_MODEL_REVISION
    raise RuntimeError(
        "embedding model revision mismatch: "
        f"expected {EMBED_MODEL_REVISION} (or a snapshot with identical model files), "
        f"loaded {revision or 'unknown'}"
    )


@cache
def embedding_model():
    """The pinned embedding model, loaded once per process.

    The embed stage, the semantic anchors and every query embedding go through
    here, so all of them enforce the same revision: a query embedded by a
    different snapshot than the stored vectors would rank against vectors from
    a different space without any error.
    """
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME, threads=EMBED_THREADS)
    resolved_model_revision(model)
    return model
