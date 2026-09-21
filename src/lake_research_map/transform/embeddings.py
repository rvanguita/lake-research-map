"""Embedding model contract shared by the pipeline and its readers.

Embedding runs entirely locally via `fastembed` (ONNX runtime, no PyTorch), so
the stage needs no API key and no GPU. The work itself lives in
`transform/versioned_gold.py::build_dataset_embeddings`, which writes the
immutable candidate snapshot; this module holds the model identity, the batch
and thread bounds, and the revision check those writers and `quality.py` agree
on. A previous `build_embeddings` wrote the *live* `lit_chunks` table directly
from a dashboard button -- publication rebuilds that table from the candidate,
so those vectors were discarded on the next publish and never passed
`embed_contract`. It was removed rather than fixed: there is no correct way to
mutate a published version in place.
"""

from __future__ import annotations

import os

# BAAI/bge-small-en-v1.5: 384-dim, ~512-token context. fastembed truncates
# longer inputs automatically -- consistent with the 1500-char chunking cap
# in gold_articles.py, which already keeps chunks close to that context size.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_REVISION = os.environ.get(
    "LAKE_RESEARCH_MAP_EMBED_REVISION",
    "52398278842ec682c6f32300af41344b1c0b0bb2",
)
# Keep the default deliberately conservative: embedding is CPU and memory
# intensive, and a killed terminal must not lose an entire in-memory batch.
# Operators can increase these values for a larger machine.
EMBED_BATCH_SIZE = max(1, int(os.environ.get("LAKE_RESEARCH_MAP_EMBED_BATCH_SIZE", "64")))
EMBED_THREADS = max(1, int(os.environ.get("LAKE_RESEARCH_MAP_EMBED_THREADS", "2")))


def resolved_model_revision(model) -> str:
    """Return and enforce the Hugging Face snapshot used by fastembed."""
    model_dir = getattr(getattr(model, "model", None), "_model_dir", None)
    revision = getattr(model_dir, "name", None)
    if revision != EMBED_MODEL_REVISION:
        raise RuntimeError(
            "embedding model revision mismatch: "
            f"expected {EMBED_MODEL_REVISION}, loaded {revision or 'unknown'}"
        )
    return revision
