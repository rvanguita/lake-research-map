"""Models the retrieval package loads on demand.

- The cross-encoder reranker. Optional and experimental: PRD §8.5 forbids
  making any retrieval mode the default without human relevance labels, and the
  WP-13 labels are not in yet. It is pinned exactly like the embedder -- by
  snapshot, or by byte-identical model files -- so a Hub commit that only
  touches a README cannot silently swap it.
- An untruncating tokenizer, to measure what the embedder cuts off. fastembed's
  own tokenizer truncates at the model window before it counts, so it cannot
  say whether a text was truncated.

Everything is imported lazily and loaded once per process.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from transform.embeddings import (
    EMBED_THREADS,
    embedding_model,
    model_content_digest,
)

RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_REVISION = os.environ.get(
    "LAKE_RESEARCH_MAP_RERANK_REVISION",
    "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
)
RERANK_MODEL_FILES = (
    "config.json",
    "onnx/model.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
RERANK_CONTENT_SHA256 = os.environ.get(
    "LAKE_RESEARCH_MAP_RERANK_CONTENT_SHA256",
    "296cf7c9b9222b9d2911be1e77b1855be1d4ff9f786e7fc651df02756de27c86",
)


def resolved_rerank_revision(model) -> str:
    """Return and enforce the reranker snapshot, by commit or by model-file content."""
    model_dir = getattr(getattr(model, "model", None), "_model_dir", None)
    revision = getattr(model_dir, "name", None)
    if revision == RERANK_MODEL_REVISION:
        return revision
    if (
        model_dir is not None
        and model_content_digest(model_dir, RERANK_MODEL_FILES) == RERANK_CONTENT_SHA256
    ):
        return RERANK_MODEL_REVISION
    raise RuntimeError(
        "reranker model revision mismatch: "
        f"expected {RERANK_MODEL_REVISION} (or a snapshot with identical model files), "
        f"loaded {revision or 'unknown'}"
    )


@cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = TextCrossEncoder(model_name=RERANK_MODEL_NAME, threads=EMBED_THREADS)
    resolved_rerank_revision(model)
    return model


def rerank_scores(query: str, texts: Sequence[str]) -> list[float]:
    """Cross-encoder relevance logits, one per text; higher is more relevant."""
    if not texts:
        return []
    return [float(score) for score in reranker().rerank(query, list(texts))]


def embed_query(text: str):
    """The query vector, from the same pinned model that embedded the corpus."""
    return next(iter(embedding_model().embed([text])))


def _embedder_dir() -> Path:
    return Path(embedding_model().model._model_dir)


@cache
def _untruncated_tokenizer():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(_embedder_dir() / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


@cache
def context_window_tokens() -> int:
    """The embedder's input window, special tokens included (512 for BGE-small)."""
    config = json.loads((_embedder_dir() / "tokenizer_config.json").read_text())
    return int(config.get("model_max_length", 512))


def token_lengths(texts: Sequence[str]) -> list[int]:
    """Token counts with special tokens and without truncation."""
    if not texts:
        return []
    return [len(encoding.ids) for encoding in _untruncated_tokenizer().encode_batch(list(texts))]
