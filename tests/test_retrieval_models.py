"""The reranker pin: by snapshot, or by byte-identical model files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lake_research_map.retrieval import models
from lake_research_map.retrieval.models import (
    RERANK_MODEL_FILES,
    RERANK_MODEL_REVISION,
    resolved_rerank_revision,
)
from lake_research_map.transform.embeddings import model_content_digest


def _snapshot(root, name: str, *, weights: bytes = b"weights"):
    directory = root / name
    for file_name in RERANK_MODEL_FILES:
        path = directory / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(weights if file_name.endswith(".onnx") else file_name.encode())
    return SimpleNamespace(model=SimpleNamespace(_model_dir=directory))


def test_the_pinned_reranker_snapshot_is_accepted(tmp_path):
    assert resolved_rerank_revision(_snapshot(tmp_path, RERANK_MODEL_REVISION)) == (
        RERANK_MODEL_REVISION
    )


def test_a_snapshot_with_identical_reranker_files_resolves_to_the_pin(tmp_path, monkeypatch):
    pinned = _snapshot(tmp_path, RERANK_MODEL_REVISION)
    monkeypatch.setattr(
        models,
        "RERANK_CONTENT_SHA256",
        model_content_digest(pinned.model._model_dir, RERANK_MODEL_FILES),
    )

    assert resolved_rerank_revision(_snapshot(tmp_path, "b" * 40)) == RERANK_MODEL_REVISION
    with pytest.raises(RuntimeError, match="reranker model revision mismatch"):
        resolved_rerank_revision(_snapshot(tmp_path, "c" * 40, weights=b"retrained"))


def test_rerank_scores_of_nothing_is_nothing():
    # No model is loaded for an empty candidate list.
    assert models.rerank_scores("query", []) == []
    assert models.token_lengths([]) == []
