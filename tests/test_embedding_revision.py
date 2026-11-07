"""The embedding-model pin: by snapshot, or by byte-identical model files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from transform import embeddings
from transform.embeddings import (
    EMBED_MODEL_REVISION,
    MODEL_DEFINING_FILES,
    model_content_digest,
    resolved_model_revision,
)


def _snapshot(root, name: str, *, weights: bytes = b"weights"):
    directory = root / name
    directory.mkdir()
    for file_name in MODEL_DEFINING_FILES:
        payload = weights if file_name == "model_optimized.onnx" else file_name.encode()
        (directory / file_name).write_bytes(payload)
    (directory / "README.md").write_text(f"readme of {name}")
    return SimpleNamespace(model=SimpleNamespace(_model_dir=directory))


def test_the_pinned_snapshot_is_accepted(tmp_path):
    assert resolved_model_revision(_snapshot(tmp_path, EMBED_MODEL_REVISION)) == (
        EMBED_MODEL_REVISION
    )


def test_a_newer_snapshot_with_identical_model_files_resolves_to_the_pin(tmp_path, monkeypatch):
    pinned = _snapshot(tmp_path, EMBED_MODEL_REVISION)
    monkeypatch.setattr(
        embeddings, "EMBED_MODEL_CONTENT_SHA256", model_content_digest(pinned.model._model_dir)
    )
    readme_only_commit = _snapshot(tmp_path, "aa8f8b060edb00e03bfdd08813a2949946c8ba55")

    assert resolved_model_revision(readme_only_commit) == EMBED_MODEL_REVISION


def test_a_snapshot_with_different_weights_is_refused(tmp_path, monkeypatch):
    pinned = _snapshot(tmp_path, EMBED_MODEL_REVISION)
    monkeypatch.setattr(
        embeddings, "EMBED_MODEL_CONTENT_SHA256", model_content_digest(pinned.model._model_dir)
    )
    retrained = _snapshot(tmp_path, "f" * 40, weights=b"other weights")

    with pytest.raises(RuntimeError, match="revision mismatch"):
        resolved_model_revision(retrained)


def test_a_snapshot_missing_a_model_file_has_no_digest(tmp_path):
    model = _snapshot(tmp_path, "partial")
    (model.model._model_dir / "tokenizer.json").unlink()

    assert model_content_digest(model.model._model_dir) is None
