"""WP-21 / RQ-08: the benchmark surface, and what it refuses to claim.

The retrieval harness (now `apps/api/evaluation.py`) and the pooled sampler both existed and were reachable
only from `tests/`, so the project could compute Recall@k, MRR and nDCG and
had nowhere to show them. The panel exists now -- and the property that
matters is not the arithmetic, it is that a table scored against one
AI-assisted pass is labelled as not being evidence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from apps.api.api import BENCHMARK_MODES, RetrievalUnavailable
from apps.dashboard.pages import quality


def _judgements(reviewer: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "workflow": "retrieval",
                "subject_id": f"q01::10.1/{index}",
                "reviewer_id": reviewer,
                "label": "relevant",
            }
            for index in range(3)
        ]
    )


class _FakeRetriever:
    """Answers every mode with the same three articles; can refuse the reranker."""

    def __init__(self, rerank_available: bool = True) -> None:
        self.rerank_available = rerank_available

    def search(self, text, *, top_k=10, filters=None, rerank=False, **_settings):
        if rerank and not self.rerank_available:
            raise RetrievalUnavailable("the reranker is unavailable (offline)")
        return [SimpleNamespace(doi=f"10.1/{i}") for i in range(min(top_k, 3))]


_DEFAULT = object()


@pytest.fixture
def run(monkeypatch):
    """Render the panel, capturing what it wrote instead of drawing it.

    Injection points: `quality.loaders.review_labels` / `quality.loaders.retriever`
    (the page's two data reads) and the page module's own `st` calls.
    """

    def _run(judgements, retriever=_DEFAULT):
        written: dict[str, list] = {"info": [], "warning": [], "caption": [], "frames": []}
        monkeypatch.setattr(quality.loaders, "review_labels", lambda workflow: judgements)
        monkeypatch.setattr(
            quality.loaders,
            "retriever",
            lambda: _FakeRetriever() if retriever is _DEFAULT else retriever,
        )
        monkeypatch.setattr(quality.st, "divider", lambda: None)
        monkeypatch.setattr(quality.st, "subheader", lambda *a, **k: None)
        monkeypatch.setattr(quality.st, "info", lambda m: written["info"].append(m))
        monkeypatch.setattr(quality.st, "warning", lambda m: written["warning"].append(m))
        monkeypatch.setattr(quality.st, "caption", lambda m: written["caption"].append(m))
        monkeypatch.setattr(quality.st, "dataframe", lambda df, **k: written["frames"].append(df))
        quality._retrieval_benchmark()
        return written

    return _run


def test_an_ai_only_judgement_set_is_refused_as_evidence(run):
    written = run(_judgements("codex-assisted"))

    # The harness ran -- a table exists -- but the claim does not.
    assert written["frames"], "the benchmark should still compute"
    assert written["warning"], "an AI-only judgement set must be disclosed"
    message = written["warning"][0]
    assert "Not approved evidence" in message
    assert "codex-assisted" in message
    assert "WP-13" in message
    assert not written["caption"], "no approving caption may accompany the warning"


def test_independent_reviewers_produce_an_approving_caption(run):
    written = run(_judgements("rene"))

    assert written["frames"]
    assert not written["warning"]
    assert "rene" in written["caption"][0]


def test_no_judgements_explains_how_to_produce_them_instead_of_scoring(run):
    written = run(pd.DataFrame())

    assert not written["frames"], "nothing may be scored without judgements"
    assert "evidence retrieval" in written["info"][0]


def test_judgements_with_nothing_marked_relevant_are_not_scored(run):
    frame = _judgements("rene")
    frame["label"] = "not_relevant"

    written = run(frame)

    assert not written["frames"]
    assert "none marked a result relevant" in written["warning"][0]


def test_the_table_reports_every_mode_with_its_metrics(run):
    written = run(_judgements("rene"))
    table = written["frames"][0]

    assert set(table["Mode"]) == set(BENCHMARK_MODES)
    for column in (f"Recall@{quality.RETRIEVAL_BENCHMARK_K}", "MRR", "p50 latency (ms)"):
        assert column in table.columns


def test_a_mode_that_cannot_run_is_left_out_and_named(run):
    written = run(_judgements("rene"), retriever=_FakeRetriever(rerank_available=False))

    modes = set(written["frames"][0]["Mode"])
    assert "Hybrid RRF + rerank, abstracts" not in modes
    assert len(modes) == len(BENCHMARK_MODES) - 1
    assert "Not run here: Hybrid RRF + rerank, abstracts" in written["caption"][0]


def test_the_disclosure_names_the_chunk_level_pool(run):
    written = run(_judgements("codex-assisted"))

    assert "pooled at depth 10 from the chunk-level modes" in written["warning"][0]
    assert not written["caption"]


def test_no_published_version_is_explained_not_scored(run):
    written = run(_judgements("rene"), retriever=None)

    assert not written["frames"]
    assert "No published version" in written["info"][0]
