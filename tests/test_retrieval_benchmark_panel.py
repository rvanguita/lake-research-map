"""WP-21 / RQ-08: the benchmark surface, and what it refuses to claim.

`retrieval_eval.py` and the pooled sampler both existed and were reachable
only from `tests/`, so the project could compute Recall@k, MRR and nDCG and
had nowhere to show them. The panel exists now -- and the property that
matters is not the arithmetic, it is that a table scored against one
AI-assisted pass is labelled as not being evidence.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from lake_research_map.dashboard.pages import quality


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


def _chunks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": [f"10.1/{i}" for i in range(3)],
            "text": ["a", "b", "c"],
            "embedding_bin": [b"", b"", b""],
        }
    )


def _run(judgements, chunks=None):
    """Render the panel, capturing what it wrote instead of drawing it."""
    written: dict[str, list] = {"info": [], "warning": [], "caption": [], "frames": []}

    def _search(text, frame, top_k=10):
        return pd.DataFrame({"doi": [f"10.1/{i}" for i in range(min(top_k, 3))]})

    with (
        patch.object(quality.loaders, "review_labels", lambda workflow: judgements),
        patch.object(
            quality.loaders, "chunk_search_data", lambda: _chunks() if chunks is None else chunks
        ),
        patch.object(quality, "semantic_search", _search),
        patch.object(quality, "bm25_search", _search),
        patch.object(quality, "hybrid_search_rrf", _search),
        patch.object(quality.st, "divider", lambda: None),
        patch.object(quality.st, "subheader", lambda *a, **k: None),
        patch.object(quality.st, "info", lambda m: written["info"].append(m)),
        patch.object(quality.st, "warning", lambda m: written["warning"].append(m)),
        patch.object(quality.st, "caption", lambda m: written["caption"].append(m)),
        patch.object(quality.st, "dataframe", lambda df, **k: written["frames"].append(df)),
    ):
        quality._retrieval_benchmark()
    return written


def test_an_ai_only_judgement_set_is_refused_as_evidence():
    written = _run(_judgements("codex-assisted"))

    # The harness ran -- a table exists -- but the claim does not.
    assert written["frames"], "the benchmark should still compute"
    assert written["warning"], "an AI-only judgement set must be disclosed"
    message = written["warning"][0]
    assert "Not approved evidence" in message
    assert "codex-assisted" in message
    assert "WP-13" in message
    assert not written["caption"], "no approving caption may accompany the warning"


def test_independent_reviewers_produce_an_approving_caption():
    written = _run(_judgements("rene"))

    assert written["frames"]
    assert not written["warning"]
    assert "rene" in written["caption"][0]


def test_no_judgements_explains_how_to_produce_them_instead_of_scoring():
    written = _run(pd.DataFrame())

    assert not written["frames"], "nothing may be scored without judgements"
    assert "evidence retrieval" in written["info"][0]


def test_judgements_with_nothing_marked_relevant_are_not_scored():
    frame = _judgements("rene")
    frame["label"] = "not_relevant"

    written = _run(frame)

    assert not written["frames"]
    assert "none marked a result relevant" in written["warning"][0]


def test_the_table_reports_every_mode_with_its_metrics():
    written = _run(_judgements("rene"))
    table = written["frames"][0]

    assert set(table["Mode"]) == {"Dense (BGE-small)", "BM25", "Hybrid RRF"}
    for column in (f"Recall@{quality.RETRIEVAL_BENCHMARK_K}", "MRR", "p50 latency (ms)"):
        assert column in table.columns
