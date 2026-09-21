from __future__ import annotations

import pandas as pd

from lake_research_map.dashboard import loaders


def test_publication_category_filter_keeps_only_selected_categories(monkeypatch):
    frame = pd.DataFrame(
        {
            "doi": ["10.1/journal", "10.1/conference", "10.1/review", "10.1/other"],
            "publication_category": ["journal", "conference", "review", "other"],
        }
    )
    monkeypatch.setattr(loaders, "articles", lambda: ("gold", frame))

    filtered = loaders.filter_articles.__wrapped__(publication_categories=("conference", "review"))

    assert filtered["doi"].tolist() == ["10.1/conference", "10.1/review"]


def test_empty_category_selection_preserves_the_full_corpus(monkeypatch):
    frame = pd.DataFrame(
        {
            "doi": ["10.1/journal", "10.1/other"],
            "publication_category": ["journal", "other"],
        }
    )
    monkeypatch.setattr(loaders, "articles", lambda: ("gold", frame))

    filtered = loaders.filter_articles.__wrapped__()

    assert filtered["doi"].tolist() == frame["doi"].tolist()
