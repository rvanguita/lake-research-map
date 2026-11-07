"""FR-08: the dashboard reads the Gold version the publication pointer names.

Gold holds several immutable snapshots side by side in `lit_dataset_articles`
-- the active one, older ones kept for reactivation, and the candidate a run is
still building. A reader that took "the newest version" or "every row" would
show a half-validated candidate the moment `gold` finished, before its
contracts had run. `data.load_articles("gold")` must follow
`lit_publication_state.active_version_id` and nothing else.

Injection point: `apps.dashboard.data.get_engine`, bound to the `gold_session`
fixture's in-memory engine (the same one `tests/test_duplicate_resolution.py`
uses). Writing the rows directly, rather than through `materialize_version`,
keeps the test about the reader: Qdrant and the publication contract are not
involved.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from apps.dashboard import data
from db.gold_models import Article, DatasetArticle, DatasetVersion, PublicationState
from db.silver_models import Article as SilverArticle

PUBLISHED = "a" * 64
CANDIDATE = "b" * 64


def _version(session, version_id: str, status: str, created_at: datetime, dois: list[str]):
    session.add(DatasetVersion(version_id=version_id, status=status, created_at=created_at))
    for doi in dois:
        # Every column of `data.GOLD_ANALYTICAL_COLUMNS`, so the snapshot also
        # passes the dashboard's readiness contract.
        session.add(
            DatasetArticle(
                dataset_version_id=version_id,
                doi=doi,
                sources=["ieee"],
                title=doi,
                authors=["A. Author"],
                year=2024,
                venue="IEEE Trans. Power Syst.",
                keywords=["planning"],
                abstract="An abstract.",
                citation_count=1,
                reference_count=2,
                has_pdf=False,
                is_non_article=False,
                publication_category="journal",
            )
        )


@pytest.fixture
def gold(gold_session, monkeypatch):
    monkeypatch.setattr(data, "get_engine", lambda _layer: gold_session.get_bind())
    _version(gold_session, PUBLISHED, "active", datetime(2026, 11, 1), ["10.1/a1", "10.1/a2"])
    # Newer, unpublished and disjoint: any reader that is not pointer-bound
    # picks some of these up.
    _version(gold_session, CANDIDATE, "candidate", datetime(2026, 11, 6), ["10.1/b1"])
    # The live projection table too, holding something neither version has.
    gold_session.add(Article(doi="10.1/live-only", sources=["ieee"], title="live"))
    gold_session.commit()
    return gold_session


def test_load_articles_returns_the_published_version_not_the_newer_candidate(gold):
    gold.add(PublicationState(id=1, active_version_id=PUBLISHED))
    gold.commit()

    articles = data.load_articles("gold")

    assert sorted(articles["doi"]) == ["10.1/a1", "10.1/a2"]
    assert set(articles["dataset_version_id"]) == {PUBLISHED}


def test_moving_the_pointer_moves_what_the_dashboard_reads(gold):
    gold.add(PublicationState(id=1, active_version_id=CANDIDATE))
    gold.commit()

    assert data.active_dataset_version() == CANDIDATE
    assert list(data.load_articles("gold")["doi"]) == ["10.1/b1"]


@pytest.mark.parametrize("pointer_row", [False, True], ids=["no-state-row", "null-pointer"])
def test_no_published_version_reads_as_empty_gold(gold, pointer_row):
    """With no pointer, Gold is empty -- never a guess at "the latest" version."""
    if pointer_row:
        gold.add(PublicationState(id=1, active_version_id=None))
        gold.commit()

    assert data.active_dataset_version() is None
    assert data.load_articles("gold").empty


def test_with_no_published_version_the_dashboard_degrades_to_silver_and_says_so(
    gold, silver_session, monkeypatch
):
    """`select_articles_layer` is where the fallback lives, not `load_articles`.

    An empty pointer-bound Gold fails the readiness contract, so the dashboard
    shows Silver -- labelled non-canonical, with the reason -- rather than the
    candidate's rows or the live `lit_articles` projection.
    """
    silver_session.add(SilverArticle(doi="10.1/s1", title="silver", record_type="article"))
    silver_session.commit()
    engines = {"gold": gold.get_bind(), "silver": silver_session.get_bind()}
    monkeypatch.setattr(data, "get_engine", lambda layer: engines[layer])

    layer, articles, status = data.select_articles_layer()

    assert layer == "silver"
    assert list(articles["doi"]) == ["10.1/s1"]
    assert status["is_canonical"] is False
    assert status["dataset_version_id"] is None
    assert status["fallback_reasons"] == ("The Gold layer is empty or unavailable.",)


def test_a_published_version_is_selected_as_canonical(gold):
    gold.add(PublicationState(id=1, active_version_id=PUBLISHED))
    gold.commit()

    layer, articles, status = data.select_articles_layer()

    assert layer == "gold"
    assert status == {
        "layer": "gold",
        "is_canonical": True,
        "dataset_version_id": PUBLISHED,
        "fallback_reasons": (),
    }
    assert sorted(articles["doi"]) == ["10.1/a1", "10.1/a2"]
