"""WP-21 / PRD §10: a page must say which dataset version produced its numbers.

`article_population_status()` had carried `dataset_version_id` since WP-04 and
no page ever read it -- only `is_canonical` was consumed -- so nine of ten
pages reported figures that could not be tied to a snapshot.
"""

from __future__ import annotations

from unittest.mock import patch

from lake_research_map.dashboard import loaders


def _render(status, *, shown, total, published="2026-09-21 23:19"):
    captions: list[str] = []
    with (
        patch.object(loaders.st, "caption", captions.append),
        patch.object(loaders, "active_version_published_at", lambda: published),
    ):
        loaders.render_population_provenance(status, shown=shown, total=total)
    return captions[0]


def test_the_caption_names_the_version_and_the_population():
    caption = _render(
        {"layer": "gold", "is_canonical": True, "dataset_version_id": "2974743a4b264bcd2dec"},
        shown=3115,
        total=3115,
    )

    # Short enough to read, long enough to identify the snapshot.
    assert "2974743a4b26" in caption
    assert "2974743a4b264bcd2dec" not in caption
    assert "published 2026-09-21 23:19" in caption
    assert "3,115 articles" in caption


def test_filtered_views_report_both_numbers():
    caption = _render(
        {"layer": "gold", "is_canonical": True, "dataset_version_id": "abc123def456"},
        shown=1204,
        total=3115,
    )

    # A figure computed over a filtered subset must not read as the corpus.
    assert "1,204 of 3,115 articles after filters" in caption


def test_a_degraded_layer_says_there_is_no_published_version():
    caption = _render(
        {"layer": "silver", "is_canonical": False, "dataset_version_id": None},
        shown=10,
        total=10,
    )

    assert "Layer `silver`" in caption
    assert "no published version" in caption
    assert "published" in caption


def test_a_missing_publication_timestamp_is_omitted_not_invented():
    caption = _render(
        {"layer": "gold", "is_canonical": True, "dataset_version_id": "abc123def456"},
        shown=5,
        total=5,
        published=None,
    )

    assert "Dataset `abc123def456`" in caption
    assert "published" not in caption
