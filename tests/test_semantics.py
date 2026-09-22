"""Tests for the pure parts of `transform.semantics`.

Vectors are hand-built so the expected answer is arithmetic, not a property of
the real embedding model -- these tests never load fastembed or touch MySQL.
"""

from __future__ import annotations

import itertools

import numpy as np

from lake_research_map.transform.semantics import (
    discover_themes,
    near_duplicate_pairs,
    project_2d,
    reduced_space,
    relevance_scores,
    theme_labels_from_terms,
    unresolved_duplicate_pairs,
)


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype="float32")
    return vector / np.linalg.norm(vector)


def test_relevance_scores_ranks_by_angle_to_anchor():
    anchor = _unit(1, 0)
    matrix = np.array([_unit(1, 0), _unit(1, 1), _unit(0, 1)])

    scores = relevance_scores(matrix, anchor)

    assert scores[0] > scores[1] > scores[2]
    assert np.isclose(scores[0], 1.0, atol=1e-6)
    assert np.isclose(scores[2], 0.0, atol=1e-6)


def test_relevance_scores_normalizes_unnormalized_input():
    """A caller passing raw (non-unit) vectors must get the same ranking."""
    anchor = np.array([5.0, 0.0], dtype="float32")
    matrix = np.array([[3.0, 0.0], [0.0, 7.0]], dtype="float32")

    scores = relevance_scores(matrix, anchor)

    assert np.isclose(scores[0], 1.0, atol=1e-6)
    assert np.isclose(scores[1], 0.0, atol=1e-6)


def test_relevance_scores_handles_empty_matrix():
    assert relevance_scores(np.zeros((0, 4), dtype="float32"), _unit(1, 0, 0, 0)).shape == (0,)


def test_near_duplicate_pairs_respects_threshold_and_never_self_pairs():
    matrix = np.array([_unit(1, 0), _unit(1, 0.02), _unit(0, 1)])
    dois = ["10.1/a", "10.1/b", "10.1/c"]

    pairs = near_duplicate_pairs(matrix, dois, threshold=0.95)

    assert len(pairs) == 1
    doi_a, doi_b, similarity = pairs[0]
    assert {doi_a, doi_b} == {"10.1/a", "10.1/b"}
    assert doi_a != doi_b
    assert similarity >= 0.95
    assert unresolved_duplicate_pairs(pairs, {("10.1/b", "10.1/a")}) == []


def test_near_duplicate_pairs_sorted_by_similarity_descending():
    matrix = np.array([_unit(1, 0), _unit(1, 0.3), _unit(1, 0.01)])
    dois = ["10.1/a", "10.1/b", "10.1/c"]

    pairs = near_duplicate_pairs(matrix, dois, threshold=0.5)

    assert [p[2] for p in pairs] == sorted((p[2] for p in pairs), reverse=True)


def test_near_duplicate_pairs_empty_below_two_rows():
    assert near_duplicate_pairs(np.array([_unit(1, 0)]), ["10.1/a"]) == []


def test_discover_themes_separates_two_obvious_groups():
    """Two well-separated blobs must land in different themes, with labels
    drawn from the words that distinguish them."""
    rng = np.random.default_rng(0)
    group_a = np.tile(_unit(1, 0), (6, 1)) + rng.normal(0, 0.01, (6, 2))
    group_b = np.tile(_unit(0, 1), (6, 1)) + rng.normal(0, 0.01, (6, 2))
    matrix = np.vstack([group_a, group_b]).astype("float32")
    texts = ["voltage feeder substation grid"] * 6 + ["logistics warehouse freight truck"] * 6

    labels, theme_labels = discover_themes(matrix, texts, k=2)

    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]
    assert set(theme_labels) == set(int(label) for label in labels)
    # Labels are Title Case for display, so compare case-insensitively.
    joined = " ".join(theme_labels.values()).lower()
    assert "logistics" in joined or "warehouse" in joined or "freight" in joined


def test_discover_themes_is_deterministic():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 1, (20, 5)).astype("float32")
    texts = [f"topic {i % 3} power distribution planning" for i in range(20)]

    first_labels, first_names = discover_themes(matrix, texts, k=3)
    second_labels, second_names = discover_themes(matrix, texts, k=3)

    assert np.array_equal(first_labels, second_labels)
    assert first_names == second_names


def test_discover_themes_caps_k_at_sample_count():
    matrix = np.array([_unit(1, 0), _unit(0, 1)], dtype="float32")

    labels, theme_labels = discover_themes(matrix, ["alpha text", "beta text"], k=8)

    assert len(labels) == 2
    assert len(theme_labels) <= 2


def test_project_2d_returns_one_coordinate_pair_per_row():
    rng = np.random.default_rng(2)
    matrix = rng.normal(0, 1, (12, 6)).astype("float32")

    coords = project_2d(matrix)

    assert coords.shape == (12, 2)
    assert np.isfinite(coords).all()


def test_project_2d_handles_degenerate_input():
    """Fewer than 3 rows can't be projected -- must degrade, not raise."""
    assert project_2d(np.zeros((0, 4), dtype="float32")).shape == (0, 2)
    assert project_2d(np.array([_unit(1, 0)], dtype="float32")).shape == (1, 2)


# --- contrastive screening -------------------------------------------------


def test_contrastive_margin_separates_the_two_readings_of_the_query():
    # The whole point of the second anchor: a paper can sit reasonably close to
    # the topic anchor and still be closer to the off-topic one, which a single
    # score cannot express.
    topic = _unit(1, 0)
    offtopic = _unit(0, 1)
    matrix = np.array([_unit(1, 0.2), _unit(0.2, 1)])

    margin = relevance_scores(matrix, topic) - relevance_scores(matrix, offtopic)

    assert margin[0] > 0 > margin[1]


# --- reduced space ---------------------------------------------------------


def test_reduced_space_caps_components_at_the_data():
    rng = np.random.default_rng(0)
    matrix = (rng.normal(size=(12, 8)) * 7).astype("float32")

    reduced = reduced_space(matrix)

    # min(PCA_COMPONENTS, n_samples, n_features) -- asking for 50 from 8
    # features would raise, which is what a fresh/tiny corpus would hit.
    assert reduced.shape == (12, 8)


def test_reduced_space_ignores_vector_magnitude():
    # Cosine is the only geometry that means anything here, so a longer vector
    # must not land anywhere different from its unit version.
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype="float32")
    scaled = base.copy()
    scaled[0] *= 9

    assert np.allclose(reduced_space(base), reduced_space(scaled), atol=1e-5)


# --- theme labels ----------------------------------------------------------


def _labelled_corpus() -> tuple[list[str], np.ndarray]:
    """Three themes with their own vocabulary, over a shared generic sentence.

    Uneven group sizes on purpose: `max_df` drops a term that shows up in more
    than half the corpus, so two equal halves would sit exactly on the cutoff.
    """
    generic = "distribution planning study"
    groups = {
        0: ("substation feeder voltage regulation", 7),
        1: ("warehouse routing inventory freight", 7),
        2: ("forecasting neural network demand", 6),
    }
    texts: list[str] = []
    labels: list[int] = []
    for theme, (vocabulary, count) in groups.items():
        for _ in range(count):
            texts.append(f"{generic} {vocabulary}")
            labels.append(theme)
    return texts, np.array(labels)


def test_theme_labels_name_each_group_from_its_own_vocabulary():
    texts, labels = _labelled_corpus()

    result = theme_labels_from_terms(texts, labels)

    assert "Substation" in result[0]
    assert "Warehouse" in result[1]
    assert "Forecasting" in result[2]


def test_theme_labels_drop_terms_the_whole_corpus_shares():
    # Without this, every theme in a distribution-planning corpus ends up
    # labelled "distribution - planning - study".
    texts, labels = _labelled_corpus()

    result = theme_labels_from_terms(texts, labels)

    for label in result.values():
        assert "Distribution" not in label
        assert "Planning" not in label


def test_theme_labels_never_repeat_a_term_inside_another():
    texts = ["energy storage battery dispatch " * 3] * 8 + [
        "cable trench duct installation " * 3
    ] * 9
    labels = np.array([0] * 8 + [1] * 9)

    result = theme_labels_from_terms(texts, labels)

    for label in result.values():
        terms = [term.lower() for term in label.split(" · ")]
        for first, second in itertools.combinations(terms, 2):
            assert first not in second and second not in first


def test_theme_labels_fall_back_to_numbers_without_a_vocabulary():
    # Nothing to build a vocabulary from: the themes still need a name.
    labels = np.array([0, 0, 1, 1])

    result = theme_labels_from_terms(["", "", "", ""], labels)

    assert result == {0: "Tema 1", 1: "Tema 2"}


def test_build_semantics_end_to_end_with_injected_anchors(gold_session):
    """build_semantics runs to completion with injected anchor vectors,
    bypassing the fastembed model entirely."""
    from lake_research_map.db.gold_models import Chunk, DuplicateOverride, DuplicatePair, Semantics
    from lake_research_map.transform.semantics import build_semantics

    rng = np.random.default_rng(42)
    n = 12
    dim = 384

    # Create gold chunks with embeddings to simulate an embedded corpus.
    shared = rng.standard_normal(dim).astype(np.float32)
    shared /= np.linalg.norm(shared)
    for i in range(n):
        vec = shared if i < 2 else rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        gold_session.add(
            Chunk(
                doi=f"10.1000/test-{i}",
                seq=0,
                chunk_type="abstract",
                text=f"distribution planning article number {i} about power systems"
                if i < 8
                else f"logistics warehouse routing freight article {i}",
                char_len=60,
                embedding=vec.tolist(),
            )
        )
    gold_session.add(
        DuplicateOverride(
            doi_a="10.1000/test-0",
            doi_b="10.1000/test-1",
            decision="keep",
            canonical_doi=None,
            reason="Distinct publications after review",
        )
    )
    gold_session.commit()

    # Inject anchor vectors instead of loading the real embedding model.
    topic_anchor = rng.standard_normal(dim).astype(np.float32)
    off_anchor = rng.standard_normal(dim).astype(np.float32)

    stats = build_semantics(gold_session, anchor_vectors=(topic_anchor, off_anchor))

    assert stats["articles"] == n
    assert stats["themes"] > 0
    assert "median_relevance" in stats
    assert "median_margin" in stats
    assert stats["duplicate_pairs_reviewed"] == 1

    # Verify lit_semantics rows were written.
    sem_count = gold_session.query(Semantics).count()
    assert sem_count == n
    assert gold_session.query(DuplicatePair).count() == 0
    assert gold_session.query(DuplicateOverride).count() == 1


def test_theme_sweep_keeps_the_losing_candidates():
    """WP-17: the k search is only auditable if the rejected candidates survive.

    Silhouette is nearly flat across the range on this corpus, so the near-tie
    rule -- not the maximum -- is what picks k. A sweep that reported only the
    winner would hide that.
    """
    import numpy as np

    from lake_research_map.transform.semantics import MAX_THEMES, MIN_THEMES, theme_sweep

    rng = np.random.default_rng(0)
    matrix = np.vstack([rng.normal(centre, 0.4, (40, 6)) for centre in (-5, 0, 5, 10)])

    sweep = theme_sweep(matrix)

    assert [row["k"] for row in sweep] == list(range(MIN_THEMES, MAX_THEMES + 1))
    # Four well-separated blobs: k=4 must win outright here.
    assert max(sweep, key=lambda row: row["silhouette"])["k"] == 4
    for row in sweep:
        assert 0.0 < row["smallest_cluster_share"] <= 1.0
        assert isinstance(row["rejected_small_cluster"], bool)
        assert row["rejected_small_cluster"] == (row["smallest_cluster_share"] < 0.02)

    # Too few rows to sweep at all returns nothing rather than guessing.
    assert theme_sweep(np.zeros((2, 3))) == []


def test_projection_measures_separate_invented_from_lost_neighbours():
    """Trustworthiness alone cannot fail a projection that tears a cluster apart."""
    import numpy as np

    from lake_research_map.dashboard.analytics import semantic_stability_diagnostics

    rng = np.random.default_rng(7)
    matrix = np.vstack([rng.normal(-3, 0.1, (20, 4)), rng.normal(3, 0.1, (20, 4))])
    labels = np.array([0] * 20 + [1] * 20)

    faithful = semantic_stability_diagnostics(matrix, labels, matrix[:, :2], n_bootstrap=3, seed=3)
    scrambled = semantic_stability_diagnostics(
        matrix, labels, rng.normal(0, 1, (40, 2)), n_bootstrap=3, seed=3
    )

    for key in ("projection_trustworthiness", "projection_continuity", "knn_overlap"):
        assert faithful[key] > scrambled[key], key
    assert faithful["knn_overlap"] > 0.6
    assert scrambled["knn_overlap"] < 0.5
    assert faithful["neighbors"] >= 1
