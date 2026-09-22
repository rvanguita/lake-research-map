"""Walking a sorted key is sampling, not iteration.

The first live crawl stopped at its quota having observed 989 Elsevier works
(`10.1016`) and zero IEEE ones (`10.1109`, 1,340 articles in the corpus),
because `10.1016` sorts entirely before `10.1109`. Every coverage figure
measured on that subset described one publisher while reading as a statement
about the corpus.
"""

from __future__ import annotations

from collections import Counter

from lake_research_map.ingest.openalex import (
    interleave_by_registrant,
    registrant_label,
    registrant_prefix,
)


def _corpus(elsevier: int, ieee: int, iet: int = 0) -> list[str]:
    return (
        [f"10.1016/j.x{index:05d}" for index in range(elsevier)]
        + [f"10.1109/T.{index:05d}" for index in range(ieee)]
        + [f"10.1049/iet.{index:05d}" for index in range(iet)]
    )


def test_sorted_order_is_what_produced_the_bias():
    """The behaviour being replaced, asserted so the regression is legible."""
    dois = sorted(_corpus(1612, 1340))
    first_thousand = Counter(registrant_prefix(doi) for doi in dois[:1000])

    assert first_thousand["10.1016"] == 1000
    assert first_thousand["10.1109"] == 0


def test_any_prefix_of_the_interleaved_order_is_proportional():
    dois = _corpus(1612, 1340, 84)
    order = interleave_by_registrant(sorted(dois))
    share = {
        prefix: count / len(dois)
        for prefix, count in Counter(registrant_prefix(doi) for doi in dois).items()
    }

    for cut in (100, 500, 1000, 2000):
        seen = Counter(registrant_prefix(doi) for doi in order[:cut])
        for prefix, expected in share.items():
            got = seen[prefix] / cut
            assert abs(got - expected) < 0.02, f"{prefix} at cut {cut}: {got:.3f} vs {expected:.3f}"


def test_no_publisher_is_starved_even_at_a_small_cut():
    order = interleave_by_registrant(sorted(_corpus(1612, 1340, 84)))
    seen = Counter(registrant_prefix(doi) for doi in order[:200])

    # The failure mode was a whole publisher at zero.
    assert seen["10.1016"] > 0
    assert seen["10.1109"] > 0


def test_the_order_is_deterministic_across_calls():
    dois = sorted(_corpus(50, 40, 10))
    assert interleave_by_registrant(dois) == interleave_by_registrant(dois)


def test_every_doi_survives_the_reordering():
    dois = sorted(_corpus(37, 23, 7))
    order = interleave_by_registrant(dois)

    assert sorted(order) == dois
    assert len(order) == len(dois)


def test_a_single_publisher_is_left_in_its_own_order():
    dois = sorted(f"10.1016/j.x{index:03d}" for index in range(10))
    assert interleave_by_registrant(dois) == dois


def test_an_empty_population_is_not_an_error():
    assert interleave_by_registrant([]) == []


def test_registrant_labels_fall_back_to_the_prefix():
    assert registrant_label("10.1016/j.x") == "Elsevier"
    assert registrant_label("10.1109/T.1") == "IEEE"
    # An unknown registrant is still reportable, just by its number.
    assert registrant_label("10.99999/unknown") == "10.99999"
