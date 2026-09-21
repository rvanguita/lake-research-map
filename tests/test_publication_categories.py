from lake_research_map.transform.publication_categories import (
    classify_publication,
    preferred_classification,
)


def test_review_has_precedence_over_venue_type():
    result = classify_publication(
        record_type="inproceedings",
        title="A systematic review of distribution planning",
        venue="International Conference on Energy Systems",
    )
    assert (result.category, result.basis) == ("review", "title_review")


def test_conference_uses_record_type_or_proceedings_metadata():
    typed = classify_publication(
        record_type="inproceedings", title="Planning method", venue="Power Systems"
    )
    metadata = classify_publication(
        record_type="article",
        title="Planning method",
        venue="Procedia Computer Science",
        note="International conference",
    )
    assert typed.category == "conference"
    assert typed.basis == "conference_record_type"
    assert metadata.category == "conference"
    assert metadata.basis == "conference_venue_metadata"


def test_journal_and_other_are_deterministic_fallbacks():
    journal = classify_publication(
        record_type="article", title="Planning method", venue="Applied Energy"
    )
    other = classify_publication(
        record_type="incollection", title="Planning chapter", venue="Planning handbook"
    )
    assert (journal.category, journal.basis) == ("journal", "journal_record_type")
    assert (other.category, other.basis) == ("other", "other_record_type")


def test_preferred_classification_uses_review_first_precedence():
    class Row:
        def __init__(self, category, basis):
            self.publication_category = category
            self.publication_category_basis = basis

    category, basis = preferred_classification(
        [Row("journal", "journal_record_type"), Row("review", "title_review")]
    )
    assert (category, basis) == ("review", "title_review")
