"""Deterministic, auditable publication-category classification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PUBLICATION_CATEGORIES = ("journal", "conference", "review", "other")
CATEGORY_PRECEDENCE = {"other": 0, "journal": 1, "conference": 2, "review": 3}
# A category is the primary decision. This secondary order makes the selected
# basis deterministic when two source records produce the same category.
BASIS_PRECEDENCE = {
    "other_record_type": 0,
    "journal_record_type": 10,
    "journal_metadata": 11,
    "conference_venue_metadata": 20,
    "conference_record_type": 21,
    "review_metadata": 30,
    "title_review": 31,
}

_REVIEW_RE = re.compile(
    r"\b(review|survey|overview)\b|state[- ]of[- ]the[- ]art|systematic mapping",
    re.IGNORECASE,
)
_CONFERENCE_RE = re.compile(
    r"\b(conference|proceedings|symposium|congress|workshop)\b|"
    r"\bprocedia\b|\bifac[- ]?papersonline\b",
    re.IGNORECASE,
)
_CONFERENCE_TYPES = {"conference", "inproceedings", "proceedings"}
_JOURNAL_TYPES = {"article"}


def _normalise_text(value: str | None) -> str:
    """Normalize publisher metadata before applying keyword rules."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True, slots=True)
class PublicationClassification:
    category: str
    basis: str


def classify_publication(
    *,
    record_type: str | None,
    title: str | None,
    venue: str | None,
    document_type: str | None = None,
    note: str | None = None,
) -> PublicationClassification:
    """Return one mutually exclusive category and the rule that selected it."""
    normalized_type = _normalise_text(record_type)
    title_text = _normalise_text(title)
    document_metadata = " ".join(
        value for value in (_normalise_text(document_type), _normalise_text(note)) if value
    )
    venue_metadata = " ".join(
        value for value in (_normalise_text(venue), document_metadata) if value
    )

    if _REVIEW_RE.search(title_text):
        return PublicationClassification("review", "title_review")
    if _REVIEW_RE.search(document_metadata):
        return PublicationClassification("review", "review_metadata")
    if normalized_type in _CONFERENCE_TYPES:
        return PublicationClassification("conference", "conference_record_type")
    if _CONFERENCE_RE.search(venue_metadata):
        return PublicationClassification("conference", "conference_venue_metadata")
    if normalized_type in _JOURNAL_TYPES:
        return PublicationClassification("journal", "journal_record_type")
    if re.search(r"\b(journal|magazine|early access)\b", document_metadata):
        return PublicationClassification("journal", "journal_metadata")
    return PublicationClassification("other", "other_record_type")


def preferred_classification(rows) -> tuple[str, str]:
    """Choose a stable winner when source records for one DOI disagree."""
    candidates = [
        (row.publication_category, row.publication_category_basis)
        for row in rows
        if row.publication_category in PUBLICATION_CATEGORIES
    ]
    if not candidates:
        return "other", "other_record_type"
    return max(
        candidates,
        key=lambda item: (
            CATEGORY_PRECEDENCE[item[0]],
            BASIS_PRECEDENCE.get(item[1], 0),
            item[1],
        ),
    )
