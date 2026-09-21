"""Deterministic, auditable publication-category classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

PUBLICATION_CATEGORIES = ("journal", "conference", "review", "other")
CATEGORY_PRECEDENCE = {"other": 0, "journal": 1, "conference": 2, "review": 3}

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
    normalized_type = (record_type or "").strip().casefold()
    title_text = title or ""
    venue_metadata = " ".join(value or "" for value in (venue, document_type, note))

    if _REVIEW_RE.search(title_text):
        return PublicationClassification("review", "title_review")
    if normalized_type in _CONFERENCE_TYPES:
        return PublicationClassification("conference", "conference_record_type")
    if _CONFERENCE_RE.search(venue_metadata):
        return PublicationClassification("conference", "conference_venue_metadata")
    if normalized_type in _JOURNAL_TYPES or re.search(
        r"\b(journal|magazine|early access)\b", document_type or "", re.IGNORECASE
    ):
        return PublicationClassification("journal", "journal_record_type")
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
    return max(candidates, key=lambda item: CATEGORY_PRECEDENCE[item[0]])
