"""Persistent review decisions for cross-DOI near-duplicate candidates.

The semantic stage owns the detected-pair table. This module treats overrides
as durable human input, filters reviewed pairs from read-time queues, and
produces a deterministic merge plan for the derived Gold layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import DuplicateOverride, DuplicatePair
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.bronze_articles import normalize_doi

DECISIONS = {"merge", "keep"}


class DuplicateResolutionError(ValueError):
    """A reviewer-facing validation error suitable for concise CLI output."""


@dataclass(frozen=True, slots=True)
class MergePlan:
    groups: dict[str, tuple[str, ...]]
    inactive_overrides: int


def normalized_pair(doi_a: str, doi_b: str) -> tuple[str, str]:
    """Normalize and lexically order an unordered DOI pair."""
    left = normalize_doi(doi_a)
    right = normalize_doi(doi_b)
    if left is None or right is None:
        raise DuplicateResolutionError("Both DOI values must be non-empty.")
    if left == right:
        raise DuplicateResolutionError("A duplicate decision requires two distinct DOI values.")
    return (left, right) if left < right else (right, left)


def _pair_key(doi_a: str, doi_b: str) -> tuple[str, str]:
    """Order already-normalized database values without accepting empty input."""
    return (doi_a, doi_b) if doi_a < doi_b else (doi_b, doi_a)


def _existing_override(gold_session: Session, pair: tuple[str, str]) -> DuplicateOverride | None:
    return gold_session.scalar(
        select(DuplicateOverride).where(
            DuplicateOverride.doi_a == pair[0],
            DuplicateOverride.doi_b == pair[1],
        )
    )


def _reviewed_pairs(gold_session: Session) -> set[tuple[str, str]]:
    return set(gold_session.execute(select(DuplicateOverride.doi_a, DuplicateOverride.doi_b)))


def _is_candidate(gold_session: Session, pair: tuple[str, str]) -> bool:
    return (
        gold_session.scalar(
            select(DuplicatePair.id)
            .where(
                or_(
                    and_(DuplicatePair.doi_a == pair[0], DuplicatePair.doi_b == pair[1]),
                    and_(DuplicatePair.doi_a == pair[1], DuplicatePair.doi_b == pair[0]),
                )
            )
            .limit(1)
        )
        is not None
    )


def _require_silver_dois(silver_session: Session, pair: tuple[str, str]) -> None:
    expected = set(pair)
    existing = set(
        silver_session.scalars(select(SilverArticle.doi).where(SilverArticle.doi.in_(expected)))
    )
    missing = sorted(expected - existing)
    if missing:
        raise DuplicateResolutionError(f"DOI values not found in Silver: {', '.join(missing)}")


def build_merge_plan(overrides: list[DuplicateOverride], available_dois: set[str]) -> MergePlan:
    """Validate all merge decisions and return stable canonical groups.

    Merge groups are deliberately flat stars. A DOI may be canonical for many
    duplicates, but may not itself be a duplicate, and a duplicate may have
    only one canonical. This makes the result independent of insertion order.
    """
    duplicate_to_canonical: dict[str, str] = {}
    canonicals: set[str] = set()
    merge_rows: list[tuple[str, str]] = []

    for override in sorted(overrides, key=lambda row: (row.doi_a, row.doi_b)):
        if override.decision != "merge":
            continue
        canonical = override.canonical_doi
        if canonical not in (override.doi_a, override.doi_b):
            raise DuplicateResolutionError(
                f"Invalid canonical DOI for pair {override.doi_a}, {override.doi_b}."
            )
        duplicate = override.doi_b if canonical == override.doi_a else override.doi_a
        existing = duplicate_to_canonical.get(duplicate)
        if existing is not None and existing != canonical:
            raise DuplicateResolutionError(
                f"{duplicate} is assigned to both {existing} and {canonical}."
            )
        duplicate_to_canonical[duplicate] = canonical
        canonicals.add(canonical)
        merge_rows.append((canonical, duplicate))

    chained = sorted(canonicals & duplicate_to_canonical.keys())
    if chained:
        raise DuplicateResolutionError(
            f"Merge chains are not allowed; canonical DOI is also a duplicate: {chained[0]}."
        )

    groups: dict[str, list[str]] = {}
    inactive = 0
    for canonical, duplicate in merge_rows:
        if canonical not in available_dois or duplicate not in available_dois:
            inactive += 1
            continue
        groups.setdefault(canonical, []).append(duplicate)

    stable_groups = {
        canonical: tuple(sorted(duplicates)) for canonical, duplicates in sorted(groups.items())
    }
    return MergePlan(groups=stable_groups, inactive_overrides=inactive)


def record_decision(
    gold_session: Session,
    silver_session: Session,
    *,
    decision: str,
    doi_a: str,
    doi_b: str,
    reason: str,
    canonical_doi: str | None = None,
) -> DuplicateOverride:
    """Create or replace one reviewed decision and commit it atomically."""
    if decision not in DECISIONS:
        raise DuplicateResolutionError(f"Unsupported duplicate decision: {decision!r}")
    reason = reason.strip()
    if not reason:
        raise DuplicateResolutionError("A review reason is required.")

    pair = normalized_pair(doi_a, doi_b)
    _require_silver_dois(silver_session, pair)
    existing = _existing_override(gold_session, pair)
    if existing is None and not _is_candidate(gold_session, pair):
        raise DuplicateResolutionError(
            "The DOI pair is not in the current near-duplicate review queue."
        )

    canonical = None
    if decision == "merge":
        canonical = normalize_doi(canonical_doi)
        if canonical not in pair:
            raise DuplicateResolutionError(
                "The canonical DOI must be one member of the reviewed pair."
            )

    candidate = DuplicateOverride(
        doi_a=pair[0],
        doi_b=pair[1],
        decision=decision,
        canonical_doi=canonical,
        reason=reason,
    )

    available_dois = set(silver_session.scalars(select(SilverArticle.doi)))
    with gold_session.no_autoflush:
        other_overrides = gold_session.scalars(
            select(DuplicateOverride).where(
                ~((DuplicateOverride.doi_a == pair[0]) & (DuplicateOverride.doi_b == pair[1]))
            )
        ).all()
        build_merge_plan([*other_overrides, candidate], available_dois)

    if existing is None:
        saved = candidate
        gold_session.add(saved)
    else:
        existing.decision = decision
        existing.canonical_doi = canonical
        existing.reason = reason
        saved = existing
    gold_session.commit()
    gold_session.refresh(saved)
    return saved


def undo_decision(gold_session: Session, *, doi_a: str, doi_b: str) -> bool:
    """Remove an override. Derived tables change on their next rebuild."""
    pair = normalized_pair(doi_a, doi_b)
    existing = _existing_override(gold_session, pair)
    if existing is None:
        return False
    gold_session.delete(existing)
    gold_session.commit()
    return True


def active_merge_plan(gold_session: Session, available_dois: set[str]) -> MergePlan:
    overrides = gold_session.scalars(
        select(DuplicateOverride).order_by(DuplicateOverride.doi_a, DuplicateOverride.doi_b)
    ).all()
    return build_merge_plan(overrides, available_dois)


def list_review_rows(gold_session: Session, *, include_resolved: bool = False) -> list[dict]:
    """Return the current unresolved queue and, optionally, review history."""
    reviewed = _reviewed_pairs(gold_session)
    rows = [
        {
            "kind": "candidate",
            "doi_a": pair.doi_a,
            "doi_b": pair.doi_b,
            "similarity": pair.similarity,
            "decision": None,
            "canonical_doi": None,
            "reason": None,
        }
        for pair in gold_session.scalars(
            select(DuplicatePair).order_by(DuplicatePair.similarity.desc())
        )
        if _pair_key(pair.doi_a, pair.doi_b) not in reviewed
    ]
    if include_resolved:
        rows.extend(
            {
                "kind": "resolved",
                "doi_a": override.doi_a,
                "doi_b": override.doi_b,
                "similarity": None,
                "decision": override.decision,
                "canonical_doi": override.canonical_doi,
                "reason": override.reason,
            }
            for override in gold_session.scalars(
                select(DuplicateOverride).order_by(DuplicateOverride.updated_at.desc())
            )
        )
    return rows
