"""`duplicates list|merge|keep|undo`: human decisions on near-duplicate pairs."""

from __future__ import annotations

import argparse

from db.bootstrap import bootstrap
from db.engines import get_session


def _add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--doi-a", required=True, help="First DOI in the reviewed pair")
    parser.add_argument("--doi-b", required=True, help="Second DOI in the reviewed pair")


def configure(subparsers) -> None:
    duplicates = subparsers.add_parser(
        "duplicates", help="Review persistent cross-DOI near-duplicate decisions"
    )
    actions = duplicates.add_subparsers(dest="duplicate_action", required=True)

    list_parser = actions.add_parser("list", help="List unresolved duplicate candidates")
    list_parser.add_argument("--all", action="store_true", help="Include resolved decision history")

    merge = actions.add_parser("merge", help="Merge a duplicate into an explicit canonical DOI")
    merge.add_argument("--canonical-doi", required=True)
    merge.add_argument("--duplicate-doi", required=True)
    merge.add_argument("--reason", required=True)

    keep = actions.add_parser("keep", help="Keep a reviewed pair as distinct publications")
    _add_pair_arguments(keep)
    keep.add_argument("--reason", required=True)

    undo = actions.add_parser("undo", help="Remove a previous duplicate decision")
    _add_pair_arguments(undo)


def run(args: argparse.Namespace) -> None:
    from transform.duplicate_resolution import (
        list_review_rows,
        record_decision,
        undo_decision,
    )

    bootstrap()
    gold_session = get_session("gold")
    silver_session = None
    try:
        if args.duplicate_action == "list":
            rows = list_review_rows(gold_session, include_resolved=args.all)
            if not rows:
                print("No near-duplicate review rows found.")
                return
            for row in rows:
                similarity = (
                    f" similarity={row['similarity']:.4f}" if row["similarity"] is not None else ""
                )
                decision = (
                    f" decision={row['decision']} canonical={row['canonical_doi'] or '-'}"
                    if row["decision"]
                    else ""
                )
                reason = f" reason={row['reason']}" if row["reason"] else ""
                print(
                    f"{row['kind']} doi_a={row['doi_a']} doi_b={row['doi_b']}"
                    f"{similarity}{decision}{reason}"
                )
            return

        if args.duplicate_action == "merge":
            silver_session = get_session("silver")
            override = record_decision(
                gold_session,
                silver_session,
                decision="merge",
                doi_a=args.canonical_doi,
                doi_b=args.duplicate_doi,
                canonical_doi=args.canonical_doi,
                reason=args.reason,
            )
            print(
                f"Recorded merge: {args.duplicate_doi} -> {override.canonical_doi}. "
                "Run --stage all to build and publish a newly versioned dataset."
            )
            return

        if args.duplicate_action == "keep":
            silver_session = get_session("silver")
            record_decision(
                gold_session,
                silver_session,
                decision="keep",
                doi_a=args.doi_a,
                doi_b=args.doi_b,
                reason=args.reason,
            )
            print(
                "Recorded keep-separate decision. The pair is now excluded from the review queue."
            )
            return

        removed = undo_decision(gold_session, doi_a=args.doi_a, doi_b=args.doi_b)
        if removed:
            print(
                "Removed duplicate decision. Run --stage all to build and publish a newly "
                "versioned dataset."
            )
        else:
            print("No matching duplicate decision exists.")
    finally:
        gold_session.close()
        if silver_session is not None:
            silver_session.close()
