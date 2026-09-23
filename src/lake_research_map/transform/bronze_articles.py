"""Build lit_bronze.articles from lit_raw -- this is where cross-source
consolidation begins (IEEE CSV + IEEE .bib + Elsevier .bib all become one
common schema). No cross-source dedup or quality filtering yet (silver's
job); pure pagination duplicates within a source collapse naturally via the
`(source, source_id)` unique key.

IEEE reconciliation: the CSV (304 rows / 301 DOIs) is the authoritative
record list; the paginated .bib files only cover ~275 entries and are mostly
a subset. Bronze builds one record per CSV row (enriched with the matching
.bib abstract/keywords when present), plus one record per .bib entry whose
DOI never appears in the CSV -- see CLAUDE.md "counts don't line up".
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.raw_models import BibEntry, IeeeCsvRow
from lake_research_map.ingest.enrichment import (
    load_enrichment_cache,
    load_enrichment_observations,
    merge_enrichment,
)
from lake_research_map.transform.publication_categories import classify_publication


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.strip().casefold() or None


def _strip_braces(text: str) -> str:
    return text.replace("{", "").replace("}", "")


def _split_bibtex_authors(author_field: str | None) -> list[str]:
    if not author_field:
        return []
    return [_strip_braces(a).strip() for a in author_field.split(" and ") if a.strip()]


def _split_ieee_csv_authors(author_field) -> list[str]:
    if not isinstance(author_field, str) or not author_field.strip():
        return []
    return [a.strip() for a in author_field.split(";") if a.strip()]


def _split_keywords(text: str | None, sep: str) -> list[str]:
    if not text:
        return []
    return [k.strip() for k in text.split(sep) if k.strip()]


# US/Canada affiliations end "<city>, <state>, USA", everyone else ends
# "<city>, <country>". Taking the last comma-segment therefore lands on the
# country either way. These are the values that segment produces but that
# aren't countries.
_NON_COUNTRY_TOKENS = {"na", "n/a", "", "-"}


def _split_countries(affiliations: str | None) -> list[str]:
    """Distinct countries from the IEEE CSV's `Author Affiliations`.

    The field is `;`-separated, one entry per author, and each entry ends with
    the country -- so one article yields one country per author, deduplicated
    here because what matters downstream is which countries took part, not how
    many co-authors each contributed. IEEE-only: Elsevier's .bib carries no
    affiliation at all, so any analysis built on this covers ~17% of the corpus
    and must say so.
    """
    if not isinstance(affiliations, str) or not affiliations.strip():
        return []
    countries: list[str] = []
    for affiliation in affiliations.split(";"):
        segments = [s.strip() for s in affiliation.split(",") if s.strip()]
        if not segments:
            continue
        country = segments[-1]
        if country.casefold() in _NON_COUNTRY_TOKENS:
            continue
        if country not in countries:
            countries.append(country)
    return countries


def _parse_online_date(value) -> dt.date | None:
    """Parse the IEEE CSV's `Online Date` (e.g. `24 Jun 2024`).

    This is the only field in the corpus that survives with finer-than-yearly
    resolution -- bronze otherwise keeps just `year` -- so it's what any
    monthly trend has to be built on.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.strptime(value.strip(), "%d %b %Y").date()
    except ValueError:
        return None


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _upsert(session: Session, source: str, source_id: str, **fields) -> None:
    existing = session.scalar(
        select(BronzeArticle).where(
            BronzeArticle.source == source, BronzeArticle.source_id == source_id
        )
    )
    if existing is None:
        session.add(BronzeArticle(source=source, source_id=source_id, **fields))
    else:
        for key, value in fields.items():
            setattr(existing, key, value)


def _source_token(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]


def _classification_fields(
    *,
    record_type: str | None,
    title: str | None,
    venue: str | None,
    document_type: str | None = None,
    note: str | None = None,
) -> dict[str, str]:
    classification = classify_publication(
        record_type=record_type,
        title=title,
        venue=venue,
        document_type=document_type,
        note=note,
    )
    return {
        "publication_category": classification.category,
        "publication_category_basis": classification.basis,
    }


def _build_ieee_records(raw_session: Session, bronze_session: Session) -> int:
    csv_rows = raw_session.scalars(select(IeeeCsvRow)).all()
    bib_entries = raw_session.scalars(select(BibEntry).where(BibEntry.source == "ieee")).all()

    bib_by_doi: dict[str, BibEntry] = {}
    for entry in bib_entries:
        doi = normalize_doi(entry.doi)
        if doi:
            bib_by_doi[doi] = entry

    csv_dois: set[str] = set()
    written = 0

    for row in csv_rows:
        f = row.fields
        doi = normalize_doi(f.get("DOI"))
        if doi:
            csv_dois.add(doi)
        matching_bib = bib_by_doi.get(doi) if doi else None

        keywords = _split_keywords(f.get("Author Keywords"), ";") + _split_keywords(
            f.get("IEEE Terms"), ";"
        )
        abstract = f.get("Abstract")
        if not abstract and matching_bib:
            abstract = matching_bib.fields.get("abstract")
        if not keywords and matching_bib:
            keywords = _split_keywords(matching_bib.fields.get("keywords"), ";")

        start_page = f.get("Start Page")
        end_page = f.get("End Page")
        pages = None
        if start_page and end_page:
            pages = f"{start_page}-{end_page}"
        elif start_page:
            pages = str(start_page)

        record_type = matching_bib.entry_type.lower() if matching_bib else "article"
        title = f.get("Document Title")
        venue = f.get("Publication Title")
        document_type = f.get("Document Identifier") or None
        _upsert(
            bronze_session,
            source="ieee",
            source_id=f"csv:{_source_token(row.source_file)}:{row.row_index}",
            record_type=record_type,
            **_classification_fields(
                record_type=record_type,
                title=title,
                venue=venue,
                document_type=document_type,
                note=matching_bib.fields.get("note") if matching_bib else None,
            ),
            doi=doi,
            title=title,
            authors=_split_ieee_csv_authors(f.get("Authors")),
            year=_to_int(f.get("Publication Year")),
            venue=venue,
            volume=str(f.get("Volume")) if f.get("Volume") is not None else None,
            issue=str(f.get("Issue")) if f.get("Issue") is not None else None,
            pages=pages,
            issn=f.get("ISSN"),
            url=f.get("PDF Link"),
            abstract=abstract,
            keywords=keywords,
            citation_count=_to_int(f.get("Article Citation Count")),
            reference_count=_to_int(f.get("Reference Count")),
            countries=_split_countries(f.get("Author Affiliations")),
            online_date=_parse_online_date(f.get("Online Date")),
            document_type=document_type,
            license=f.get("License") or None,
            raw_csv_id=row.id,
            raw_bib_id=matching_bib.id if matching_bib else None,
        )
        written += 1

    # .bib entries whose DOI never showed up in the CSV -- CLAUDE.md notes the
    # counts genuinely don't line up; keep them rather than silently dropping.
    for entry in bib_entries:
        doi = normalize_doi(entry.doi)
        if doi and doi in csv_dois:
            continue
        f = entry.fields
        record_type = entry.entry_type.lower()
        title = f.get("title")
        venue = f.get("journal") or f.get("booktitle")
        _upsert(
            bronze_session,
            source="ieee",
            source_id=f"bib:{_source_token(entry.source_file)}:{entry.bib_key}",
            record_type=record_type,
            **_classification_fields(
                record_type=record_type,
                title=title,
                venue=venue,
                note=f.get("note"),
            ),
            doi=doi,
            title=title,
            authors=_split_bibtex_authors(f.get("author")),
            year=_to_int(f.get("year")),
            venue=venue,
            volume=f.get("volume"),
            issue=f.get("number"),
            pages=f.get("pages"),
            issn=f.get("ISSN") or f.get("issn"),
            url=f.get("url"),
            abstract=f.get("abstract"),
            keywords=_split_keywords(f.get("keywords"), ";"),
            citation_count=None,
            reference_count=None,
            raw_csv_id=None,
            raw_bib_id=entry.id,
        )
        written += 1

    return written


def _build_elsevier_records(raw_session: Session, bronze_session: Session) -> int:
    bib_entries = raw_session.scalars(select(BibEntry).where(BibEntry.source == "elsevier")).all()
    written = 0
    for entry in bib_entries:
        f = entry.fields
        doi = normalize_doi(entry.doi)
        record_type = entry.entry_type.lower()
        title = f.get("title")
        venue = f.get("journal") or f.get("booktitle")

        _upsert(
            bronze_session,
            source="elsevier",
            source_id=f"bib:{_source_token(entry.source_file)}:{entry.bib_key}",
            record_type=record_type,
            **_classification_fields(
                record_type=record_type,
                title=title,
                venue=venue,
                note=f.get("note"),
            ),
            doi=doi,
            title=title,
            authors=_split_bibtex_authors(f.get("author")),
            year=_to_int(f.get("year")),
            venue=venue,
            volume=f.get("volume"),
            issue=f.get("number"),
            pages=f.get("pages"),
            issn=f.get("issn"),
            url=f.get("url"),
            abstract=f.get("abstract"),
            keywords=_split_keywords(f.get("keywords"), ","),
            citation_count=None,
            reference_count=None,
            raw_csv_id=None,
            raw_bib_id=entry.id,
        )
        written += 1

    return written


def _enrich_citation_counts(bronze_session: Session) -> int:
    """Backfill citation_count/reference_count from the enrichment cache.

    Only fills a field that is currently NULL -- the IEEE CSV's own counts
    are the authoritative source where they exist and must never be
    overwritten by the cache.
    """
    cache = merge_enrichment(load_enrichment_cache(), load_enrichment_observations(bronze_session))
    if not cache:
        return 0

    enriched = 0
    articles = bronze_session.scalars(
        select(BronzeArticle).where(BronzeArticle.doi.in_(cache.keys()))
    ).all()
    for article in articles:
        values = cache.get(article.doi)
        if not values:
            continue
        changed = False
        if article.citation_count is None and values.get("citation_count") is not None:
            article.citation_count = values["citation_count"]
            changed = True
        if article.reference_count is None and values.get("reference_count") is not None:
            article.reference_count = values["reference_count"]
            changed = True
        if changed:
            enriched += 1

    return enriched


def build_bronze_articles(
    raw_session: Session, bronze_session: Session, dataset_version_id: str | None = None
) -> dict[str, int | dict[str, int]]:
    # Bronze is a working projection of the active Raw snapshot. Rebuilding it
    # is the simplest deterministic way to propagate removals and avoids stale
    # rows from the historical upsert-only implementation.
    bronze_session.execute(delete(BronzeArticle))
    written = _build_ieee_records(raw_session, bronze_session)
    written += _build_elsevier_records(raw_session, bronze_session)
    bronze_session.flush()
    if dataset_version_id is not None:
        bronze_session.execute(
            BronzeArticle.__table__.update().values(dataset_version_id=dataset_version_id)
        )
    enriched = _enrich_citation_counts(bronze_session)
    bronze_session.flush()
    categories: dict[str, int] = {}
    for category, count in bronze_session.execute(
        select(BronzeArticle.publication_category, func.count()).group_by(
            BronzeArticle.publication_category
        )
    ):
        categories[str(category)] = int(count)
    return {"written": written, "enriched": enriched, "categories": categories}
