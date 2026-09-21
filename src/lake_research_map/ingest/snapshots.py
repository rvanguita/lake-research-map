"""Immutable source snapshots and deterministic dataset fingerprints.

The operational Raw tables remain a projection of the current working
version.  This module preserves the evidence needed to reproduce that
projection: every consumed byte is copied once into a content-addressed
archive and every logical version stores an ordered path-to-revision manifest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lake_research_map.config import DATA_DIR, REPO_ROOT

FINGERPRINT_SCHEMA_VERSION = 1
ARCHIVE_DIR = DATA_DIR / ".lake_research_map" / "objects"


@dataclass(frozen=True, slots=True)
class ScannedSource:
    path: str
    source: str
    kind: str
    sha256: str
    size_bytes: int
    mtime: dt.datetime
    revision_id: str
    archive_path: str


@dataclass(frozen=True, slots=True)
class VersionFingerprint:
    version_id: str
    source_manifest_sha256: str
    config_sha256: str
    code_revision: str | None
    code_sha256: str
    curation_sha256: str


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _supported_paths(data_dir: Path) -> list[tuple[Path, str, str]]:
    paths: list[tuple[Path, str, str]] = []
    for source in ("ieee", "elsevier"):
        directory = data_dir / source
        config = directory / "config.csv"
        if config.is_file():
            paths.append((config, source, "config"))
        paths.extend((path, source, "bib") for path in sorted(directory.glob("*.bib")))
    paths.extend((path, "ieee", "csv") for path in sorted((data_dir / "ieee").glob("export*.csv")))
    paths.extend(
        (path, "articles", "pdf") for path in sorted((data_dir / "articles").glob("*.pdf"))
    )
    enrichment = data_dir / "enrichment_cache.json"
    if enrichment.is_file():
        paths.append((enrichment, "openalex", "enrichment"))
    return sorted(paths, key=lambda item: _relative(item[0], data_dir))


def _archive_object(path: Path, sha256: str, archive_dir: Path) -> Path:
    destination = archive_dir / sha256[:2] / sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != path.stat().st_size:
            raise OSError(f"archived object has an invalid size: {destination}")
        return destination

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{sha256}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, path.open("rb") as source:
            shutil.copyfileobj(source, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        if _sha256_file(temporary) != sha256:
            raise OSError(f"archive checksum mismatch for {path}")
        os.replace(temporary, destination)
        destination.chmod(0o444)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination


def scan_sources(
    data_dir: Path = DATA_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    archive_dir: Path | None = None,
) -> list[ScannedSource]:
    """Hash and archive every source that can affect the medallion pipeline."""
    archive_root = archive_dir or data_dir / ".lake_research_map" / "objects"
    scanned: list[ScannedSource] = []
    for path, source, kind in _supported_paths(data_dir):
        sha256 = _sha256_file(path)
        archived = _archive_object(path, sha256, archive_root)
        stored_path = _relative(path, repo_root)
        revision_id = _stable_hash({"path": stored_path, "sha256": sha256})
        stat = path.stat()
        scanned.append(
            ScannedSource(
                path=stored_path,
                source=source,
                kind=kind,
                sha256=sha256,
                size_bytes=stat.st_size,
                mtime=dt.datetime.fromtimestamp(stat.st_mtime),
                revision_id=revision_id,
                archive_path=_relative(archived, repo_root),
            )
        )
    return sorted(scanned, key=lambda item: item.path)


def manifest_hash(sources: list[ScannedSource]) -> str:
    manifest = [
        {
            "path": item.path,
            "source": item.source,
            "kind": item.kind,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in sorted(sources, key=lambda item: item.path)
    ]
    return _stable_hash(manifest)


def config_hash(sources: list[ScannedSource]) -> str:
    return _stable_hash(
        [{"path": item.path, "sha256": item.sha256} for item in sources if item.kind == "config"]
    )


def code_fingerprint(repo_root: Path = REPO_ROOT) -> tuple[str | None, str]:
    """Hash transformation code and locked dependencies, including dirty edits."""
    paths = sorted((repo_root / "src").rglob("*.py"))
    paths.extend(
        path for path in (repo_root / "pyproject.toml", repo_root / "uv.lock") if path.exists()
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(_relative(path, repo_root).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return revision or None, digest.hexdigest()


def build_fingerprint(
    sources: list[ScannedSource],
    *,
    curation_records: list[dict[str, Any]] | None = None,
    repo_root: Path = REPO_ROOT,
) -> VersionFingerprint:
    source_sha = manifest_hash(sources)
    config_sha = config_hash(sources)
    code_revision, code_sha = code_fingerprint(repo_root)
    curation_sha = _stable_hash(curation_records or [])
    version_id = _stable_hash(
        {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "source_manifest_sha256": source_sha,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
            "curation_sha256": curation_sha,
        }
    )
    return VersionFingerprint(
        version_id=version_id,
        source_manifest_sha256=source_sha,
        config_sha256=config_sha,
        code_revision=code_revision,
        code_sha256=code_sha,
        curation_sha256=curation_sha,
    )


def source_as_dict(source: ScannedSource) -> dict[str, Any]:
    return asdict(source)


def diff_manifests(
    previous: dict[str, tuple[str | None, str]], current: dict[str, ScannedSource]
) -> list[dict[str, str | None]]:
    """Return deterministic per-path changes, recognizing same-hash renames."""
    removed = {path: value for path, value in previous.items() if path not in current}
    added = {path: source for path, source in current.items() if path not in previous}

    changes: list[dict[str, str | None]] = []
    added_by_hash: dict[str, list[str]] = {}
    for path, source in added.items():
        added_by_hash.setdefault(source.sha256, []).append(path)

    renamed_added: set[str] = set()
    renamed_removed: set[str] = set()
    for old_path, (old_revision, old_hash) in removed.items():
        candidates = added_by_hash.get(old_hash, [])
        if candidates:
            new_path = candidates.pop(0)
            renamed_added.add(new_path)
            renamed_removed.add(old_path)
            changes.append(
                {
                    "path": new_path,
                    "change_type": "renamed",
                    "previous_revision_id": old_revision,
                    "current_revision_id": current[new_path].revision_id,
                }
            )

    for path in sorted(set(previous) | set(current)):
        if path in renamed_added or path in renamed_removed:
            continue
        old_value = previous.get(path)
        old_revision = old_value[0] if old_value else None
        old_hash = old_value[1] if old_value else None
        new_revision = current.get(path).revision_id if path in current else None
        if old_revision is None:
            kind = "added"
        elif new_revision is None:
            kind = "removed"
        elif old_hash != current[path].sha256:
            kind = "modified"
        else:
            kind = "unchanged"
        changes.append(
            {
                "path": path,
                "change_type": kind,
                "previous_revision_id": old_revision,
                "current_revision_id": new_revision,
            }
        )
    return sorted(changes, key=lambda item: (str(item["path"]), str(item["change_type"])))
