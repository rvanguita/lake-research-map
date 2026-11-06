"""Environment configuration for the medallion pipeline.

Reads MySQL connection details from `.env` and exposes one database name /
connection URL per medallion layer (raw, bronze, silver, gold). Every layer
lives in its own MySQL database, named plainly after the layer itself (e.g.
`bronze`) -- note these databases are shared with unrelated tables from other
projects on the same MySQL server; the pipeline only ever touches its own
`lit_*`-prefixed tables within them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LAYERS = ("raw", "bronze", "silver", "gold")


def relative_path(path) -> str:
    """Path relative to the repo root, as stored in the database.

    The same file has a different absolute path in every checkout and inside
    any container (`/home/<user>/.../data/references/x.bib` on one machine,
    `/opt/project/data/references/x.bib` in an image). Since raw-layer
    idempotency keys off this string, storing the absolute path made the same
    file look like two different files and duplicated every row on a
    cross-environment run -- it happened when the pipeline also ran inside a
    container. Relative to REPO_ROOT it is identical everywhere.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:  # outside the repo -- keep it absolute
        return str(resolved)


def absolute_path(stored_path: str) -> Path:
    """Inverse of `relative_path`, for actually opening a stored file."""
    path = Path(stored_path)
    return path if path.is_absolute() else REPO_ROOT / path


DATA_DIR = REPO_ROOT / "data"
IEEE_DIR = DATA_DIR / "ieee"
ELSEVIER_DIR = DATA_DIR / "elsevier"
ARTICLES_DIR = DATA_DIR / "articles"
REFERENCES_DIR = DATA_DIR / "references"

# Both layouts remain supported. The original exports lived directly under
# data/{ieee,elsevier}; newer downloads are grouped under data/references with
# their publisher-facing directory names. Keeping both paths lets an append
# run retain historical evidence while ingesting the new batches in place.
IEEE_BIB_DIRS = (IEEE_DIR, REFERENCES_DIR / "IEEE Xplore")
ELSEVIER_BIB_DIRS = (ELSEVIER_DIR, REFERENCES_DIR / "Science Direct")
SEARCH_CONFIG_PATHS = (
    ("ieee", IEEE_DIR / "config.csv"),
    ("ieee", REFERENCES_DIR / "IEEE Xplore" / "config.csv"),
    ("elsevier", ELSEVIER_DIR / "config.csv"),
    ("elsevier", DATA_DIR / "config.csv"),
    ("elsevier", REFERENCES_DIR / "Science Direct" / "config.csv"),
)

# Official CAPES/Qualis journal-classification export (2017-2020 quadriênio, all
# evaluation areas) -- see dashboard/qualis.py. Not checked into git (see
# .gitignore): large, externally-sourced reference data, not project-authored.
_CAPES_QUALIS_CANDIDATES = (
    DATA_DIR / "capes-qualis.xlsx",
    DATA_DIR / "classificações_publicadas_todas_as_areas_avaliacao1783453567318.xlsx",
)
CAPES_QUALIS_XLSX = next(
    (path for path in _CAPES_QUALIS_CANDIDATES if path.exists()),
    _CAPES_QUALIS_CANDIDATES[0],
)


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str

    @classmethod
    def from_env(cls) -> MySQLSettings:
        """Read the one database identity this project has.

        There used to be per-role users (`MYSQL_PIPELINE_*`, `MYSQL_DASHBOARD_*`)
        selected by `LAKE_RESEARCH_MAP_DB_ROLE`. They were dropped: this is a
        single-machine deployment, so the isolation they bought did not pay for
        the failure mode they introduced -- user and password resolved
        independently, so setting only one half of a role's pair silently
        connected as the fallback user with the role's credential.
        """
        return cls(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
        )

    def database_name(self, layer: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, expected one of {LAYERS}")
        return layer

    def server_url(self) -> str:
        """Connection URL with no database selected (for CREATE DATABASE)."""
        return (
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/?charset=utf8mb4"
        )

    def layer_url(self, layer: str) -> str:
        db = self.database_name(layer)
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{db}?charset=utf8mb4"
        )


def get_settings() -> MySQLSettings:
    return MySQLSettings.from_env()


@dataclass(frozen=True)
class QdrantSettings:
    """Where the chunk vectors live (ADR-08).

    `url` has no default on purpose. Every stage and reader that touches vectors
    needs it, and a missing setting should fail naming the variable rather than
    quietly connect to whatever happens to answer on localhost. On the host it
    is `http://127.0.0.1:6333`; inside docker compose the dashboard service
    overrides it to `http://qdrant:6333`.
    """

    url: str
    api_key: str | None
    timeout: int

    @classmethod
    def from_env(cls) -> QdrantSettings:
        return cls(
            url=os.environ.get("QDRANT_URL", "").strip(),
            api_key=os.environ.get("QDRANT_API_KEY", "").strip() or None,
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "60")),
        )


def get_qdrant_settings() -> QdrantSettings:
    return QdrantSettings.from_env()
