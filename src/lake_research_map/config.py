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

    The same file has a different absolute path depending on where the
    pipeline runs -- `/home/<user>/.../data/ieee/x.csv` on the host vs
    `/opt/airflow/project/data/ieee/x.csv` inside the Airflow container. Since
    raw-layer idempotency keys off this string, storing the absolute path made
    the same file look like two different files and duplicated every row on a
    cross-environment run. Relative to REPO_ROOT it is identical in both.
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
    ("elsevier", ELSEVIER_DIR / "config.csv"),
    ("elsevier", DATA_DIR / "config.csv"),
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
        role = os.environ.get("LAKE_RESEARCH_MAP_DB_ROLE", "pipeline").strip().upper()
        return cls(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get(f"MYSQL_{role}_USER", os.environ.get("MYSQL_USER", "root")),
            password=os.environ.get(f"MYSQL_{role}_PASSWORD", os.environ.get("MYSQL_PASSWORD", "")),
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


def get_airflow_base_url() -> str:
    """Base URL of the Airflow webserver the dashboard triggers DAG runs on.

    Defaults to localhost:8080 (running `streamlit run main.py` on the host,
    with docker compose publishing Airflow's port there). Inside the compose
    network, the `dashboard` service overrides this to `http://airflow:8080`.
    """
    return os.environ.get("AIRFLOW_BASE_URL", "http://localhost:8080")
