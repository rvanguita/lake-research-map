"""Database timestamp helpers shared by all medallion layers."""

from __future__ import annotations

import datetime as dt


def naive_utc_now() -> dt.datetime:
    """Return UTC without tzinfo for the project's MySQL DATETIME columns."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)
