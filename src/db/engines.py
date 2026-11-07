"""SQLAlchemy engine/session factories, one per medallion layer."""

from __future__ import annotations

from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import LAYERS, get_settings


@cache
def get_engine(layer: str) -> Engine:
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}, expected one of {LAYERS}")
    settings = get_settings()
    return create_engine(settings.layer_url(layer), pool_pre_ping=True, future=True)


def get_session(layer: str) -> Session:
    Session_ = sessionmaker(bind=get_engine(layer), future=True)
    return Session_()
