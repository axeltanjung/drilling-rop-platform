"""SQLite database access via SQLAlchemy engine."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("database")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    return engine


def init_db() -> None:
    """Ensure the database file/dir exists. Tables are created on write."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Database ready: %s", settings.database_url)


def table_exists(name: str) -> bool:
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT 1 FROM {name} LIMIT 1"))
        return True
    except Exception:
        return False


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})
