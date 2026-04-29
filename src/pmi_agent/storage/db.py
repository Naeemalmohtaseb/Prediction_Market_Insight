"""Persistence scaffold."""

from pathlib import Path
from sqlite3 import Connection, connect


def connect_sqlite(path: str | Path) -> Connection:
    """Open a SQLite connection for future local caching."""

    return connect(Path(path))
