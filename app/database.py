import sqlite3
import os
from contextlib import contextmanager
from app.config import get_settings

settings = get_settings()

DB_PATH = settings.database_url.replace("sqlite:///", "")


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id   TEXT PRIMARY KEY,
            tenant_name TEXT NOT NULL UNIQUE,
            api_key     TEXT NOT NULL UNIQUE,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            filename    TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS request_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id   TEXT NOT NULL,
            endpoint    TEXT NOT NULL,
            query       TEXT,
            latency_ms  REAL,
            tokens_used INTEGER,
            cached      INTEGER DEFAULT 0,
            precision_k REAL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rate_limit (
            api_key    TEXT PRIMARY KEY,
            request_count INTEGER NOT NULL DEFAULT 0,
            window_start  TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
