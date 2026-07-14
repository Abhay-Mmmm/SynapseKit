from __future__ import annotations

import sqlite3
import threading
from contextlib import suppress
from typing import Any

from ._cache import AsyncLRUCache


class SQLiteLLMCache:
    """Persistent LLM cache backed by SQLite.

    Stores cache entries on disk so they survive process restarts.
    Uses the same ``make_key`` logic as :class:`AsyncLRUCache`.

    Implements the context manager protocol so the underlying connection is
    always closed, even if an exception occurs:

    Usage::

        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        with SQLiteLLMCache("llm_cache.db") as cache:
            cache.put(key, value)
            cached = cache.get(key)

        # Or without context manager — close() is called by __del__ as fallback
        cache = SQLiteLLMCache("llm_cache.db")
        cache.put(key, value)
        cached = cache.get(key)
        cache.close()
    """

    make_key = staticmethod(AsyncLRUCache.make_key)

    def __init__(
        self, db_path: str = "synapsekit_llm_cache.db", busy_timeout_ms: int = 5000
    ) -> None:
        self._db_path = db_path
        # A single shared connection is reused from async handlers that may run
        # on different threads (executor pool), so allow cross-thread use and
        # guard every access with a lock. WAL + a busy timeout let concurrent
        # writers wait instead of failing with "database is locked".
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._conn.commit()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        with suppress(Exception):
            self._conn.close()

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> SQLiteLLMCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Last-resort cleanup if the caller forgets to call close()."""
        with suppress(Exception):
            self.close()
