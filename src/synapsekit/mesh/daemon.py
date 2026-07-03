"""Daemon facade for the personal knowledge mesh."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from .._compat import run_sync
from .core import KnowledgeMesh, MeshIndexSummary, MeshQueryResult, MeshStatus


@dataclass(frozen=True)
class MeshDaemonConfig:
    """Runtime options for ``MeshDaemon``."""

    poll_interval: float = 5.0
    watch: bool = False


class MeshDaemon:
    """Manage mesh indexing and optional polling-based filesystem watching."""

    def __init__(
        self,
        mesh: KnowledgeMesh | None = None,
        *,
        config: MeshDaemonConfig | None = None,
    ) -> None:
        self.mesh = mesh or KnowledgeMesh()
        self.config = config or MeshDaemonConfig()
        self._stop_event = asyncio.Event()

    async def start(self, *, watch: bool | None = None, force: bool = False) -> MeshIndexSummary:
        """Start the daemon and run an initial incremental index."""

        active_watch = self.config.watch if watch is None else watch
        self.mesh.store.update_status(
            state="running" if active_watch else "ready",
            pid=os.getpid(),
            started_at=datetime.now(UTC).isoformat(),
        )
        summary = await self.mesh.reindex(force=force)
        if active_watch:
            await self._watch_loop()
        return summary

    def start_sync(self, *, watch: bool | None = None, force: bool = False) -> MeshIndexSummary:
        """Sync wrapper for ``start``."""

        return run_sync(self.start(watch=watch, force=force))

    async def stop(self) -> MeshStatus:
        """Stop the daemon and mark the mesh as stopped."""

        self._stop_event.set()
        self.mesh.store.update_status(state="stopped", pid=None)
        return self.mesh.status()

    def stop_sync(self) -> MeshStatus:
        """Sync wrapper for ``stop``."""

        return run_sync(self.stop())

    def status(self) -> MeshStatus:
        """Return current daemon status."""

        return self.mesh.status()

    async def reindex(self, *, force: bool = False) -> MeshIndexSummary:
        """Run an incremental reindex."""

        return await self.mesh.reindex(force=force)

    async def query(self, query: str, *, top_k: int | None = None) -> MeshQueryResult:
        """Query the underlying mesh."""

        return await self.mesh.query(query, top_k=top_k)

    async def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.poll_interval)
            await self.mesh.reindex()
