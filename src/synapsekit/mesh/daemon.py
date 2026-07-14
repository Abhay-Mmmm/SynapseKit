"""Daemon facade for the personal knowledge mesh."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from datetime import UTC, datetime

from .._compat import run_sync
from .core import KnowledgeMesh, MeshIndexSummary, MeshQueryResult, MeshStatus

_STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)
# How long ``stop()`` waits for a SIGTERM'd process to actually exit.
_STOP_TIMEOUT_SECONDS = 5.0
_STOP_POLL_SECONDS = 0.05


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
        self._signal_handlers_installed = False

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
            self._install_signal_handlers()
            try:
                await self._watch_loop()
            finally:
                self._remove_signal_handlers()
        return summary

    def start_sync(self, *, watch: bool | None = None, force: bool = False) -> MeshIndexSummary:
        """Sync wrapper for ``start``."""

        return run_sync(self.start(watch=watch, force=force))

    async def stop(self) -> MeshStatus:
        """Stop the daemon.

        If this ``MeshDaemon`` instance is itself the running watch process
        (i.e. its own in-process ``_stop_event`` is being waited on), setting
        the event is sufficient. Otherwise — the common case for the CLI,
        where ``mesh stop`` constructs a brand-new ``MeshDaemon`` — the
        actually-running watch process is signaled by PID via SIGTERM so its
        own ``_watch_loop`` can exit. We then wait briefly for the PID's
        status row to flip to ``stopped`` before giving up and marking it
        stopped ourselves.
        """

        self._stop_event.set()

        pid = self.mesh.store.status_value("pid")
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            signaled = self._signal_pid(pid)
            if signaled:
                await self._wait_for_stopped(pid)

        # Ensure the status row reflects "stopped" even if the remote
        # process could not be signaled (e.g. already exited) or did not
        # update its own status in time.
        if self.mesh.store.status_value("state") != "stopped":
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
            if self._stop_event.is_set():
                break
            await self.mesh.reindex()
        self.mesh.store.update_status(state="stopped", pid=None)

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers that set ``_stop_event``.

        Runs only on the process actually executing the watch loop, so a
        cross-process ``os.kill(pid, SIGTERM)`` from ``mesh stop`` reaches a
        handler that can stop this daemon's own asyncio event loop.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self._stop_event.set)
        self._signal_handlers_installed = True

    def _remove_signal_handlers(self) -> None:
        if not self._signal_handlers_installed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(sig)
        self._signal_handlers_installed = False

    @staticmethod
    def _signal_pid(pid: int) -> bool:
        """Send SIGTERM to ``pid``. Returns ``False`` if it no longer exists."""

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False
        return True

    async def _wait_for_stopped(self, pid: int) -> None:
        """Poll the status row until it reflects ``stopped`` or the PID exits."""

        elapsed = 0.0
        while elapsed < _STOP_TIMEOUT_SECONDS:
            if self.mesh.store.status_value("state") == "stopped":
                return
            if not _pid_alive(pid):
                return
            await asyncio.sleep(_STOP_POLL_SECONDS)
            elapsed += _STOP_POLL_SECONDS


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
