from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from synapsekit.mesh import KnowledgeMesh, MeshConfig, MeshDaemon, MeshDaemonConfig

_WATCH_SCRIPT = """
import sys
from synapsekit.mesh import KnowledgeMesh, MeshConfig, MeshDaemon, MeshDaemonConfig

root = sys.argv[1]
state = sys.argv[2]

mesh = KnowledgeMesh(
    MeshConfig(
        roots=[root],
        state_dir=state,
        vector_backend="memory",
        graph_backend="memory",
        use_git=False,
    )
)
daemon = MeshDaemon(mesh, config=MeshDaemonConfig(poll_interval=0.1, watch=True))
daemon.start_sync(watch=True)
"""


def _wait_until(predicate, *, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_mesh_stop_signals_the_actual_running_watch_process(tmp_path: Path) -> None:
    # Regression test for the cross-process stop bug: "synapsekit mesh stop"
    # used to construct a brand-new MeshDaemon and only flip a SQLite status
    # row, never touching the asyncio.Event inside the *actually running*
    # watch-loop process, so the real process never exited. Here we spawn a
    # real watch daemon as a subprocess and confirm a freshly constructed
    # MeshDaemon().stop_sync() (mirroring what the CLI does) makes that
    # separate process actually terminate.
    root = tmp_path / "root"
    root.mkdir()
    (root / "README.md").write_text("# Notes\n", encoding="utf-8")
    state = tmp_path / "state"

    script_path = tmp_path / "watch_daemon.py"
    script_path.write_text(_WATCH_SCRIPT, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script_path), str(root), str(state)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        mesh_for_status = KnowledgeMesh(
            MeshConfig(
                roots=[root],
                state_dir=state,
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        # Wait for the subprocess to record its PID in the shared status row,
        # which only happens once MeshDaemon.start() has begun.
        started = _wait_until(lambda: mesh_for_status.store.status_value("pid") is not None)
        assert started, "watch daemon subprocess never reported a pid in status"

        recorded_pid = mesh_for_status.store.status_value("pid")
        assert recorded_pid == proc.pid

        # This mirrors exactly what `synapsekit mesh stop` does: build a
        # brand-new MeshDaemon over the same state dir and call stop_sync().
        stopper = MeshDaemon(mesh_for_status)
        status = stopper.stop_sync()

        assert status.state == "stopped"

        # The critical assertion: the *actual subprocess* must have exited,
        # not just the status row flipped while the real process lives on.
        exited = _wait_until(lambda: proc.poll() is not None, timeout=10.0)
        assert exited, "watch daemon subprocess did not exit after mesh stop"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_mesh_daemon_start_records_pid_in_status(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[root],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )
    daemon = MeshDaemon(mesh, config=MeshDaemonConfig(watch=False))
    daemon.start_sync(watch=False)

    assert mesh.store.status_value("pid") == os.getpid()


def test_sigterm_sets_stop_event_via_installed_handler(tmp_path: Path) -> None:
    # Unit-level check of the signal wiring itself: installing the daemon's
    # signal handlers and then delivering a real SIGTERM to this test
    # process must flip the in-process _stop_event, independent of any
    # subprocess machinery.
    import asyncio

    root = tmp_path / "root"
    root.mkdir()
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[root],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )
    daemon = MeshDaemon(mesh, config=MeshDaemonConfig(watch=True))

    async def run() -> None:
        daemon._install_signal_handlers()
        try:
            assert not daemon._stop_event.is_set()
            os.kill(os.getpid(), signal.SIGTERM)
            # Give the event loop a tick to process the delivered signal.
            await asyncio.sleep(0.1)
            assert daemon._stop_event.is_set()
        finally:
            daemon._remove_signal_handlers()

    asyncio.run(run())
