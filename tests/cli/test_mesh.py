from __future__ import annotations

import argparse
import json
from pathlib import Path

from synapsekit.cli.mesh import run_mesh


def test_mesh_cli_reindex_and_query_json(capsys, tmp_path: Path) -> None:
    root = tmp_path / "root"
    state = tmp_path / "state"
    root.mkdir()
    (root / "README.md").write_text("# Auth\n\nFastAPI middleware notes.", encoding="utf-8")

    run_mesh(_args("reindex", root, state, force=False))
    reindex_payload = json.loads(capsys.readouterr().out)
    assert reindex_payload["ingested_chunks"] == 1

    query_args = _args("query", root, state)
    query_args.query = "FastAPI middleware"
    query_args.top_k = 1
    run_mesh(query_args)
    query_payload = json.loads(capsys.readouterr().out)
    assert query_payload["hits"][0]["path"].endswith("README.md")
    assert query_payload["hits"][0]["line_start"] == 1


def _args(command: str, root: Path, state: Path, **extra) -> argparse.Namespace:
    payload = {
        "mesh_command": command,
        "root": [str(root)],
        "include": None,
        "ignore_file": None,
        "state_dir": str(state),
        "db": None,
        "graph_db": None,
        "vector_backend": "memory",
        "graph_backend": "memory",
        "no_git": True,
        "no_git_history": True,
        "json": True,
        "force": False,
        "watch": False,
        "poll_interval": 5.0,
        "top_k": 5,
    }
    payload.update(extra)
    return argparse.Namespace(**payload)
