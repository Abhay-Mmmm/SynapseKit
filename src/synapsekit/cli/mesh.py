"""``synapsekit mesh`` commands for the personal knowledge mesh."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from synapsekit.mesh import KnowledgeMesh, MeshConfig, MeshDaemon, MeshDaemonConfig


def _mesh_config(args: Any) -> MeshConfig:
    defaults = MeshConfig()
    roots = getattr(args, "root", None) or [str(Path.cwd())]
    include = getattr(args, "include", None)
    return MeshConfig(
        roots=list(roots),
        include=list(include) if include else list(defaults.include),
        ignore_file=getattr(args, "ignore_file", None) or defaults.ignore_file,
        state_dir=getattr(args, "state_dir", None) or defaults.state_dir,
        db_path=getattr(args, "db", None),
        graph_path=getattr(args, "graph_db", None),
        vector_backend=getattr(args, "vector_backend", "auto"),
        graph_backend=getattr(args, "graph_backend", defaults.graph_backend),
        use_git=not bool(getattr(args, "no_git", False)),
        include_git_history=not bool(getattr(args, "no_git_history", False)),
        retrieval_top_k=int(getattr(args, "top_k", 5) or 5),
    )


def _mesh(args: Any) -> KnowledgeMesh:
    return KnowledgeMesh(_mesh_config(args))


def _print_payload(payload: Any, *, output_json: bool = False) -> None:
    if output_json:
        print(json.dumps(payload, indent=2))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    print(payload)


def run_mesh(args: Any) -> None:
    """Dispatch ``synapsekit mesh`` subcommands."""

    command = getattr(args, "mesh_command", None)
    output_json = bool(getattr(args, "json", False))
    mesh = _mesh(args)

    if command == "start":
        daemon = MeshDaemon(
            mesh,
            config=MeshDaemonConfig(
                poll_interval=float(getattr(args, "poll_interval", 5.0)),
                watch=bool(getattr(args, "watch", False)),
            ),
        )
        summary = daemon.start_sync(watch=bool(getattr(args, "watch", False)))
        _print_payload(asdict(summary), output_json=output_json)
        return

    if command == "stop":
        status = MeshDaemon(mesh).stop_sync()
        _print_payload(asdict(status), output_json=output_json)
        return

    if command == "status":
        _print_payload(asdict(mesh.status()), output_json=output_json)
        return

    if command == "reindex":
        summary = mesh.reindex_sync(force=bool(getattr(args, "force", False)))
        _print_payload(asdict(summary), output_json=output_json)
        return

    if command == "query":
        result = mesh.query_sync(str(getattr(args, "query", "")), top_k=getattr(args, "top_k", 5))
        payload = {
            "answer": result.answer,
            "hits": [
                {
                    "path": hit.path,
                    "line_start": hit.line_start,
                    "line_end": hit.line_end,
                    "score": hit.score,
                    "headings": list(hit.headings),
                    "text": hit.text,
                }
                for hit in result.hits
            ],
            "graph_entities": list(result.graph_entities),
        }
        if output_json:
            _print_payload(payload, output_json=True)
        else:
            print(result.answer)
        return

    if command == "mcp":
        from synapsekit.mcp import MCPServer

        MCPServer(name="synapsekit-mesh", tools=mesh.as_mcp_tools()).run()
        return

    raise SystemExit("Missing mesh action. Use start, stop, status, reindex, query, or mcp.")


def build_mesh_parser(subparsers: Any) -> None:
    """Register the ``mesh`` parser with the top-level CLI."""

    parser = subparsers.add_parser("mesh", help="Index and query local project knowledge")
    parser.add_argument("--root", action="append", help="Root directory to index")
    parser.add_argument("--include", action="append", help="Glob pattern to include")
    parser.add_argument("--ignore-file", default=None, help="Gitignore-style mesh ignore file")
    parser.add_argument("--state-dir", default=None, help="Mesh state directory")
    parser.add_argument("--db", default=None, help="sqlite-vec database path")
    parser.add_argument("--graph-db", default=None, help="Kuzu graph database path")
    parser.add_argument(
        "--graph-backend",
        choices=["auto", "memory", "kuzu"],
        default="memory",
        help="Graph backend",
    )
    parser.add_argument(
        "--vector-backend",
        choices=["auto", "memory", "sqlite_vec"],
        default="auto",
        help="Vector backend",
    )
    parser.add_argument("--no-git", action="store_true", help="Do not discover git repos")
    parser.add_argument(
        "--no-git-history",
        action="store_true",
        help="Do not index recent git commit subjects",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    mesh_sub = parser.add_subparsers(dest="mesh_command")

    start_cmd = mesh_sub.add_parser("start", help="Start and index the mesh")
    start_cmd.add_argument("--watch", action="store_true", help="Keep polling for changes")
    start_cmd.add_argument("--poll-interval", type=float, default=5.0, help="Watch poll interval")

    mesh_sub.add_parser("stop", help="Mark the mesh daemon stopped")
    mesh_sub.add_parser("status", help="Show mesh status")

    reindex_cmd = mesh_sub.add_parser("reindex", help="Incrementally reindex changed files")
    reindex_cmd.add_argument("--force", action="store_true", help="Reindex every discovered file")

    query_cmd = mesh_sub.add_parser("query", help="Query the mesh")
    query_cmd.add_argument("query", help="Search question")
    query_cmd.add_argument("--top-k", type=int, default=5, help="Maximum ranked hits")

    mesh_sub.add_parser("mcp", help="Run mesh MCP server over stdio")
