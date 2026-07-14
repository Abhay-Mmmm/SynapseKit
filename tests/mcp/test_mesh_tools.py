from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from synapsekit.mcp.server.core import MCPServer
from synapsekit.mesh import KnowledgeMesh, MeshConfig, build_mesh_tools


def test_mesh_mcp_tools_query_and_status(tmp_path: Path) -> None:
    async def run() -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "README.md").write_text("# Auth\n\nFastAPI auth middleware.", encoding="utf-8")
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[root],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )
        await mesh.reindex()

        tools = {tool.name: tool for tool in build_mesh_tools(mesh)}
        query = await tools["mesh_query"].run(query="FastAPI", top_k=1)
        status = await tools["mesh_status"].run()

        assert "mesh_reindex" in tools
        assert json.loads(query.output)["hits"][0]["path"].endswith("README.md")
        assert json.loads(status.output)["active_chunks"] == 1

    asyncio.run(run())


def test_mcp_server_accepts_knowledge_mesh_target(tmp_path: Path) -> None:
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[tmp_path],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )
    registered: dict[str, Any] = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_mod = MagicMock()
    mock_server_mod.Server = MagicMock(return_value=mock_server_inst)
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        MCPServer(mesh)._build_server()

    listed = asyncio.run(registered["list_tools"]())
    assert listed
    asyncio.run(registered["call_tool"](name="mesh_status", arguments={}))
