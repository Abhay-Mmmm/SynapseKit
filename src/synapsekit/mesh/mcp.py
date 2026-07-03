"""MCP tools for exposing a knowledge mesh."""

from __future__ import annotations

import json
from typing import Any

from ..agents.base import BaseTool, ToolResult
from .core import KnowledgeMesh


class MeshQueryTool(BaseTool):
    """Query a ``KnowledgeMesh`` from MCP."""

    name = "mesh_query"
    description = "Query the local personal knowledge mesh with file and line citations."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or search query."},
            "top_k": {"type": "integer", "description": "Maximum ranked hits to return."},
        },
        "required": ["query"],
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        top_k = kwargs.get("top_k")
        result = await self.mesh.query(
            str(kwargs.get("query", "")),
            top_k=int(top_k) if top_k is not None else None,
        )
        return ToolResult(
            output=json.dumps(
                {
                    "answer": result.answer,
                    "hits": [hit_to_dict(hit) for hit in result.hits],
                    "graph_entities": list(result.graph_entities),
                },
                indent=2,
            )
        )


class MeshReindexTool(BaseTool):
    """Reindex a ``KnowledgeMesh`` from MCP."""

    name = "mesh_reindex"
    description = "Incrementally reindex changed local mesh documents."
    parameters = {
        "type": "object",
        "properties": {
            "force": {"type": "boolean", "description": "Reindex all discovered files."}
        },
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        summary = await self.mesh.reindex(force=bool(kwargs.get("force", False)))
        return ToolResult(output=json.dumps(summary.__dict__, indent=2))


class MeshDuplicatesTool(BaseTool):
    """Find duplicates in a ``KnowledgeMesh`` from MCP."""

    name = "mesh_duplicates"
    description = "Find likely duplicate concepts or snippets across indexed projects."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum duplicate matches to return."}
        },
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        matches = self.mesh.duplicates(limit=int(kwargs.get("limit", 20)))
        return ToolResult(output=json.dumps([match.__dict__ for match in matches], indent=2))


class MeshStatusTool(BaseTool):
    """Return ``KnowledgeMesh`` status from MCP."""

    name = "mesh_status"
    description = "Return local knowledge mesh indexing status."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=json.dumps(self.mesh.status().__dict__, indent=2))


def build_mesh_tools(mesh: KnowledgeMesh) -> list[BaseTool]:
    """Return all MCP tools for ``mesh``."""

    return [
        MeshQueryTool(mesh),
        MeshReindexTool(mesh),
        MeshDuplicatesTool(mesh),
        MeshStatusTool(mesh),
    ]


def hit_to_dict(hit: Any) -> dict[str, Any]:
    """Serialize a mesh hit without requiring callers to import dataclasses."""

    return {
        "text": hit.text,
        "score": hit.score,
        "path": hit.path,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "headings": list(hit.headings),
        "repo_root": hit.repo_root,
        "metadata": hit.metadata,
    }
