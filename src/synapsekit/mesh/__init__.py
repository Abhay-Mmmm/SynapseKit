"""Personal Knowledge Mesh public API."""

from __future__ import annotations

from .core import (
    KnowledgeMesh,
    LocalMeshLLM,
    MeshConfig,
    MeshHit,
    MeshIndexStore,
    MeshIndexSummary,
    MeshQueryResult,
    MeshStatus,
)
from .daemon import MeshDaemon, MeshDaemonConfig
from .embeddings import HashingEmbeddings
from .loaders import DEFAULT_MESH_INCLUDES, GitRepoLoader, LocalMdLoader, MarkdownChunk
from .mcp import (
    MeshDuplicatesTool,
    MeshQueryTool,
    MeshReindexTool,
    MeshStatusTool,
    build_mesh_tools,
)
from .privacy import DEFAULT_MESH_IGNORE, MeshPrivacyFilter, PrivacyDecision
from .resolution import CrossProjectEntityResolver, DuplicationDetector, DuplicationMatch

__all__ = [
    "DEFAULT_MESH_IGNORE",
    "DEFAULT_MESH_INCLUDES",
    "CrossProjectEntityResolver",
    "DuplicationDetector",
    "DuplicationMatch",
    "GitRepoLoader",
    "HashingEmbeddings",
    "KnowledgeMesh",
    "LocalMdLoader",
    "LocalMeshLLM",
    "MarkdownChunk",
    "MeshConfig",
    "MeshDaemon",
    "MeshDaemonConfig",
    "MeshDuplicatesTool",
    "MeshHit",
    "MeshIndexStore",
    "MeshIndexSummary",
    "MeshPrivacyFilter",
    "MeshQueryResult",
    "MeshQueryTool",
    "MeshReindexTool",
    "MeshStatus",
    "MeshStatusTool",
    "PrivacyDecision",
    "build_mesh_tools",
]
