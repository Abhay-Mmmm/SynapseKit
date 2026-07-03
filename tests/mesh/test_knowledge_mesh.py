from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit import KnowledgeMesh, LocalMdLoader, MeshConfig


def test_knowledge_mesh_incremental_query_and_duplicates(tmp_path: Path) -> None:
    async def run() -> None:
        root = tmp_path / "workspace"
        state = tmp_path / "state"
        root.mkdir()
        (root / "auth.md").write_text(
            "# Auth\n\nAuthService uses FastAPI middleware for login.\n",
            encoding="utf-8",
        )
        retry_text = "# Retry\n\nRetry decorator handles exponential backoff for transient requests.\n"
        (root / "retry-a.md").write_text(retry_text, encoding="utf-8")
        (root / "retry-b.md").write_text(retry_text, encoding="utf-8")

        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[root],
                state_dir=state,
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        first = await mesh.reindex()
        second = await mesh.reindex()
        result = await mesh.query("FastAPI auth middleware", top_k=2)
        duplicates = mesh.duplicates()
        status = mesh.status()

        assert first.discovered_files == 3
        assert first.ingested_chunks >= 3
        assert second.changed_chunks == 0
        assert second.skipped_chunks == first.discovered_chunks
        assert result.hits
        assert result.hits[0].path.endswith("auth.md")
        assert result.hits[0].line_start is not None
        assert duplicates
        assert status.active_chunks >= 3
        assert status.offline_default is True

    asyncio.run(run())


def test_top_level_mesh_exports() -> None:
    assert KnowledgeMesh is not None
    assert LocalMdLoader is not None
