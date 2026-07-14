from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit import KnowledgeMesh, LocalMdLoader, MeshConfig
from synapsekit.mesh.core import _hits_from_embeddings


def test_knowledge_mesh_incremental_query_and_duplicates(tmp_path: Path) -> None:
    async def run() -> None:
        root = tmp_path / "workspace"
        state = tmp_path / "state"
        root.mkdir()
        (root / "auth.md").write_text(
            "# Auth\n\nAuthService uses FastAPI middleware for login.\n",
            encoding="utf-8",
        )
        retry_text = (
            "# Retry\n\nRetry decorator handles exponential backoff for transient requests.\n"
        )
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


def _embedding_row(chunk_id: str) -> dict:
    return {
        "text": f"text for {chunk_id}",
        "score": 1.0,
        "metadata": {"chunk_id": chunk_id, "path": f"/tmp/{chunk_id}.md"},
    }


def test_hits_from_embeddings_none_means_no_filter() -> None:
    # active_chunk_ids=None means "no active-chunk filter is in effect" — every
    # embedding row that has a path should be returned as a hit.
    embeddings = [_embedding_row("a"), _embedding_row("b")]

    hits = _hits_from_embeddings(embeddings, None, limit=10)

    assert {hit.metadata["chunk_id"] for hit in hits} == {"a", "b"}


def test_hits_from_embeddings_empty_set_means_nothing_active() -> None:
    # Regression test: an empty active_chunk_ids set previously was treated
    # as falsy (same as "no filter"), so every embedding row leaked through
    # even though the index has zero active chunks. An empty *active* set is
    # a real filter state — nothing should match.
    embeddings = [_embedding_row("a"), _embedding_row("b")]

    hits = _hits_from_embeddings(embeddings, set(), limit=10)

    assert hits == []


def test_hits_from_embeddings_filters_to_active_only() -> None:
    embeddings = [_embedding_row("a"), _embedding_row("b"), _embedding_row("c")]

    hits = _hits_from_embeddings(embeddings, {"b"}, limit=10)

    assert [hit.metadata["chunk_id"] for hit in hits] == ["b"]


def test_mesh_query_returns_no_hits_when_index_freshly_empty(tmp_path: Path) -> None:
    # End-to-end regression: querying a mesh with zero active chunks must not
    # surface stale/foreign embeddings just because active_chunk_ids is empty.
    async def run() -> None:
        root = tmp_path / "workspace"
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
        # Ingest directly into the vector store, bypassing mesh.reindex(), so
        # the SQLite active_chunk_ids index stays empty while the vector
        # store has real content — this reproduces the bug scenario where
        # active_chunk_ids() legitimately returns an empty set.
        await mesh.rag.ingest(
            [
                {
                    "text": "AuthService uses FastAPI middleware for login.",
                    "metadata": {
                        "path": str(root / "auth.md"),
                        "source": str(root / "auth.md"),
                        "chunk_id": "orphan_chunk",
                    },
                }
            ]
        )

        result = await mesh.query("FastAPI auth middleware")

        assert result.hits == []

    asyncio.run(run())


def test_reindex_deletes_stale_vectors_on_content_change(tmp_path: Path) -> None:
    # Regression test for the reindex leak: when a file's content changes,
    # the old content-hash-derived chunk_id becomes inactive in SQLite, but
    # the corresponding stale vector must also be removed from the vector
    # store — otherwise queries can keep surfacing text that no longer
    # exists in the source file (unbounded leak over repeated edits).
    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        doc = root / "notes.md"
        doc.write_text("# Notes\n\nOriginalUniqueMarkerZZZ appears here.\n", encoding="utf-8")

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
        vector_store = mesh.rag.vector_store
        assert len(vector_store) >= 1

        # Change the file content so a new content-hash and new chunk_id are
        # generated; the old chunk_id is deactivated during reindex.
        doc.write_text("# Notes\n\nReplacedUniqueMarkerYYY appears here.\n", encoding="utf-8")
        await mesh.reindex()

        active_ids = mesh.store.active_chunk_ids()
        metadatas = list(vector_store._metadata)

        # No vector-store row should reference a chunk_id that is no longer
        # active — this is the core leak the fix addresses.
        stale = [m for m in metadatas if m.get("chunk_id") not in active_ids]
        assert stale == []

        # The old text must be gone and the new text must be present.
        texts = vector_store._texts
        assert not any("OriginalUniqueMarkerZZZ" in text for text in texts)
        assert any("ReplacedUniqueMarkerYYY" in text for text in texts)

    asyncio.run(run())
