from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from synapsekit import KnowledgeMesh, LocalMdLoader, MeshConfig
from synapsekit.mesh.core import MeshIndexStore, _hits_from_embeddings


def _build_mesh(root: Path, state: Path) -> KnowledgeMesh:
    return KnowledgeMesh(
        MeshConfig(
            roots=[root],
            state_dir=state,
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )


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


def test_query_top_k_zero_returns_no_hits(tmp_path: Path) -> None:
    # Regression for #814: ``top_k=0`` previously hit ``top_k or default`` and
    # was conflated with "unset", silently returning the configured default
    # number of hits. An explicit zero must return zero hits.
    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "auth.md").write_text(
            "# Auth\n\nAuthService uses FastAPI middleware for login.\n",
            encoding="utf-8",
        )
        mesh = _build_mesh(root, tmp_path / "state")
        await mesh.reindex()

        # Sanity check: without a limit we do get hits, so an empty result for
        # top_k=0 is meaningful and not just an empty index.
        default_result = await mesh.query("FastAPI auth middleware")
        assert default_result.hits

        zero_result = await mesh.query("FastAPI auth middleware", top_k=0)
        assert zero_result.hits == []

    asyncio.run(run())


def test_query_top_k_none_uses_configured_default(tmp_path: Path) -> None:
    # Complements the top_k=0 test: ``None`` must still fall back to the
    # configured ``retrieval_top_k`` (not zero).
    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        for i in range(6):
            (root / f"note-{i}.md").write_text(
                f"# Note {i}\n\nFastAPI middleware topic number {i} for login flow.\n",
                encoding="utf-8",
            )
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[root],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
                retrieval_top_k=3,
            )
        )
        await mesh.reindex()

        result = await mesh.query("FastAPI middleware login")
        assert 0 < len(result.hits) <= 3

    asyncio.run(run())


def test_chunks_path_active_index_exists(tmp_path: Path) -> None:
    # Regression for #813: the (path, active) composite index must be created so
    # ``WHERE path=? AND active=1`` and ``WHERE active=1`` avoid full scans.
    store = MeshIndexStore(tmp_path / "index.sqlite3")
    try:
        conn = sqlite3.connect(store.path)
        try:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
            assert "idx_chunks_path_active" in names

            # Confirm the index actually covers (path, active) in that order.
            columns = [
                row[2]
                for row in conn.execute(
                    "PRAGMA index_info('idx_chunks_path_active')"
                ).fetchall()
            ]
            assert columns == ["path", "active"]
        finally:
            conn.close()
    finally:
        store.close()


def test_query_filters_only_candidate_chunk_ids(tmp_path: Path) -> None:
    # Regression for #814: query() must not materialise the entire active
    # chunk-ID set. It should only check the candidate IDs surfaced by the
    # vector query. We verify behaviour: an orphan vector whose chunk_id is not
    # active (never reindexed into SQLite) is correctly filtered out, while
    # active chunks still surface.
    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "auth.md").write_text(
            "# Auth\n\nAuthService uses FastAPI middleware for login.\n",
            encoding="utf-8",
        )
        mesh = _build_mesh(root, tmp_path / "state")
        await mesh.reindex()

        # Inject an extra vector whose chunk_id is NOT active in SQLite.
        await mesh.rag.ingest(
            [
                {
                    "text": "OrphanVectorMarkerQQQ FastAPI middleware login.",
                    "metadata": {
                        "path": str(root / "orphan.md"),
                        "source": str(root / "orphan.md"),
                        "chunk_id": "orphan_not_in_index",
                    },
                }
            ]
        )

        result = await mesh.query("FastAPI middleware login", top_k=10)

        chunk_ids = {hit.metadata.get("chunk_id") for hit in result.hits}
        assert "orphan_not_in_index" not in chunk_ids
        assert not any("OrphanVectorMarkerQQQ" in hit.text for hit in result.hits)
        assert result.hits  # the real active chunk still surfaces

    asyncio.run(run())


def test_filter_active_chunk_ids_returns_active_subset(tmp_path: Path) -> None:
    # Unit-level regression for the candidate-filter helper introduced for #814.
    from synapsekit.loaders.base import Document

    store = MeshIndexStore(tmp_path / "index.sqlite3")
    try:
        store.mark_file_chunks(
            "/tmp/a.md",
            [
                Document(text="alpha", metadata={"chunk_id": "c1"}),
                Document(text="beta", metadata={"chunk_id": "c2"}),
            ],
        )
        # Empty candidate list short-circuits to empty set.
        assert store.filter_active_chunk_ids([]) == set()
        # Only active + requested IDs come back; unknown IDs are dropped.
        assert store.filter_active_chunk_ids(["c1", "unknown"]) == {"c1"}
        assert store.filter_active_chunk_ids(["c1", "c2"]) == {"c1", "c2"}

        # Deactivating a path removes its chunks from the active subset.
        store.mark_file_chunks(
            "/tmp/a.md",
            [Document(text="alpha2", metadata={"chunk_id": "c3"})],
        )
        assert store.filter_active_chunk_ids(["c1", "c2", "c3"]) == {"c3"}
    finally:
        store.close()


def test_store_usable_from_worker_thread(tmp_path: Path) -> None:
    # Regression for #812: reindex/query offload SQLite work to asyncio.to_thread,
    # so the store's connection must tolerate use from a non-creating thread
    # (check_same_thread=False + lock). Before the fix this raised
    # "SQLite objects created in a thread can only be used in that same thread".
    async def run() -> None:
        store = MeshIndexStore(tmp_path / "index.sqlite3")
        try:
            from synapsekit.loaders.base import Document

            await asyncio.to_thread(
                store.mark_file_chunks,
                "/tmp/a.md",
                [Document(text="alpha", metadata={"chunk_id": "c1"})],
            )
            ids = await asyncio.to_thread(store.active_chunk_ids)
            assert ids == {"c1"}
        finally:
            store.close()

    asyncio.run(run())


def test_reindex_and_query_run_on_larger_mesh(tmp_path: Path) -> None:
    # Behavioural check for #812/#814: a larger mesh reindexes and queries
    # correctly with the to_thread offloading and candidate filtering in place.
    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        for i in range(40):
            (root / f"doc-{i:02d}.md").write_text(
                f"# Doc {i}\n\nMeshTopic{i} discusses retrieval and FastAPI login flow {i}.\n",
                encoding="utf-8",
            )
        mesh = _build_mesh(root, tmp_path / "state")

        summary = await mesh.reindex()
        assert summary.discovered_files == 40
        assert summary.ingested_chunks >= 40

        result = await mesh.query("MeshTopic7 retrieval login", top_k=5)
        assert result.hits
        assert len(result.hits) <= 5
        # All returned hits are backed by active chunks.
        active = mesh.store.active_chunk_ids()
        assert all(hit.metadata.get("chunk_id") in active for hit in result.hits)

    asyncio.run(run())
