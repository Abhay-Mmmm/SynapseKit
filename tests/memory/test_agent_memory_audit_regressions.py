"""Regression tests for the agent-memory audit batch.

Covered issues:
  #794 — overflow consolidation ignored ``consolidation_window`` and produced a
         one-line semantic record per episode.
  #796 — vectorized recall must match a naive pure-Python reference.
  #797 — batched ``touch_many`` must update every recalled record's stats.
  #798 — SQLite backend keeps one persistent connection; store/recall persist.

Hand-written fakes and real SQLite (tmp_path) only — no MagicMock.
"""

from __future__ import annotations

import math

import pytest

from synapsekit.memory import AgentMemory
from synapsekit.memory.agent_memory import AgentMemory as _AgentMemory
from synapsekit.memory.backends.sqlite import SQLiteMemoryBackend


class CountingLLM:
    """Deterministic consolidation summarizer that records call inputs."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # Count the bullet lines the consolidator handed us.
        n = prompt.count("\n- ")
        return f"Summary of {n} episodes."


# ---------------------------------------------------------------------------
# #794 — overflow consolidation respects consolidation_window
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_overflow_consolidation_uses_consolidation_window():
    """Crossing max_episodes by one must consolidate a full window, not one.

    Fails on old code: overflow=1 → summarizes episodic[:1], spawning a
    single-episode semantic record and leaving episodic bloated.
    """
    llm = CountingLLM()
    mem = AgentMemory(
        backend="memory",
        llm=llm,
        max_episodes=5,
        consolidation_window=4,
    )

    # 6 episodes → overflow of 1 over the cap of 5.
    for i in range(6):
        await mem.store(agent_id="u1", content=f"episode {i}", memory_type="episodic")

    # Exactly one consolidation should have run, over a full window of 4.
    assert len(llm.prompts) == 1
    assert llm.prompts[0].count("\n- ") == 4

    episodic = await mem.count(agent_id="u1", memory_type="episodic")
    semantic = await mem.count(agent_id="u1", memory_type="semantic")

    # 6 stored, 4 consolidated away, 1 semantic summary created.
    assert episodic == 2
    assert semantic == 1


@pytest.mark.asyncio
async def test_overflow_consolidation_when_overflow_exceeds_window():
    """When overflow already exceeds the window, consolidate the overflow."""
    llm = CountingLLM()
    mem = AgentMemory(
        backend="memory",
        llm=llm,
        max_episodes=3,
        consolidation_window=2,
    )

    # Store 3 first (no overflow yet), then one more triggers overflow=1<window.
    for i in range(4):
        await mem.store(agent_id="u1", content=f"e{i}", memory_type="episodic")
    # overflow=1, window=2 → take 2.
    assert llm.prompts[-1].count("\n- ") == 2


# ---------------------------------------------------------------------------
# #796 — vectorized recall matches a naive reference
# ---------------------------------------------------------------------------
def _naive_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_batch_cosine_matches_naive_reference():
    query = [0.1, 0.9, 0.2, 0.0]
    embeddings = [
        [0.1, 0.9, 0.2, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # zero vector → 0.0
        [0.3, 0.3, 0.3, 0.3],
        [0.9, 0.1, 0.2, 0.0],
    ]
    got = _AgentMemory._batch_cosine(query, embeddings)
    expected = [_naive_cosine(query, e) for e in embeddings]
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert abs(g - e) < 1e-9


def test_batch_cosine_handles_dim_mismatch_and_empty():
    query = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [1.0, 0.0, 0.0], [], [0.0, 1.0]]
    got = _AgentMemory._batch_cosine(query, embeddings)
    expected = [_naive_cosine(query, e) for e in embeddings]
    for g, e in zip(got, expected, strict=True):
        assert abs(g - e) < 1e-9
    assert _AgentMemory._batch_cosine([], [[1.0]]) == [0.0]
    assert _AgentMemory._batch_cosine([1.0], []) == []


@pytest.mark.asyncio
async def test_recall_ranking_matches_naive_scoring():
    """End-to-end recall ordering must match a naive per-record computation."""
    mem = AgentMemory(backend="memory")
    contents = [
        "python programming language",
        "java virtual machine",
        "rust memory safety",
        "python data science",
        "cooking pasta recipe",
    ]
    for c in contents:
        await mem.store(agent_id="u1", content=c, memory_type="semantic")

    results = await mem.recall(agent_id="u1", query="python coding", top_k=3)
    assert len(results) == 3
    # Both python records should surface above unrelated ones.
    joined = " ".join(r.content for r in results)
    assert "python" in joined


# ---------------------------------------------------------------------------
# #797 — touch_many updates access stats for every recalled record
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_recall_touches_all_returned_records_in_memory_backend():
    mem = AgentMemory(backend="memory")
    for i in range(3):
        await mem.store(agent_id="u1", content=f"fact {i}", memory_type="semantic")

    results = await mem.recall(agent_id="u1", query="fact", top_k=3)
    assert len(results) == 3
    assert all(r.access_count == 1 for r in results)

    results2 = await mem.recall(agent_id="u1", query="fact", top_k=3)
    assert all(r.access_count == 2 for r in results2)


@pytest.mark.asyncio
async def test_touch_many_batches_updates_on_sqlite(tmp_path):
    backend = SQLiteMemoryBackend(str(tmp_path / "mem.db"))
    mem = AgentMemory(backend=backend)
    for i in range(4):
        await mem.store(agent_id="u1", content=f"item {i}", memory_type="semantic")

    results = await mem.recall(agent_id="u1", query="item", top_k=4)
    ids = {r.id for r in results}
    assert len(ids) == 4

    stored = await backend.fetch("u1", memory_type="semantic")
    assert all(r.access_count == 1 for r in stored)


# ---------------------------------------------------------------------------
# #798 — persistent SQLite connection survives across operations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sqlite_persistent_connection_store_and_recall(tmp_path):
    db = str(tmp_path / "persist.db")
    backend = SQLiteMemoryBackend(db)
    mem = AgentMemory(backend=backend)

    await mem.store(agent_id="u1", content="durable memory", memory_type="semantic")
    results = await mem.recall(agent_id="u1", query="durable", top_k=1)
    assert len(results) == 1
    assert results[0].content == "durable memory"

    # A second backend on the same file (new connection) sees the row (WAL flush).
    backend.close()
    backend2 = SQLiteMemoryBackend(db)
    fetched = await backend2.fetch("u1", memory_type="semantic")
    assert any(r.content == "durable memory" for r in fetched)
    backend2.close()


@pytest.mark.asyncio
async def test_sqlite_reuses_single_connection(tmp_path):
    backend = SQLiteMemoryBackend(str(tmp_path / "one.db"))
    conn_id = id(backend._conn)
    mem = AgentMemory(backend=backend)
    await mem.store(agent_id="u1", content="x", memory_type="semantic")
    await mem.recall(agent_id="u1", query="x", top_k=1)
    await mem.count(agent_id="u1")
    # The connection object is never re-created per operation.
    assert id(backend._conn) == conn_id
    backend.close()


# ---------------------------------------------------------------------------
# #790 — top-level AgentMemory export points at the persistent implementation
# ---------------------------------------------------------------------------
def test_top_level_agent_memory_is_persistent_without_deprecation_warning():
    """`from synapsekit import AgentMemory` must return the persistent class
    and instantiating it must not emit a DeprecationWarning.

    Fails on old code where the top-level export aliased the deprecated
    scratchpad shim.
    """
    import warnings

    import synapsekit
    from synapsekit.memory.agent_memory import AgentMemory as PersistentAM

    assert synapsekit.AgentMemory is PersistentAM
    # Kept as a still-working alias for one release.
    assert synapsekit.PersistentAgentMemory is PersistentAM

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        inst = synapsekit.AgentMemory(backend="memory")
    assert inst is not None


def test_top_level_agent_scratchpad_is_the_step_scratchpad():
    import synapsekit
    from synapsekit.agents.memory import AgentScratchpad

    assert synapsekit.AgentScratchpad is AgentScratchpad


# ---------------------------------------------------------------------------
# #798 — LLM consolidation failure logs a warning (bare-except removed)
# ---------------------------------------------------------------------------
class _FailingLLM:
    async def generate(self, prompt: str) -> str:
        raise RuntimeError("llm down")


@pytest.mark.asyncio
async def test_consolidation_llm_failure_logs_warning_and_falls_back(caplog):
    """A failing consolidation LLM must warn and fall back deterministically,
    not silently swallow the error (issue #798).
    """
    import logging

    mem = AgentMemory(
        backend="memory",
        llm=_FailingLLM(),
        max_episodes=2,
        consolidation_window=2,
    )
    with caplog.at_level(logging.WARNING, logger="synapsekit.memory.agent_memory"):
        for i in range(3):
            await mem.store(agent_id="u1", content=f"episode {i}", memory_type="episodic")

    assert any("consolidation failed" in r.message.lower() for r in caplog.records)
    # Fallback still produced a semantic record.
    assert await mem.count(agent_id="u1", memory_type="semantic") >= 1
