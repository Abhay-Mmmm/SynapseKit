from __future__ import annotations

import pytest

from synapsekit.memory import AgentMemory, GraphMemoryBackend
from synapsekit.retrieval.property_graph import NetworkXPropertyGraphBackend


@pytest.mark.asyncio
async def test_agent_memory_graph_backend_matches_core_interface() -> None:
    graph = NetworkXPropertyGraphBackend()
    memory = AgentMemory(backend="graph", store=graph, max_episodes=50)

    first = await memory.store(
        agent_id="agent-a",
        content="User prefers graph-aware RAG",
        memory_type="semantic",
    )
    second = await memory.store(
        agent_id="agent-a",
        content="Apollo depends on Platform Team",
        memory_type="episodic",
        metadata={
            "related_to": [first.id],
            "relation_type": "supports",
            "weight": 0.8,
        },
    )

    recalled = await memory.recall(agent_id="agent-a", query="graph RAG", top_k=1)

    assert recalled[0].id == first.id
    assert await memory.count(agent_id="agent-a") == 2
    assert graph.get_node(f"memory:agent-a:{first.id}") is not None
    edge = graph.get_edge(f"memory:agent-a:{second.id}:supports:memory:agent-a:{first.id}")
    assert edge is not None
    assert edge.properties["relation_type"] == "supports"
    assert edge.properties["weight"] == 0.8


@pytest.mark.asyncio
async def test_graph_memory_backend_delete_clear_and_prune() -> None:
    backend = GraphMemoryBackend()
    memory = AgentMemory(backend=backend)

    expired = await memory.store(
        agent_id="agent-a",
        content="temporary",
        memory_type="episodic",
        ttl_days=0,
    )
    kept = await memory.store(
        agent_id="agent-a",
        content="stable",
        memory_type="semantic",
    )

    assert await backend.prune_expired() == 1
    assert await memory.count(agent_id="agent-a") == 1
    assert await memory.delete(agent_id="agent-a", record_id=expired.id) is False
    assert await memory.count(agent_id="agent-a") == 1
    assert await memory.clear(agent_id="agent-a", memory_type="semantic") == 1
    assert await memory.delete(agent_id="agent-a", record_id=kept.id) is False


@pytest.mark.asyncio
async def test_graph_memory_backend_delete_removes_graph_node_and_edges() -> None:
    """Regression for bug 2: delete() must remove the node (and incident
    edges) from graph_store, not just pop the record from self._records.
    AgentMemory.consolidate() relies on this to avoid leaking graph nodes
    and dangling edges unboundedly during normal operation."""
    graph = NetworkXPropertyGraphBackend()
    backend = GraphMemoryBackend(store=graph)
    memory = AgentMemory(backend=backend)

    first = await memory.store(
        agent_id="agent-a",
        content="User prefers graph-aware RAG",
        memory_type="semantic",
    )
    second = await memory.store(
        agent_id="agent-a",
        content="Apollo depends on Platform Team",
        memory_type="episodic",
        metadata={
            "related_to": [first.id],
            "relation_type": "supports",
            "weight": 0.8,
        },
    )

    first_node_id = f"memory:agent-a:{first.id}"
    second_node_id = f"memory:agent-a:{second.id}"
    assert graph.get_node(first_node_id) is not None
    assert graph.get_node(second_node_id) is not None
    assert graph.neighbors(first_node_id) != []

    deleted = await memory.delete(agent_id="agent-a", record_id=first.id)

    assert deleted is True
    assert graph.get_node(first_node_id) is None
    # The edge from second -> first must be gone too (no dangling edge
    # referencing a deleted memory id).
    remaining_edge_endpoints = {(edge.source, edge.target) for edge in graph.edges()}
    assert (second_node_id, first_node_id) not in remaining_edge_endpoints
    assert graph.neighbors(second_node_id) == []
    # The second record's own node must be untouched.
    assert graph.get_node(second_node_id) is not None
