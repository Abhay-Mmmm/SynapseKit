from __future__ import annotations

import numpy as np
import pytest

from synapsekit.retrieval.property_graph import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphVectorStore,
    KnowledgeGraphExtraction,
    KnowledgeGraphExtractor,
    Neo4jPropertyGraphBackend,
    NetworkXPropertyGraphBackend,
    PropertyGraphEdge,
    PropertyGraphNode,
)


class FakeEmbeddings:
    """Hand-written deterministic embeddings backend for tests.

    Mirrors the SynapsekitEmbeddings async interface (embed/embed_one)
    without using MagicMock, per repo testing standards.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.dim, dtype=np.float32)
            vector[index % self.dim] = 1.0
            vecs.append(vector)
        return np.array(vecs, dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        arr = await self.embed([text])
        return arr[0]


def make_mock_embeddings(dim: int = 4) -> FakeEmbeddings:
    return FakeEmbeddings(dim=dim)


class StaticExtractor(KnowledgeGraphExtractor):
    async def extract(self, text: str) -> KnowledgeGraphExtraction:
        if "Apollo" in text:
            return KnowledgeGraphExtraction(
                entities=[
                    ExtractedEntity("Apollo", type="project", confidence=0.9),
                    ExtractedEntity("Dana", type="person", confidence=0.9),
                    ExtractedEntity("Platform Team", type="org", confidence=0.9),
                ],
                relationships=[
                    ExtractedRelationship("Dana", "Apollo", "leads", confidence=0.9),
                    ExtractedRelationship(
                        "Apollo",
                        "Platform Team",
                        "built_by",
                        confidence=0.9,
                    ),
                ],
            )
        return KnowledgeGraphExtraction(
            entities=[ExtractedEntity("Billing", type="system", confidence=0.9)]
        )


@pytest.mark.asyncio
async def test_extractor_ingest_stores_confidence_and_source_doc() -> None:
    graph = NetworkXPropertyGraphBackend()
    extractor = KnowledgeGraphExtractor(store=graph)

    await extractor.ingest([{"text": "Dana leads Apollo.", "metadata": {"source": "doc-a"}}])

    assert graph.nodes()
    assert graph.edges()
    edge = graph.edges()[0]
    assert edge.properties["source_doc"] == "doc-a"
    assert edge.properties["confidence"] >= 0.5


def test_backend_traversal_respects_hops_confidence_and_direction() -> None:
    graph = NetworkXPropertyGraphBackend()
    graph.upsert_node(PropertyGraphNode("a", "A"))
    graph.upsert_node(PropertyGraphNode("b", "B"))
    graph.upsert_node(PropertyGraphNode("c", "C"))
    graph.upsert_edge(PropertyGraphEdge("a", "b", "knows", {"confidence": 0.9}))
    graph.upsert_edge(PropertyGraphEdge("b", "c", "knows", {"confidence": 0.2}))

    nodes, edges = graph.traverse(["a"], max_hops=2, min_confidence=0.5)

    assert [node.id for node in nodes] == ["a", "b"]
    assert [edge.id for edge in edges] == ["a:knows:b"]
    assert graph.neighbors("b", direction="in") == ["a"]
    with pytest.raises(ValueError, match="direction"):
        graph.neighbors("a", direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_hops"):
        graph.traverse(["a"], max_hops=-1)


@pytest.mark.asyncio
async def test_graph_vector_store_is_drop_in_vector_store_and_expands_graph_hits() -> None:
    store = GraphVectorStore(
        make_mock_embeddings(),
        extractor=StaticExtractor(),
        max_hops=2,
        graph_weight=1.0,
    )

    await store.add(
        [
            "Apollo is led by Dana and built by Platform Team.",
            "Billing invoices customers.",
        ],
        metadata=[{"source": "apollo", "tenant": "a"}, {"source": "billing", "tenant": "b"}],
    )

    results = await store.search("Who leads Apollo?", top_k=2)

    assert len(store) == 2
    assert len(results) == 2
    assert results[0]["metadata"]["retrieval_source"] == "graph_vector"
    assert "Apollo" in results[0]["metadata"]["graph_nodes"]
    assert all("text" in item and "score" in item and "metadata" in item for item in results)


@pytest.mark.asyncio
async def test_graph_vector_store_metadata_filter_and_empty_paths() -> None:
    store = GraphVectorStore(make_mock_embeddings(), extractor=StaticExtractor())

    assert await store.search("empty") == []
    await store.add(["Apollo is led by Dana."], metadata=[{"source": "apollo", "tenant": "a"}])

    assert await store.search("Apollo", top_k=0) == []
    assert await store.search("Apollo", metadata_filter={"tenant": "missing"}) == []
    with pytest.raises(ValueError, match="metadata length"):
        await store.add(["one"], metadata=[])


def test_neo4j_backend_roundtrip_with_testcontainers() -> None:
    pytest.importorskip("neo4j")
    container_mod = pytest.importorskip("testcontainers.core.container")
    wait_mod = pytest.importorskip("testcontainers.core.waiting_utils")
    docker_container = container_mod.DockerContainer
    wait_for_logs = wait_mod.wait_for_logs

    password = "synapsekit-password"
    with (
        docker_container("neo4j:5")
        .with_env("NEO4J_AUTH", f"neo4j/{password}")
        .with_exposed_ports(7687) as container
    ):
        wait_for_logs(container, "Bolt enabled", timeout=60)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(7687)
        backend = Neo4jPropertyGraphBackend(
            f"bolt://{host}:{port}",
            username="neo4j",
            password=password,
        )
        try:
            backend.upsert_node(
                PropertyGraphNode("apollo", "Apollo", properties={"source_doc": "d1"})
            )
            backend.upsert_node(PropertyGraphNode("dana", "Dana", properties={"source_doc": "d1"}))
            backend.upsert_edge(
                PropertyGraphEdge(
                    "dana",
                    "apollo",
                    "leads",
                    {"confidence": 0.9, "source_doc": "d1"},
                )
            )

            nodes, edges = backend.traverse(["dana"], max_hops=1, min_confidence=0.5)

            assert {node.id for node in nodes} == {"apollo", "dana"}
            assert [edge.relation for edge in edges] == ["leads"]
            assert backend.document_ids_for_nodes(["apollo", "dana"]) == ["d1"]

            # Regression for bug 1: non-default max_hops used to be passed as
            # a Cypher query PARAMETER inside a variable-length relationship
            # pattern (*0..$hops), which Neo4j's grammar rejects with a
            # CypherSyntaxError. It must now be interpolated as a literal.
            backend.upsert_node(
                PropertyGraphNode("platform", "Platform Team", properties={"source_doc": "d1"})
            )
            backend.upsert_edge(
                PropertyGraphEdge(
                    "apollo",
                    "platform",
                    "built_by",
                    {"confidence": 0.9, "source_doc": "d1"},
                )
            )
            multi_hop_nodes, multi_hop_edges = backend.traverse(
                ["dana"], max_hops=3, min_confidence=0.0
            )
            assert {node.id for node in multi_hop_nodes} == {"apollo", "dana", "platform"}
            assert {edge.relation for edge in multi_hop_edges} == {"leads", "built_by"}

            zero_hop_nodes, zero_hop_edges = backend.traverse(["dana"], max_hops=0)
            assert {node.id for node in zero_hop_nodes} == {"dana"}
            assert zero_hop_edges == []

            with pytest.raises(ValueError, match="max_hops"):
                backend.traverse(["dana"], max_hops=-1)
        finally:
            backend.close()


def test_networkx_backend_remove_node_deletes_incident_edges_and_indexes() -> None:
    """Regression for bug 2: deleting a node must not leave dangling edges
    or stale label/document index entries behind."""
    graph = NetworkXPropertyGraphBackend()
    graph.upsert_node(PropertyGraphNode("a", "A", properties={"source_doc": "doc-1"}))
    graph.upsert_node(PropertyGraphNode("b", "B", properties={"source_doc": "doc-1"}))
    graph.upsert_node(PropertyGraphNode("c", "C", properties={"source_doc": "doc-2"}))
    graph.upsert_edge(PropertyGraphEdge("a", "b", "knows", {"source_doc": "doc-1"}))
    graph.upsert_edge(PropertyGraphEdge("b", "c", "knows", {"source_doc": "doc-1"}))

    removed = graph.remove_node("b")

    assert removed is True
    assert graph.get_node("b") is None
    assert "b" not in [node.id for node in graph.nodes()]
    remaining_edge_endpoints = {(edge.source, edge.target) for edge in graph.edges()}
    assert ("a", "b") not in remaining_edge_endpoints
    assert ("b", "c") not in remaining_edge_endpoints
    assert graph.node_id_for_label("B") is None
    assert "b" not in graph.node_ids_for_document("doc-1")
    assert graph.neighbors("a") == []
    # Removing again / removing an unknown node is a no-op, not an error.
    assert graph.remove_node("b") is False
    assert graph.remove_node("does-not-exist") is False


def test_seed_ids_uses_document_index_not_full_node_scan() -> None:
    """Regression for bug 4: _seed_ids must resolve a vector hit's source
    document to node ids via the O(1) node_ids_for_document index rather
    than scanning every node in the graph."""
    graph = NetworkXPropertyGraphBackend()
    graph.upsert_node(PropertyGraphNode("n1", "Node One", properties={"source_doc": "doc-a"}))
    graph.upsert_node(PropertyGraphNode("n2", "Node Two", properties={"source_doc": "doc-b"}))

    store = GraphVectorStore(make_mock_embeddings(), backend=graph)

    scan_calls = {"count": 0}
    original_nodes = graph.nodes

    def counting_nodes() -> list[PropertyGraphNode]:
        scan_calls["count"] += 1
        return original_nodes()

    graph.nodes = counting_nodes  # type: ignore[method-assign]

    seeds = store._seed_ids("query text", [{"metadata": {"source": "doc-a"}}])

    assert seeds == ["n1"]
    # The full-scan fallback (`for node in self.graph.nodes(): ...`) must not
    # be exercised when the O(1) node_ids_for_document index is available.
    assert scan_calls["count"] == 0


@pytest.mark.asyncio
async def test_graph_vector_store_search_does_not_leak_other_tenant_graph_metadata() -> None:
    """Regression for bug 3: a metadata_filter scoped search must not fuse
    another tenant's graph nodes/edges into the result metadata, even though
    traversal runs over a graph shared by all tenants."""
    store = GraphVectorStore(
        make_mock_embeddings(),
        extractor=StaticExtractor(),
        max_hops=2,
        graph_weight=1.0,
    )

    await store.add(
        [
            "Apollo is led by Dana and built by Platform Team.",
            "Billing invoices customers.",
        ],
        metadata=[
            {"source": "apollo", "tenant": "tenant-a"},
            {"source": "billing", "tenant": "tenant-b"},
        ],
    )

    results = await store.search(
        "Who leads Apollo?", top_k=5, metadata_filter={"tenant": "tenant-b"}
    )

    for item in results:
        graph_nodes = item.get("metadata", {}).get("graph_nodes", [])
        graph_edges = item.get("metadata", {}).get("graph_edges", [])
        assert "Apollo" not in graph_nodes
        assert "Dana" not in graph_nodes
        assert "Platform Team" not in graph_nodes
        assert "leads" not in graph_edges
        assert "built_by" not in graph_edges


@pytest.mark.asyncio
async def test_graph_vector_store_search_keeps_own_tenant_graph_metadata() -> None:
    """Sanity check paired with the tenant-leak regression: filtering must
    not also strip a tenant's own graph context."""
    store = GraphVectorStore(
        make_mock_embeddings(),
        extractor=StaticExtractor(),
        max_hops=2,
        graph_weight=1.0,
    )

    await store.add(
        [
            "Apollo is led by Dana and built by Platform Team.",
            "Billing invoices customers.",
        ],
        metadata=[
            {"source": "apollo", "tenant": "tenant-a"},
            {"source": "billing", "tenant": "tenant-b"},
        ],
    )

    results = await store.search(
        "Who leads Apollo?", top_k=5, metadata_filter={"tenant": "tenant-a"}
    )

    graph_nodes = {
        node for item in results for node in item.get("metadata", {}).get("graph_nodes", [])
    }
    assert "Apollo" in graph_nodes
