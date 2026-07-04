from __future__ import annotations

from unittest.mock import MagicMock

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


def make_mock_embeddings(dim: int = 4) -> MagicMock:
    mock = MagicMock()

    async def embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for index, _text in enumerate(texts):
            vector = np.zeros(dim, dtype=np.float32)
            vector[index % dim] = 1.0
            vecs.append(vector)
        return np.array(vecs, dtype=np.float32)

    async def embed_one(text: str) -> np.ndarray:
        arr = await embed([text])
        return arr[0]

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


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
        finally:
            backend.close()
