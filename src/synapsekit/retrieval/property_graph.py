"""Property-graph retrieval primitives for graph-aware RAG."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, Literal, Protocol

from ..embeddings.backend import SynapsekitEmbeddings
from ..llm.base import BaseLLM
from .base import VectorStore
from .vectorstore import InMemoryVectorStore

PropertyGraphBackendName = Literal["networkx", "neo4j"]

_ENTITY_RE = re.compile(
    r"(?:@[A-Za-z][\w.-]*|[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4})"
)
_RELATION_HINTS: tuple[tuple[str, str], ...] = (
    ("acquired", "acquired"),
    ("built", "built"),
    ("created", "created"),
    ("depends on", "depends_on"),
    ("founded", "founded"),
    ("leads", "leads"),
    ("led", "led"),
    ("released", "released"),
    ("uses", "uses"),
    ("works on", "works_on"),
)


@dataclass(slots=True)
class PropertyGraphNode:
    """A property-graph node with arbitrary metadata."""

    id: str
    label: str
    type: str = "entity"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PropertyGraphEdge:
    """A directed property-graph edge with arbitrary metadata."""

    source: str
    target: str
    relation: str = "related_to"
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}:{self.relation}:{self.target}"


@dataclass(slots=True)
class ExtractedEntity:
    """Entity extracted from a document."""

    name: str
    type: str = "entity"
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedRelationship:
    """Relationship extracted from a document."""

    source: str
    target: str
    relation: str = "related_to"
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeGraphExtraction:
    """Structured extraction output for one document."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


class PropertyGraphBackend(Protocol):
    """Storage contract used by graph RAG and graph-backed memory."""

    def upsert_node(self, node: PropertyGraphNode) -> None: ...

    def upsert_edge(self, edge: PropertyGraphEdge) -> None: ...

    def get_node(self, node_id: str) -> PropertyGraphNode | None: ...

    def get_edge(self, edge_id: str) -> PropertyGraphEdge | None: ...

    def neighbors(
        self, node_id: str, *, direction: Literal["both", "out", "in"] = "both"
    ) -> list[str]: ...

    def traverse(
        self,
        seed_ids: list[str],
        *,
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> tuple[list[PropertyGraphNode], list[PropertyGraphEdge]]: ...

    def document_ids_for_nodes(self, node_ids: list[str]) -> list[str]: ...

    def nodes(self) -> list[PropertyGraphNode]: ...

    def edges(self) -> list[PropertyGraphEdge]: ...


class _AdjacencyGraph:
    """Small fallback graph used when NetworkX is not installed."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, source: str, target: str, key: str, **attrs: Any) -> None:
        self.edges[(source, target, key)] = attrs
        self.outgoing[source].add(target)
        self.incoming[target].add(source)


class NetworkXPropertyGraphBackend:
    """In-memory property graph backend with a NetworkX-compatible fallback."""

    def __init__(self) -> None:
        self._nodes: dict[str, PropertyGraphNode] = {}
        self._edges: dict[str, PropertyGraphEdge] = {}
        self._label_to_id: dict[str, str] = {}
        self._documents_by_node: dict[str, set[str]] = defaultdict(set)
        try:
            import networkx as nx  # type: ignore[import-not-found]
        except ImportError:
            self._graph: Any = _AdjacencyGraph()
            self.uses_networkx = False
        else:
            self._graph = nx.MultiDiGraph()
            self.uses_networkx = True

    def upsert_node(self, node: PropertyGraphNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        else:
            existing.properties.update(node.properties)
            existing.type = node.type or existing.type
            existing.label = node.label or existing.label
        self._label_to_id[_normalize(node.label)] = node.id
        for alias in node.properties.get("aliases", []) or []:
            self._label_to_id[_normalize(str(alias))] = node.id
        source_doc = node.properties.get("source_doc") or node.properties.get("source")
        if source_doc:
            self._documents_by_node[node.id].add(str(source_doc))
        self._graph.add_node(
            node.id,
            label=node.label,
            type=node.type,
            **node.properties,
        )

    def upsert_edge(self, edge: PropertyGraphEdge) -> None:
        self._edges[edge.id] = edge
        source_doc = edge.properties.get("source_doc") or edge.properties.get("source")
        if source_doc:
            self._documents_by_node[edge.source].add(str(source_doc))
            self._documents_by_node[edge.target].add(str(source_doc))
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            relation=edge.relation,
            **edge.properties,
        )

    def get_node(self, node_id: str) -> PropertyGraphNode | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> PropertyGraphEdge | None:
        return self._edges.get(edge_id)

    def neighbors(
        self, node_id: str, *, direction: Literal["both", "out", "in"] = "both"
    ) -> list[str]:
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', or 'in'")
        if self.uses_networkx:
            out = set(self._graph.successors(node_id)) if direction in {"both", "out"} else set()
            inc = set(self._graph.predecessors(node_id)) if direction in {"both", "in"} else set()
            return sorted(out | inc)
        out = self._graph.outgoing.get(node_id, set()) if direction in {"both", "out"} else set()
        inc = self._graph.incoming.get(node_id, set()) if direction in {"both", "in"} else set()
        return sorted(out | inc)

    def traverse(
        self,
        seed_ids: list[str],
        *,
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> tuple[list[PropertyGraphNode], list[PropertyGraphEdge]]:
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seed_ids if seed in self._nodes)
        visited: set[str] = set()
        selected_edges: dict[str, PropertyGraphEdge] = {}

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            if depth >= max_hops:
                continue
            for edge in self._incident_edges(node_id):
                confidence = float(edge.properties.get("confidence", 1.0))
                if confidence < min_confidence:
                    continue
                selected_edges[edge.id] = edge
                other = edge.target if edge.source == node_id else edge.source
                if other not in visited:
                    queue.append((other, depth + 1))

        nodes = [self._nodes[node_id] for node_id in sorted(visited)]
        edges = [selected_edges[edge_id] for edge_id in sorted(selected_edges)]
        return nodes, edges

    def document_ids_for_nodes(self, node_ids: list[str]) -> list[str]:
        docs: set[str] = set()
        for node_id in node_ids:
            docs.update(self._documents_by_node.get(node_id, set()))
        return sorted(docs)

    def nodes(self) -> list[PropertyGraphNode]:
        return list(self._nodes.values())

    def edges(self) -> list[PropertyGraphEdge]:
        return list(self._edges.values())

    def node_id_for_label(self, label: str) -> str | None:
        return self._label_to_id.get(_normalize(label))

    def seed_ids_for_text(self, text: str) -> list[str]:
        normalized = _normalize(text)
        seeds = [
            node_id for label, node_id in self._label_to_id.items() if label and label in normalized
        ]
        if seeds:
            return list(dict.fromkeys(seeds))
        terms = set(normalized.split())
        scored: list[tuple[int, str]] = []
        for node in self._nodes.values():
            score = len(terms & set(_normalize(node.label).split()))
            if score:
                scored.append((score, node.id))
        return [node_id for _, node_id in sorted(scored, reverse=True)[:5]]

    def _incident_edges(self, node_id: str) -> list[PropertyGraphEdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.source == node_id or edge.target == node_id
        ]


class Neo4jPropertyGraphBackend:
    """Neo4j property graph adapter with lazy optional dependency loading."""

    def __init__(
        self,
        uri: str,
        *,
        username: str = "neo4j",
        password: str = "password",
        database: str | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Neo4jPropertyGraphBackend requires the 'neo4j' package. "
                "Install SynapseKit with the graph extra."
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def upsert_node(self, node: PropertyGraphNode) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (n:PropertyGraphNode {id: $id})
                SET n.label = $label, n.type = $type, n += $properties
                """,
                id=node.id,
                label=node.label,
                type=node.type,
                properties=node.properties,
            )

    def upsert_edge(self, edge: PropertyGraphEdge) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (a:PropertyGraphNode {id: $source})
                MERGE (b:PropertyGraphNode {id: $target})
                MERGE (a)-[r:RELATED {id: $id}]->(b)
                SET r.relation = $relation, r += $properties
                """,
                source=edge.source,
                target=edge.target,
                id=edge.id,
                relation=edge.relation,
                properties=edge.properties,
            )

    def get_node(self, node_id: str) -> PropertyGraphNode | None:
        with self._driver.session(database=self._database) as session:
            row = session.run(
                "MATCH (n:PropertyGraphNode {id: $id}) RETURN n",
                id=node_id,
            ).single()
        if row is None:
            return None
        data = dict(row["n"])
        return PropertyGraphNode(
            id=str(data.pop("id")),
            label=str(data.pop("label", node_id)),
            type=str(data.pop("type", "entity")),
            properties=data,
        )

    def get_edge(self, edge_id: str) -> PropertyGraphEdge | None:
        with self._driver.session(database=self._database) as session:
            row = session.run(
                """
                MATCH (a)-[r:RELATED {id: $id}]->(b)
                RETURN a.id AS source, b.id AS target, r AS rel
                """,
                id=edge_id,
            ).single()
        if row is None:
            return None
        data = dict(row["rel"])
        return PropertyGraphEdge(
            source=str(row["source"]),
            target=str(row["target"]),
            relation=str(data.pop("relation", "related_to")),
            properties=data,
        )

    def neighbors(
        self, node_id: str, *, direction: Literal["both", "out", "in"] = "both"
    ) -> list[str]:
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', or 'in'")
        pattern = {
            "both": "-[:RELATED]-",
            "out": "-[:RELATED]->",
            "in": "<-[:RELATED]-",
        }[direction]
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                f"MATCH (:PropertyGraphNode {{id: $id}}){pattern}(n) RETURN DISTINCT n.id AS id",
                id=node_id,
            )
            return sorted(str(row["id"]) for row in rows)

    def traverse(
        self,
        seed_ids: list[str],
        *,
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> tuple[list[PropertyGraphNode], list[PropertyGraphEdge]]:
        memory = NetworkXPropertyGraphBackend()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH path=(n:PropertyGraphNode)-[r:RELATED*0..$hops]-(m:PropertyGraphNode)
                WHERE n.id IN $ids
                UNWIND nodes(path) AS node
                RETURN DISTINCT node
                """,
                ids=seed_ids,
                hops=max_hops,
            )
            for row in rows:
                data = dict(row["node"])
                node_id = str(data.pop("id"))
                memory.upsert_node(
                    PropertyGraphNode(
                        id=node_id,
                        label=str(data.pop("label", node_id)),
                        type=str(data.pop("type", "entity")),
                        properties=data,
                    )
                )
            edge_rows = session.run(
                """
                MATCH path=(n:PropertyGraphNode)-[r:RELATED*1..$hops]-(m:PropertyGraphNode)
                WHERE n.id IN $ids
                UNWIND relationships(path) AS rel
                WITH DISTINCT rel
                WHERE coalesce(rel.confidence, 1.0) >= $min_confidence
                RETURN startNode(rel).id AS source, endNode(rel).id AS target, rel
                """,
                ids=seed_ids,
                hops=max_hops,
                min_confidence=min_confidence,
            )
            for row in edge_rows:
                data = dict(row["rel"])
                memory.upsert_edge(
                    PropertyGraphEdge(
                        source=str(row["source"]),
                        target=str(row["target"]),
                        relation=str(data.pop("relation", "related_to")),
                        properties=data,
                    )
                )
        return memory.nodes(), memory.edges()

    def document_ids_for_nodes(self, node_ids: list[str]) -> list[str]:
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (n:PropertyGraphNode)
                WHERE n.id IN $ids AND n.source_doc IS NOT NULL
                RETURN DISTINCT n.source_doc AS doc
                """,
                ids=node_ids,
            )
            return sorted(str(row["doc"]) for row in rows)

    def nodes(self) -> list[PropertyGraphNode]:
        with self._driver.session(database=self._database) as session:
            rows = session.run("MATCH (n:PropertyGraphNode) RETURN n")
            out: list[PropertyGraphNode] = []
            for row in rows:
                data = dict(row["n"])
                node_id = str(data.pop("id"))
                out.append(
                    PropertyGraphNode(
                        id=node_id,
                        label=str(data.pop("label", node_id)),
                        type=str(data.pop("type", "entity")),
                        properties=data,
                    )
                )
            return out

    def edges(self) -> list[PropertyGraphEdge]:
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "MATCH (a)-[r:RELATED]->(b) RETURN a.id AS source, b.id AS target, r"
            )
            out: list[PropertyGraphEdge] = []
            for row in rows:
                data = dict(row["r"])
                out.append(
                    PropertyGraphEdge(
                        source=str(row["source"]),
                        target=str(row["target"]),
                        relation=str(data.pop("relation", "related_to")),
                        properties=data,
                    )
                )
            return out

    def close(self) -> None:
        self._driver.close()


class KnowledgeGraphExtractor:
    """Extract entities and relationships into a property graph."""

    _PROMPT = """\
Extract a property graph from the text.
Return strict JSON with keys "entities" and "relationships".
Entity: {"name": str, "type": str, "confidence": float, "properties": object}
Relationship: {"source": str, "target": str, "relation": str, "confidence": float, "properties": object}

Text:
{text}
"""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        store: PropertyGraphBackend | None = None,
        *,
        min_confidence: float = 0.5,
    ) -> None:
        self.llm = llm
        self.store = store
        self.min_confidence = min_confidence

    async def extract(self, text: str) -> KnowledgeGraphExtraction:
        if self.llm is None:
            return self._heuristic_extract(text)
        response = await self.llm.generate(self._PROMPT.format(text=text))
        return self._parse_json(response)

    async def ingest(
        self,
        documents: list[str] | list[dict[str, Any]],
        *,
        store: PropertyGraphBackend | None = None,
    ) -> list[KnowledgeGraphExtraction]:
        target = store or self.store
        if target is None:
            raise ValueError("A property graph store is required for ingest().")
        outputs: list[KnowledgeGraphExtraction] = []
        for index, document in enumerate(documents):
            text, metadata = _coerce_document(document)
            source_doc = str(metadata.get("source") or metadata.get("id") or f"doc_{index + 1}")
            extraction = await self.extract(text)
            _store_extraction(target, extraction, source_doc=source_doc)
            outputs.append(extraction)
        return outputs

    def _heuristic_extract(self, text: str) -> KnowledgeGraphExtraction:
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()
        for match in _ENTITY_RE.finditer(text):
            name = match.group(0).strip(" .,;:()[]")
            if name in {"A", "An", "And", "In", "The", "This", "Who", "What"}:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                ExtractedEntity(
                    name=name,
                    type=_guess_entity_type(name),
                    confidence=0.72,
                )
            )
        relationships: list[ExtractedRelationship] = []
        names = [entity.name for entity in entities]
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            present = [name for name in names if name in sentence]
            if len(present) < 2:
                continue
            relation = _relation_for_sentence(sentence)
            for left, right in pairwise(present):
                relationships.append(
                    ExtractedRelationship(
                        source=left,
                        target=right,
                        relation=relation,
                        confidence=0.7,
                    )
                )
        return KnowledgeGraphExtraction(
            entities=[entity for entity in entities if entity.confidence >= self.min_confidence],
            relationships=[rel for rel in relationships if rel.confidence >= self.min_confidence],
        )

    def _parse_json(self, response: str) -> KnowledgeGraphExtraction:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            payload = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return KnowledgeGraphExtraction()
        if not isinstance(payload, dict):
            return KnowledgeGraphExtraction()
        return KnowledgeGraphExtraction(
            entities=self._parse_entities(payload.get("entities")),
            relationships=self._parse_relationships(payload.get("relationships")),
        )

    def _parse_entities(self, rows: Any) -> list[ExtractedEntity]:
        if not isinstance(rows, list):
            return []
        entities: list[ExtractedEntity] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            confidence = _float(row.get("confidence"), default=1.0)
            if confidence < self.min_confidence:
                continue
            properties = row.get("properties")
            entities.append(
                ExtractedEntity(
                    name=str(row["name"]),
                    type=str(row.get("type", "entity")),
                    confidence=confidence,
                    properties=dict(properties) if isinstance(properties, dict) else {},
                )
            )
        return entities

    def _parse_relationships(self, rows: Any) -> list[ExtractedRelationship]:
        if not isinstance(rows, list):
            return []
        relationships: list[ExtractedRelationship] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("source")
            target = row.get("target")
            if not source or not target:
                continue
            confidence = _float(row.get("confidence"), default=1.0)
            if confidence < self.min_confidence:
                continue
            properties = row.get("properties")
            relationships.append(
                ExtractedRelationship(
                    source=str(source),
                    target=str(target),
                    relation=str(row.get("relation", "related_to")),
                    confidence=confidence,
                    properties=dict(properties) if isinstance(properties, dict) else {},
                )
            )
        return relationships


class GraphVectorStore(VectorStore):
    """Drop-in vector store that expands vector hits through a property graph."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings | None = None,
        *,
        vector_store: VectorStore | None = None,
        backend: PropertyGraphBackendName | PropertyGraphBackend = "networkx",
        extractor: KnowledgeGraphExtractor | None = None,
        max_hops: int = 2,
        min_confidence: float = 0.0,
        graph_weight: float = 0.25,
        uri: str | None = None,
        username: str = "neo4j",
        password: str = "password",
    ) -> None:
        if vector_store is None:
            if embedding_backend is None:
                raise ValueError("embedding_backend is required when vector_store is not provided")
            vector_store = InMemoryVectorStore(embedding_backend)
        self.vector_store = vector_store
        self.graph = self._make_backend(backend, uri=uri, username=username, password=password)
        self.extractor = extractor or KnowledgeGraphExtractor(store=self.graph)
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        self.graph_weight = graph_weight
        self._doc_text_by_id: dict[str, str] = {}
        self._doc_metadata_by_id: dict[str, dict[str, Any]] = {}

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        meta = metadata if metadata is not None else [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata length must match texts length")
        enriched: list[dict[str, Any]] = []
        for index, (text, item_meta) in enumerate(zip(texts, meta, strict=True)):
            source_doc = str(
                item_meta.get("source")
                or item_meta.get("id")
                or f"doc_{len(self._doc_text_by_id) + index + 1}"
            )
            doc_meta = {**item_meta, "source": source_doc}
            self._doc_text_by_id[source_doc] = text
            self._doc_metadata_by_id[source_doc] = doc_meta
            extraction = await self.extractor.extract(text)
            _store_extraction(self.graph, extraction, source_doc=source_doc)
            enriched.append(doc_meta)
        await self.vector_store.add(texts, enriched)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        vector_results = await self.vector_store.search(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        seed_ids = self._seed_ids(query, vector_results)
        if not seed_ids:
            return vector_results[:top_k]

        nodes, edges = self.graph.traverse(
            seed_ids,
            max_hops=self.max_hops,
            min_confidence=self.min_confidence,
        )
        graph_doc_ids = self.graph.document_ids_for_nodes([node.id for node in nodes])
        return self._fuse(vector_results, graph_doc_ids, nodes, edges, top_k, metadata_filter)

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        results = await self.vector_store.search_mmr(
            query,
            top_k=top_k,
            lambda_mult=lambda_mult,
            fetch_k=fetch_k,
            metadata_filter=metadata_filter,
        )
        return results

    def save(self, path: str) -> None:
        self.vector_store.save(path)

    def load(self, path: str) -> None:
        self.vector_store.load(path)

    def __len__(self) -> int:
        length = getattr(self.vector_store, "__len__", None)
        return int(length()) if callable(length) else len(self._doc_text_by_id)

    @staticmethod
    def _make_backend(
        backend: PropertyGraphBackendName | PropertyGraphBackend,
        *,
        uri: str | None,
        username: str,
        password: str,
    ) -> PropertyGraphBackend:
        if not isinstance(backend, str):
            return backend
        if backend == "networkx":
            return NetworkXPropertyGraphBackend()
        if backend == "neo4j":
            if uri is None:
                raise ValueError("uri is required when backend='neo4j'")
            return Neo4jPropertyGraphBackend(uri, username=username, password=password)
        raise ValueError(f"Unknown graph backend: {backend!r}")

    def _seed_ids(self, query: str, vector_results: list[dict]) -> list[str]:
        seed_ids: list[str] = []
        seed_for_text = getattr(self.graph, "seed_ids_for_text", None)
        if callable(seed_for_text):
            seed_ids.extend(seed_for_text(query))
        for result in vector_results:
            source = (result.get("metadata") or {}).get("source")
            if source is None:
                continue
            for node in self.graph.nodes():
                if str(node.properties.get("source_doc")) == str(source):
                    seed_ids.append(node.id)
        return list(dict.fromkeys(seed_ids))

    def _fuse(
        self,
        vector_results: list[dict],
        graph_doc_ids: list[str],
        nodes: list[PropertyGraphNode],
        edges: list[PropertyGraphEdge],
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[dict]:
        by_text: dict[str, dict[str, Any]] = {}
        for rank, result in enumerate(vector_results, start=1):
            item = dict(result)
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("retrieval_source", "vector")
            item["metadata"] = metadata
            item["score"] = float(item.get("score", 0.0)) + 1.0 / (100 + rank)
            by_text[str(item.get("text", ""))] = item

        for rank, doc_id in enumerate(graph_doc_ids, start=1):
            metadata = self._doc_metadata_by_id.get(doc_id, {"source": doc_id})
            if metadata_filter and any(metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            text = self._doc_text_by_id.get(doc_id)
            if not text:
                continue
            current = by_text.get(text, {"text": text, "score": 0.0, "metadata": dict(metadata)})
            current["score"] = float(current.get("score", 0.0)) + self.graph_weight / rank
            merged_meta = dict(current.get("metadata") or {})
            merged_meta.update(
                {
                    "retrieval_source": "graph_vector",
                    "graph_nodes": [node.label for node in nodes],
                    "graph_edges": [edge.relation for edge in edges],
                }
            )
            current["metadata"] = merged_meta
            by_text[text] = current

        return sorted(by_text.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _store_extraction(
    graph: PropertyGraphBackend,
    extraction: KnowledgeGraphExtraction,
    *,
    source_doc: str,
) -> None:
    extracted_at = datetime.now(UTC).isoformat()
    for entity in extraction.entities:
        node_id = _node_id(entity.name)
        graph.upsert_node(
            PropertyGraphNode(
                id=node_id,
                label=entity.name,
                type=entity.type,
                properties={
                    **entity.properties,
                    "confidence": entity.confidence,
                    "source_doc": source_doc,
                    "extracted_at": extracted_at,
                },
            )
        )
    for relationship in extraction.relationships:
        graph.upsert_edge(
            PropertyGraphEdge(
                source=_node_id(relationship.source),
                target=_node_id(relationship.target),
                relation=relationship.relation,
                properties={
                    **relationship.properties,
                    "confidence": relationship.confidence,
                    "source_doc": source_doc,
                    "extracted_at": extracted_at,
                },
            )
        )


def _coerce_document(document: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(document, str):
        return document, {}
    if isinstance(document, dict):
        return str(document.get("text", "")), dict(document.get("metadata", {}))
    return str(document), {}


def _node_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.casefold()).strip("_")
    return f"pg_{slug or 'node'}"


def _normalize(value: str) -> str:
    value = re.sub(r"^@", "", value.casefold().strip())
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _guess_entity_type(name: str) -> str:
    lowered = name.casefold()
    if lowered.startswith("@"):
        return "person"
    if any(token in lowered for token in (" inc", " corp", " labs", " llc", " ltd")):
        return "org"
    return "entity"


def _relation_for_sentence(sentence: str) -> str:
    lowered = sentence.casefold()
    for hint, relation in _RELATION_HINTS:
        if hint in lowered:
            return relation
    return "related_to"


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
