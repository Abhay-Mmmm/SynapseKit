# Graph RAG

SynapseKit property-graph RAG combines vector search with graph traversal. It is useful when a
question depends on relationships between entities rather than a single semantically similar chunk.

## GraphVectorStore

`GraphVectorStore` implements the same async vector-store interface as the other retrieval
backends. Documents are embedded into an underlying vector store and also passed through
`KnowledgeGraphExtractor`, which stores entities and relationships in a property graph.

```python
from synapsekit.embeddings import SynapsekitEmbeddings
from synapsekit.retrieval import GraphVectorStore

embeddings = SynapsekitEmbeddings()
store = GraphVectorStore(embeddings, backend="networkx")

await store.add(
    ["Dana leads Apollo. Apollo is built by Platform Team."],
    metadata=[{"source": "apollo-brief"}],
)

results = await store.search("Who leads the project built by Platform Team?", top_k=3)
```

Search starts with vector results, selects graph seeds from the query and matching documents,
traverses the graph, then fuses graph-linked documents back into the ranked results. Returned
items keep the standard shape:

```python
{"text": "...", "score": 0.91, "metadata": {"source": "apollo-brief"}}
```

## KnowledgeGraphExtractor

The extractor accepts an optional LLM. Without one, it uses a deterministic local extractor for
tests and offline development.

```python
from synapsekit.retrieval import KnowledgeGraphExtractor, NetworkXPropertyGraphBackend

graph = NetworkXPropertyGraphBackend()
extractor = KnowledgeGraphExtractor(store=graph)

await extractor.ingest(
    [{"text": "Dana leads Apollo.", "metadata": {"source": "doc-1"}}]
)
```

Nodes and edges include `confidence`, `source_doc`, and `extracted_at` properties. LLM extraction
expects strict JSON with `entities` and `relationships` arrays, so custom LLM providers can be used
without changing graph storage.

## Backends

Use the in-memory backend for local development and tests:

```python
store = GraphVectorStore(embeddings, backend="networkx")
```

Use Neo4j for production deployments:

```python
store = GraphVectorStore(
    embeddings,
    backend="neo4j",
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
)
```

Neo4j support is optional. Install the graph extra before using it:

```bash
uv sync --extra graph
```

## Agent Memory

`AgentMemory` also accepts a graph backend. Each memory record is stored as a node, and callers can
connect memories by passing `related_to`, `relation_type`, and `weight` metadata.

```python
from synapsekit.memory import AgentMemory
from synapsekit.retrieval import NetworkXPropertyGraphBackend

graph = NetworkXPropertyGraphBackend()
memory = AgentMemory(backend="graph", store=graph)

first = await memory.store(
    agent_id="agent-a",
    content="User prefers concise Python answers",
    memory_type="semantic",
)

await memory.store(
    agent_id="agent-a",
    content="User is building a graph RAG prototype",
    memory_type="episodic",
    metadata={"related_to": [first.id], "relation_type": "supports", "weight": 0.8},
)
```

The backend preserves the normal `AgentMemory` methods: `store`, `recall`, `list`, `count`,
`delete`, `clear`, and TTL pruning.
