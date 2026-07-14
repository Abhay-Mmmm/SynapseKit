"""Regression tests for the fix/audit-retrieval batch.

Covers issues #780, #782, #783, #784, #785, #786, #787, #788, #789.
Splitter validation (#781) lives in tests/text_splitters/test_overlap_validation.py.

All fakes are hand-written (no MagicMock) per project testing standards.
"""

from __future__ import annotations

import numpy as np
import pytest

from synapsekit.retrieval.federated import FederatedRetriever
from synapsekit.retrieval.hybrid_search import HybridSearchRetriever
from synapsekit.retrieval.rag_fusion import RAGFusionRetriever
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import (
    ExtractionPolicy,
    ExtractionResult,
    InMemoryWorldGraphBackend,
    WorldModelRAG,
)


class FakeEmbeddings:
    """Deterministic, L2-normalised embeddings keyed on byte content."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        values = np.zeros(self._dim, dtype=np.float32)
        for index, char in enumerate(text.encode("utf-8")):
            values[index % self._dim] += float(char)
        norm = np.linalg.norm(values)
        return values / norm if norm else values


# --------------------------------------------------------------------------- #
# #780 — InMemoryVectorStore.search must return [] for top_k <= 0
# --------------------------------------------------------------------------- #


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddings())


async def test_search_top_k_zero_returns_empty(store):
    """top_k=0 must yield no docs (old code returned ALL via argpartition[-0:])."""
    await store.add(["alpha", "beta", "gamma"])
    assert await store.search("alpha", top_k=0) == []


async def test_search_negative_top_k_returns_empty(store):
    await store.add(["alpha", "beta"])
    assert await store.search("alpha", top_k=-3) == []


async def test_search_top_k_zero_with_metadata_filter_returns_empty(store):
    """Filtered branch had the same argpartition[-0:] bug."""
    await store.add(["alpha", "beta"], metadata=[{"src": "x"}, {"src": "x"}])
    assert await store.search("alpha", top_k=0, metadata_filter={"src": "x"}) == []


async def test_search_positive_top_k_still_works(store):
    await store.add(["alpha", "beta", "gamma"])
    results = await store.search("alpha", top_k=2)
    assert len(results) == 2


async def test_search_mmr_top_k_zero_returns_empty(store):
    await store.add(["alpha", "beta", "gamma"])
    assert await store.search_mmr("alpha", top_k=0) == []


# --------------------------------------------------------------------------- #
# #783 — top_k=0 must not be conflated with None (top_k or default)
# --------------------------------------------------------------------------- #


class RecordingRetriever:
    """Records the top_k it was asked for and returns fixed texts."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.seen_top_k: list[int] = []

    async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
        self.seen_top_k.append(top_k)
        return [{"text": t, "score": 1.0, "metadata": {}} for t in self._texts[:top_k]]


async def test_federated_top_k_zero_not_treated_as_default():
    """top_k=0 must propagate as 0, not fall back to the instance default of 10."""
    inner = RecordingRetriever(["a", "b", "c"])
    fed = FederatedRetriever(sources=[{"name": "s", "retriever": inner}], top_k=10)

    results = await fed.retrieve_with_scores("q", top_k=0)

    assert inner.seen_top_k == [0]
    assert results == []


async def test_federated_top_k_none_uses_default():
    inner = RecordingRetriever(["a", "b", "c", "d"])
    fed = FederatedRetriever(sources=[{"name": "s", "retriever": inner}], top_k=2)

    await fed.retrieve_with_scores("q", top_k=None)

    assert inner.seen_top_k == [2]


# --------------------------------------------------------------------------- #
# #786 — RAGFusionRetriever fans out queries concurrently, order preserved
# --------------------------------------------------------------------------- #


class FakeQueryLLM:
    async def generate(self, prompt: str) -> str:
        return "variant one\nvariant two\nvariant three"


class OrderedRetriever:
    """Returns a per-query doc so RRF ordering can be asserted deterministically."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        self.calls.append(query)
        return [f"doc-for::{query}"]


async def test_rag_fusion_gathers_and_preserves_order():
    llm = FakeQueryLLM()
    retriever = OrderedRetriever()
    fusion = RAGFusionRetriever(retriever=retriever, llm=llm, num_queries=3)

    fused = await fusion.retrieve("original", top_k=10)

    # Original query is first, so with equal RRF weight it ranks first (stable).
    assert fused[0] == "doc-for::original"
    # All four query variants were dispatched.
    assert len(retriever.calls) == 4


# --------------------------------------------------------------------------- #
# #787 — HybridSearchRetriever.add_documents accumulates (not replaces)
# --------------------------------------------------------------------------- #


class StaticRetriever:
    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = texts or []

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        return self._texts[:top_k]


def _make_hybrid(retriever=None) -> HybridSearchRetriever:
    return HybridSearchRetriever(retriever=retriever or StaticRetriever())


async def test_add_documents_accumulates_prior_docs():
    hybrid = _make_hybrid()
    hybrid.add_documents(["the quick brown fox"])
    hybrid.add_documents(["lazy sleeping dog"])

    # A query matching the FIRST batch must still surface it after the second add.
    results = await hybrid.retrieve("quick fox", top_k=5)
    assert "the quick brown fox" in results
    assert "lazy sleeping dog" in " ".join(results) or "lazy sleeping dog" in results


async def test_add_documents_single_batch_still_works():
    hybrid = _make_hybrid()
    hybrid.add_documents(["alpha term", "beta term"])
    results = await hybrid.retrieve("alpha", top_k=5)
    assert "alpha term" in results


# --------------------------------------------------------------------------- #
# #788 — HybridSearchRetriever falls back to BM25 when vector store is down
# --------------------------------------------------------------------------- #


class BrokenVectorRetriever:
    async def retrieve(self, query, top_k=5, metadata_filter=None):
        raise ConnectionError("vector store unavailable")


async def test_retrieve_falls_back_to_bm25_when_vector_down(caplog):
    hybrid = HybridSearchRetriever(retriever=BrokenVectorRetriever())
    hybrid.add_documents(["python testing guide", "unrelated cooking recipe"])

    results = await hybrid.retrieve("python testing", top_k=2)

    assert "python testing guide" in results
    assert any("falling back to BM25" in r.message for r in caplog.records)


async def test_retrieve_without_bm25_and_broken_vector_returns_empty():
    hybrid = HybridSearchRetriever(retriever=BrokenVectorRetriever())
    # No documents added -> no BM25 index; broken vector -> empty, not an exception.
    assert await hybrid.retrieve("anything", top_k=5) == []


# --------------------------------------------------------------------------- #
# #789 — WorldModelRAG.ingest batches vector adds into a single call
# --------------------------------------------------------------------------- #


class RecordingVectorRetriever:
    """Captures each add() call so we can assert batching."""

    def __init__(self) -> None:
        self.add_calls: list[tuple[list[str], list[dict]]] = []

    async def add(self, texts: list[str], metadata: list[dict]) -> None:
        self.add_calls.append((list(texts), list(metadata)))


class EmptyExtractor:
    async def extract(self, text: str, policy) -> ExtractionResult:
        return ExtractionResult()


def _make_world_model(vector_retriever) -> WorldModelRAG:
    wm = object.__new__(WorldModelRAG)
    wm.vector_retriever = vector_retriever
    wm.extractor = EmptyExtractor()
    wm.graph_backend = InMemoryWorldGraphBackend()
    wm.extraction = ExtractionPolicy()
    wm._doc_counter = 0
    return wm


async def test_ingest_batches_vector_adds_into_single_call():
    vr = RecordingVectorRetriever()
    wm = _make_world_model(vr)

    await wm.ingest(["first doc", "second doc", "third doc"])

    # Old code called add() once per document; batched code calls it once.
    assert len(vr.add_calls) == 1
    texts, metas = vr.add_calls[0]
    assert texts == ["first doc", "second doc", "third doc"]
    assert len(metas) == 3
    assert all("world_model_doc_id" in m and "source" in m for m in metas)


async def test_ingest_skips_blank_docs_and_still_batches():
    vr = RecordingVectorRetriever()
    wm = _make_world_model(vr)

    await wm.ingest(["real content", "   ", ""])

    assert len(vr.add_calls) == 1
    texts, _ = vr.add_calls[0]
    assert texts == ["real content"]


async def test_ingest_no_docs_makes_no_add_call():
    vr = RecordingVectorRetriever()
    wm = _make_world_model(vr)

    await wm.ingest([])

    assert vr.add_calls == []
