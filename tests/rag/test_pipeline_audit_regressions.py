"""Regression tests for the RAG audit batch (issues #791, #793, #795).

These use hand-written fakes only — no MagicMock — so the assertions exercise
the real code paths that broke in production.

Covered bugs:
  #791 (pipeline token totals recorded as cumulative running total)
  #793 (top_k=0 conflated with None default)
  #795 (metadata dict shared across every chunk on add)
"""

from __future__ import annotations

import pytest

from synapsekit.memory.conversation import ConversationMemory
from synapsekit.observability.tracer import TokenTracer
from synapsekit.rag.pipeline import RAGConfig, RAGPipeline


class FakeLLM:
    """LLM whose ``tokens_used`` accumulates across calls like real providers."""

    def __init__(self, per_call_input: int = 100, per_call_output: int = 50) -> None:
        self.tokens_used = {"input": 0, "output": 0}
        self._per_call_input = per_call_input
        self._per_call_output = per_call_output

    async def stream_with_messages(self, messages, **kw):
        # Simulate a provider bumping cumulative token counters per call.
        self.tokens_used["input"] += self._per_call_input
        self.tokens_used["output"] += self._per_call_output
        for tok in ("Hello", " world"):
            yield tok


class FakeRetriever:
    """Records the top_k it was asked for; returns fixed chunks."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self._chunks = chunks if chunks is not None else ["chunk a", "chunk b"]
        self.retrieve_calls: list[int] = []
        self.added: list[tuple[list[str], list[dict]]] = []

    async def retrieve(self, query: str, top_k: int = 5):
        self.retrieve_calls.append(top_k)
        return list(self._chunks)

    async def add(self, chunks, metadata):
        self.added.append((list(chunks), list(metadata)))


def _make_pipeline(llm=None, retriever=None, tracer=None):
    return RAGPipeline(
        RAGConfig(
            llm=llm or FakeLLM(),
            retriever=retriever or FakeRetriever(),
            memory=ConversationMemory(),
            tracer=tracer,
        )
    )


# ---------------------------------------------------------------------------
# #791 — token deltas, not cumulative totals
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tracer_records_per_call_token_delta_not_cumulative():
    """Three calls of 100/50 tokens each must total 300/150, not 600/300.

    Fails on old code (records the running cumulative total each call).
    """
    tracer = TokenTracer(model="gpt-4o-mini")
    llm = FakeLLM(per_call_input=100, per_call_output=50)
    pipeline = _make_pipeline(llm=llm, tracer=tracer)

    for _ in range(3):
        await pipeline.ask("question?")

    summary = tracer.summary()
    assert summary["calls"] == 3
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 150


@pytest.mark.asyncio
async def test_single_call_token_delta_matches_usage():
    tracer = TokenTracer(model="gpt-4o-mini")
    llm = FakeLLM(per_call_input=42, per_call_output=7)
    pipeline = _make_pipeline(llm=llm, tracer=tracer)

    await pipeline.ask("q?")

    summary = tracer.summary()
    assert summary["total_input_tokens"] == 42
    assert summary["total_output_tokens"] == 7


# ---------------------------------------------------------------------------
# #793 — top_k=0 must not fall back to the default
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_top_k_zero_is_respected_and_not_defaulted():
    """top_k=0 must be passed through as 0, not replaced by retrieval_top_k.

    Fails on old code (`top_k or default` turns 0 into the default of 5).
    """
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        RAGConfig(
            llm=FakeLLM(),
            retriever=retriever,
            memory=ConversationMemory(),
            retrieval_top_k=5,
        )
    )

    await pipeline.ask("q?", top_k=0)

    assert retriever.retrieve_calls == [0]


@pytest.mark.asyncio
async def test_top_k_none_uses_configured_default():
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        RAGConfig(
            llm=FakeLLM(),
            retriever=retriever,
            memory=ConversationMemory(),
            retrieval_top_k=7,
        )
    )

    await pipeline.ask("q?")

    assert retriever.retrieve_calls == [7]


# ---------------------------------------------------------------------------
# #795 — per-chunk metadata must be independent dicts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_add_gives_each_chunk_an_independent_metadata_dict():
    """Mutating one chunk's metadata must not leak into the others.

    Fails on old code (`[metadata or {} for _ in chunks]` shares one dict).
    """
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        RAGConfig(
            llm=FakeLLM(),
            retriever=retriever,
            memory=ConversationMemory(),
            chunk_size=8,
            chunk_overlap=0,
        )
    )

    long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    await pipeline.add(long_text, metadata={"source": "doc1"})

    assert retriever.added, "retriever.add should have been called"
    _chunks, metas = retriever.added[0]
    assert len(metas) >= 2, "text should split into multiple chunks for this test"

    # Distinct objects.
    assert all(m is not metas[0] for m in metas[1:])

    # Mutating one does not affect the rest.
    metas[0]["mutated"] = True
    assert all("mutated" not in m for m in metas[1:])


@pytest.mark.asyncio
async def test_add_with_none_metadata_still_produces_independent_dicts():
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        RAGConfig(
            llm=FakeLLM(),
            retriever=retriever,
            memory=ConversationMemory(),
            chunk_size=8,
            chunk_overlap=0,
        )
    )

    await pipeline.add("alpha beta gamma delta epsilon zeta", metadata=None)

    _chunks, metas = retriever.added[0]
    if len(metas) >= 2:
        metas[0]["x"] = 1
        assert all("x" not in m for m in metas[1:])
