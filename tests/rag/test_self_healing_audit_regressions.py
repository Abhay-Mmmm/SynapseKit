"""Regression tests for self-healing RAG audit fixes (issues #791, #793).

Hand-written fakes only — no MagicMock.

  #791 — token totals recorded per-call as the running cumulative sum.
  #793 — top_k=0 conflated with the None default.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from synapsekit.observability.tracer import TokenTracer
from synapsekit.rag.self_healing import SelfHealingRAG


@dataclass
class _Score:
    score: float


class FakeMetric:
    def __init__(self, score: float) -> None:
        self._score = score

    async def evaluate(self, *, question, answer, contexts):
        return _Score(self._score)


class FakeLLM:
    """Cumulative token counters, like real providers."""

    def __init__(self, per_call_input: int = 100, per_call_output: int = 50) -> None:
        self.tokens_used = {"input": 0, "output": 0}
        self._pi = per_call_input
        self._po = per_call_output

    async def generate_with_messages(self, messages, **kw):
        self.tokens_used["input"] += self._pi
        self.tokens_used["output"] += self._po
        return "an answer"


class RecordingStrategy:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.top_ks: list[int] = []

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        self.top_ks.append(top_k)
        return list(self._chunks)


@pytest.mark.asyncio
async def test_self_healing_records_token_delta_not_cumulative():
    """Two attempts of 100/50 tokens each must total 200/100, not 300/200.

    Fails on old code, which records ``tokens_used`` (a growing total) each
    attempt, so the tracer sums 100 + 200 = 300.
    """
    tracer = TokenTracer(model="gpt-4o-mini")
    llm = FakeLLM(per_call_input=100, per_call_output=50)

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[RecordingStrategy(["c1"]), RecordingStrategy(["c2"])],
        quality_threshold=0.75,
        max_retries=2,
        # First attempt fails the threshold, second passes → 2 LLM calls.
        metric=_TwoStepMetric([0.2, 0.9]),
        tracer=tracer,
    )

    await rag.ask("q")

    summary = tracer.summary()
    assert summary["calls"] == 2
    assert summary["total_input_tokens"] == 200
    assert summary["total_output_tokens"] == 100


class _TwoStepMetric:
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._i = 0

    async def evaluate(self, *, question, answer, contexts):
        score = self._scores[min(self._i, len(self._scores) - 1)]
        self._i += 1
        return _Score(score)


@pytest.mark.asyncio
async def test_self_healing_top_k_zero_respected():
    """top_k=0 must be forwarded to the strategy, not replaced by the default.

    Fails on old code (`top_k or self._retrieval_top_k` turns 0 into 5).
    """
    strategy = RecordingStrategy(["c"])
    rag = SelfHealingRAG(
        llm=FakeLLM(),
        strategies=[strategy],
        quality_threshold=0.0,  # first attempt succeeds
        retrieval_top_k=5,
        metric=FakeMetric(1.0),
    )

    await rag.ask("q", top_k=0)

    assert strategy.top_ks == [0]


@pytest.mark.asyncio
async def test_self_healing_top_k_none_uses_default():
    strategy = RecordingStrategy(["c"])
    rag = SelfHealingRAG(
        llm=FakeLLM(),
        strategies=[strategy],
        quality_threshold=0.0,
        retrieval_top_k=9,
        metric=FakeMetric(1.0),
    )

    await rag.ask("q")

    assert strategy.top_ks == [9]
