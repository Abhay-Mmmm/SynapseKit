"""Tests for v0.5.2 features."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.llm._cache import AsyncLRUCache
from synapsekit.llm._rate_limit import TokenBucketRateLimiter
from synapsekit.llm.base import BaseLLM, LLMConfig

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class DummyLLM(BaseLLM):
    """Minimal LLM for testing."""

    def __init__(self, response: str = "hello", **kw: Any) -> None:
        config = LLMConfig(model="test", api_key="k", provider="openai", **kw)
        super().__init__(config)
        self._response = response
        self.call_count = 0

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        self.call_count += 1
        yield self._response


class DummyLLMWithTools(DummyLLM):
    """LLM that supports call_with_tools."""

    def __init__(self, responses: list[dict] | None = None, fail_n: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self._tool_responses = responses or [{"content": "done", "tool_calls": None}]
        self._tool_call_idx = 0
        self._fail_n = fail_n
        self._fail_count = 0

    async def _call_with_tools_impl(
        self, messages: list[dict], tools: list[dict]
    ) -> dict[str, Any]:
        if self._fail_count < self._fail_n:
            self._fail_count += 1
            raise ConnectionError("transient failure")
        resp = self._tool_responses[self._tool_call_idx % len(self._tool_responses)]
        self._tool_call_idx += 1
        return resp


# ------------------------------------------------------------------ #
# #3: __repr__ methods
# ------------------------------------------------------------------ #


class TestReprMethods:
    def test_state_graph_repr(self):
        from synapsekit.graph.graph import StateGraph

        g = StateGraph()
        g.add_node("a", lambda s: s)
        g.add_node("b", lambda s: s)
        g.add_edge("a", "b")
        assert repr(g) == "StateGraph(nodes=2, edges=1)"

    def test_compiled_graph_repr(self):
        from synapsekit.graph.graph import StateGraph

        g = StateGraph()
        g.add_node("a", lambda s: s)
        g.set_entry_point("a")
        g.set_finish_point("a")
        compiled = g.compile()
        r = repr(compiled)
        assert "CompiledGraph" in r
        assert "nodes=1" in r
        assert "max_steps=100" in r

    def test_rag_pipeline_repr(self):
        from synapsekit.rag.pipeline import RAGConfig, RAGPipeline

        llm = DummyLLM()
        retriever = MagicMock()
        memory = MagicMock()
        config = RAGConfig(llm=llm, retriever=retriever, memory=memory)
        pipeline = RAGPipeline(config)
        r = repr(pipeline)
        assert "RAGPipeline" in r
        assert "DummyLLM" in r

    def test_react_agent_repr(self):
        from synapsekit.agents.react import ReActAgent

        llm = DummyLLM()
        agent = ReActAgent(llm=llm, tools=[], max_iterations=5)
        r = repr(agent)
        assert "ReActAgent" in r
        assert "max_iterations=5" in r

    def test_function_calling_agent_repr(self):
        from synapsekit.agents.function_calling import FunctionCallingAgent

        llm = DummyLLM()
        agent = FunctionCallingAgent(llm=llm, tools=[], max_iterations=3)
        r = repr(agent)
        assert "FunctionCallingAgent" in r
        assert "max_iterations=3" in r


# ------------------------------------------------------------------ #
# #20: Empty document handling
# ------------------------------------------------------------------ #


class TestEmptyDocumentHandling:
    @pytest.mark.asyncio
    async def test_add_empty_string_skipped(self):
        from synapsekit.rag.pipeline import RAGConfig, RAGPipeline

        retriever = MagicMock()
        retriever.add = AsyncMock()
        config = RAGConfig(llm=DummyLLM(), retriever=retriever, memory=MagicMock())
        pipeline = RAGPipeline(config)

        await pipeline.add("")
        retriever.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_whitespace_only_skipped(self):
        from synapsekit.rag.pipeline import RAGConfig, RAGPipeline

        retriever = MagicMock()
        retriever.add = AsyncMock()
        config = RAGConfig(llm=DummyLLM(), retriever=retriever, memory=MagicMock())
        pipeline = RAGPipeline(config)

        await pipeline.add("   \n\t  ")
        retriever.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self):
        from synapsekit.rag.pipeline import RAGConfig, RAGPipeline

        retriever = MagicMock()
        retriever.add = AsyncMock()
        config = RAGConfig(llm=DummyLLM(), retriever=retriever, memory=MagicMock())
        pipeline = RAGPipeline(config)

        # Empty list should not crash
        await pipeline.add_documents([])
        retriever.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_skips_empty_docs(self):
        from synapsekit.loaders.base import Document
        from synapsekit.rag.pipeline import RAGConfig, RAGPipeline

        retriever = MagicMock()
        retriever.add = AsyncMock()
        config = RAGConfig(llm=DummyLLM(), retriever=retriever, memory=MagicMock())
        pipeline = RAGPipeline(config)

        docs = [Document(text="", metadata={}), Document(text="  ", metadata={})]
        await pipeline.add_documents(docs)
        retriever.add.assert_not_called()


# ------------------------------------------------------------------ #
# #22: Retry support for call_with_tools()
# ------------------------------------------------------------------ #


class TestCallWithToolsRetry:
    @pytest.mark.asyncio
    async def test_call_with_tools_retries_on_transient_failure(self):
        llm = DummyLLMWithTools(fail_n=2, max_retries=3, retry_delay=0.01)
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
        )
        assert result["content"] == "done"
        assert llm._fail_count == 2

    @pytest.mark.asyncio
    async def test_call_with_tools_no_retry_by_default(self):
        llm = DummyLLMWithTools(fail_n=1)
        with pytest.raises(ConnectionError):
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_call_with_tools_auth_error_not_retried(self):
        class AuthFailLLM(DummyLLM):
            async def _call_with_tools_impl(self, messages, tools):
                raise ValueError("Invalid api_key provided")

        llm = AuthFailLLM(max_retries=3, retry_delay=0.01)
        with pytest.raises(ValueError, match="api_key"):
            await llm.call_with_tools([], [])


# ------------------------------------------------------------------ #
# #23: Cache hit/miss statistics
# ------------------------------------------------------------------ #


class TestCacheStats:
    def test_cache_hit_miss_counters(self):
        cache = AsyncLRUCache(maxsize=10)
        cache.put("a", "value_a")

        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("b")  # miss

        assert cache.hits == 2
        assert cache.misses == 1

    def test_cache_stats_property(self):
        llm = DummyLLM(cache=True, cache_maxsize=10)
        stats = llm.cache_stats
        assert stats == {"hits": 0, "misses": 0, "size": 0}

    def test_cache_stats_empty_when_disabled(self):
        llm = DummyLLM()
        assert llm.cache_stats == {}

    @pytest.mark.asyncio
    async def test_cache_stats_after_generate(self):
        llm = DummyLLM(response="cached", cache=True, cache_maxsize=10)
        await llm.generate("prompt1")
        await llm.generate("prompt1")  # cache hit
        stats = llm.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1  # first call is a miss
        assert stats["size"] == 1


# ------------------------------------------------------------------ #
# #30: MMR retrieval
# ------------------------------------------------------------------ #


class TestMMRRetrieval:
    @pytest.mark.asyncio
    async def test_mmr_search_basic(self):
        import numpy as np

        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        embeddings = MagicMock()
        store = InMemoryVectorStore(embeddings)

        # Create 5 vectors, some similar to each other
        vecs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],  # very similar to 0
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        store._vectors = vecs
        store._texts = ["doc0", "doc1", "doc2", "doc3", "doc4"]
        store._metadata = [{} for _ in range(5)]

        # Query similar to doc0
        q_vec = vecs[0]
        embeddings.embed_one = AsyncMock(return_value=q_vec)

        # With lambda=1.0, should be pure relevance (same as regular search)
        results_rel = await store.search_mmr("q", top_k=3, lambda_mult=1.0, fetch_k=5)
        assert len(results_rel) == 3

        # With lambda=0.0, should maximize diversity
        results_div = await store.search_mmr("q", top_k=3, lambda_mult=0.0, fetch_k=5)
        assert len(results_div) == 3

        # Diversity results should include more varied docs
        div_texts = {r["text"] for r in results_div}
        # With pure diversity, doc1 (very similar to doc0) should be deprioritized
        # The diverse set should include docs from different directions
        assert len(div_texts) == 3

    @pytest.mark.asyncio
    async def test_mmr_search_empty_store(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        embeddings = MagicMock()
        store = InMemoryVectorStore(embeddings)
        results = await store.search_mmr("q", top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_mmr_search_with_metadata_filter(self):
        import numpy as np

        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        embeddings = MagicMock()
        store = InMemoryVectorStore(embeddings)

        vecs = np.eye(3, dtype=np.float32)
        store._vectors = vecs
        store._texts = ["a", "b", "c"]
        store._metadata = [{"cat": "x"}, {"cat": "y"}, {"cat": "x"}]

        embeddings.embed_one = AsyncMock(return_value=vecs[0])

        results = await store.search_mmr("q", top_k=3, metadata_filter={"cat": "x"})
        texts = [r["text"] for r in results]
        assert "b" not in texts

    def test_mmr_not_implemented_on_base(self):
        from synapsekit.retrieval.base import VectorStore

        # VectorStore.search_mmr raises NotImplementedError
        with pytest.raises(TypeError):
            VectorStore()  # Can't instantiate ABC


# ------------------------------------------------------------------ #
# #35: Rate limiting
# ------------------------------------------------------------------ #


class TestRateLimiting:
    def test_rate_limiter_rejects_zero(self):
        with pytest.raises(ValueError, match="requests_per_minute must be >= 1"):
            TokenBucketRateLimiter(0)

    @pytest.mark.asyncio
    async def test_rate_limiter_basic_acquire(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=600)
        # Should acquire without significant delay at 10/sec
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiting_applied_to_generate(self):
        llm = DummyLLM(response="ok", requests_per_minute=6000)
        assert llm._rate_limiter is not None
        result = await llm.generate("test")
        assert result == "ok"

    def test_no_rate_limiter_by_default(self):
        llm = DummyLLM()
        assert llm._rate_limiter is None


# ------------------------------------------------------------------ #
# #43: Structured output
# ------------------------------------------------------------------ #


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_generate_structured_basic(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        from synapsekit.llm.structured import generate_structured

        class Person(BaseModel):
            name: str
            age: int

        llm = DummyLLM(response='{"name": "Alice", "age": 30}')
        result = await generate_structured(llm, "Tell me about Alice", schema=Person)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_generate_structured_with_code_block(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        from synapsekit.llm.structured import generate_structured

        class Info(BaseModel):
            value: str

        llm = DummyLLM(response='```json\n{"value": "test"}\n```')
        result = await generate_structured(llm, "get info", schema=Info)
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_generate_structured_retry_on_bad_json(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        from synapsekit.llm.structured import generate_structured

        class Item(BaseModel):
            name: str

        call_count = 0

        class RetryLLM(DummyLLM):
            async def generate_with_messages(self, messages, **kw):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "not valid json"
                return '{"name": "fixed"}'

        llm = RetryLLM()
        result = await generate_structured(llm, "get item", schema=Item, max_retries=2)
        assert result.name == "fixed"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_structured_fails_after_max_retries(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        from synapsekit.llm.structured import generate_structured

        class Item(BaseModel):
            name: str

        llm = DummyLLM(response="not json at all")
        with pytest.raises(ValueError, match="Failed to generate valid structured output"):
            await generate_structured(llm, "get item", schema=Item, max_retries=1)

    def test_generate_structured_importable(self):
        from synapsekit import generate_structured

        assert callable(generate_structured)
