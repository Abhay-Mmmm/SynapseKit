"""Integration tests for all OpenAI-compatible LLM providers.

All providers share the same mock pattern:
  - async generator for streaming chunks
  - MagicMock response object for tool-calling
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.cerebras import CerebrasLLM
from synapsekit.llm.moonshot import MoonshotLLM
from synapsekit.llm.novita import NovitaLLM
from synapsekit.llm.perplexity import PerplexityLLM
from synapsekit.llm.sambanova import SambaNovaLLM
from synapsekit.llm.writer import WriterLLM
from synapsekit.llm.xai import XaiLLM
from synapsekit.llm.zhipu import ZhipuLLM


def make_config(**kwargs) -> LLMConfig:
    return LLMConfig(
        model="test-model",
        api_key="sk-test",
        provider="openai",
        **kwargs,
    )


def _chunk(content: str | None):
    c = MagicMock()
    c.usage = None
    if content is not None:
        c.choices = [MagicMock(delta=MagicMock(content=content))]
    else:
        c.choices = []
    return c


def _usage_chunk(prompt_tokens: int = 10, completion_tokens: int = 5):
    c = MagicMock()
    c.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    c.choices = []
    return c


async def _async_stream(*chunks):
    for chunk in chunks:
        yield chunk


def _tool_response():
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "my_tool"
    tc.function.arguments = '{"x": 1}'
    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _content_response(text: str = "Just text"):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=8, completion_tokens=3)
    return resp


def _mock_openai_module(mock_client, azure: bool = False):
    mock_mod = MagicMock()
    if azure:
        mock_mod.AsyncAzureOpenAI.return_value = mock_client
    else:
        mock_mod.AsyncOpenAI.return_value = mock_client
    return mock_mod


# ---------------------------------------------------------------------------
# GroqLLM
# ---------------------------------------------------------------------------


class TestGroqLLM:
    def _make(self):
        from synapsekit.llm.groq import GroqLLM

        return GroqLLM(make_config())

    def _mock_groq(self, mock_client):
        mod = MagicMock()
        mod.AsyncGroq.return_value = mock_client
        return mod

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"groq": None}):
            with pytest.raises(ImportError, match="groq package required"):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.groq import GroqLLM

        assert inspect.isasyncgenfunction(GroqLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.groq import GroqLLM

        assert inspect.isasyncgenfunction(GroqLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Hello"), _chunk(" world"))
        )
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Hello"), _chunk(None), _chunk(" world"))
        )
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens
        assert "Hello" in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("A"), _chunk("B"))
        )
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["A", "B"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.8)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=128)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 128

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Hello"), _chunk(" world"))
        )
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"), _chunk("c"))
        )
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2
        assert mock_mod.AsyncGroq.call_count == 1

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "call tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"
        assert result["tool_calls"][0]["arguments"] == {"x": 1}

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = self._mock_groq(mock_client)
        with patch.dict("sys.modules", {"groq": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"] is None
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# DeepSeekLLM
# ---------------------------------------------------------------------------


class TestDeepSeekLLM:
    def _make(self):
        from synapsekit.llm.deepseek import DeepSeekLLM

        return DeepSeekLLM(make_config())

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.deepseek import DeepSeekLLM

        assert inspect.isasyncgenfunction(DeepSeekLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.deepseek import DeepSeekLLM

        assert inspect.isasyncgenfunction(DeepSeekLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Foo"), _chunk("Bar"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Foo", "Bar"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Foo"), _chunk(None))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens
        assert "Foo" in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("X")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["X"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.5)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=200)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Deep"), _chunk("Seek"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "DeepSeek"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 2

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "use tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"
        assert result["tool_calls"] is None


# ---------------------------------------------------------------------------
# OpenRouterLLM
# ---------------------------------------------------------------------------


class TestOpenRouterLLM:
    def _make(self):
        from synapsekit.llm.openrouter import OpenRouterLLM

        return OpenRouterLLM(make_config())

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.openrouter import OpenRouterLLM

        assert inspect.isasyncgenfunction(OpenRouterLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.openrouter import OpenRouterLLM

        assert inspect.isasyncgenfunction(OpenRouterLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Open"), _chunk("Router"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Open", "Router"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("text"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("hello")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["hello"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.3)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=64)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 64

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("open"), _chunk("router"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "openrouter"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"), _chunk("c"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# TogetherLLM
# ---------------------------------------------------------------------------


class TestTogetherLLM:
    def _make(self):
        from synapsekit.llm.together import TogetherLLM

        return TogetherLLM(make_config())

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.together import TogetherLLM

        assert inspect.isasyncgenfunction(TogetherLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.together import TogetherLLM

        assert inspect.isasyncgenfunction(TogetherLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Together"), _chunk("AI"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Together", "AI"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("valid"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens
        assert "valid" in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("tok")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["tok"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.1)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=300)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 300

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("hello")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 2

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# FireworksLLM
# ---------------------------------------------------------------------------


class TestFireworksLLM:
    def _make(self):
        from synapsekit.llm.fireworks import FireworksLLM

        return FireworksLLM(make_config())

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.fireworks import FireworksLLM

        assert inspect.isasyncgenfunction(FireworksLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.fireworks import FireworksLLM

        assert inspect.isasyncgenfunction(FireworksLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Fire"), _chunk("works"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Fire", "works"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("ok"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("spark")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["spark"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.6)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.6

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=400)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 400

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("fire"), _chunk("works"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "fireworks"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 4

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "use tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# AzureOpenAILLM
# ---------------------------------------------------------------------------


class TestAzureOpenAILLM:
    def _make(self, azure_endpoint: str | None = "https://test.openai.azure.com"):
        from synapsekit.llm.azure_openai import AzureOpenAILLM

        return AzureOpenAILLM(make_config(), azure_endpoint=azure_endpoint)

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_missing_azure_endpoint_raises(self):
        llm = self._make(azure_endpoint=None)
        mock_mod = MagicMock()
        mock_mod.AsyncAzureOpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            with pytest.raises(ValueError, match="azure_endpoint"):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.azure_openai import AzureOpenAILLM

        assert inspect.isasyncgenfunction(AzureOpenAILLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.azure_openai import AzureOpenAILLM

        assert inspect.isasyncgenfunction(AzureOpenAILLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("Azure"), _chunk("OpenAI"))
        )
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Azure", "OpenAI"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("text"))
        )
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("az")))
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["az"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.4)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.4

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=512)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("azure")))
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "azure"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        usage_c = _usage_chunk(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("x"), usage_c)
        )
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._input_tokens == 10
        assert llm._output_tokens == 5

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "use tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client, azure=True)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# Helpers for providers sharing the same openai AsyncOpenAI pattern
# These providers: Moonshot, Zhipu, SambaNova, XAI, Novita, Writer, Perplexity,
#                  Cerebras, LMStudio, VLLM
# ---------------------------------------------------------------------------


def _run_basic_openai_compat_tests(provider_cls, token_counting="per_chunk"):
    """Return a test class for an OpenAI-compat provider that uses AsyncOpenAI."""

    class _Tests:
        def _make(self, **kw):
            return provider_cls(make_config(), **kw)

        def test_missing_dep_raises(self):
            llm = self._make()
            with patch.dict("sys.modules", {"openai": None}):
                with pytest.raises(ImportError):
                    llm._get_client()

        def test_stream_is_async_gen_function(self):
            assert inspect.isasyncgenfunction(provider_cls.stream)

        def test_stream_with_messages_is_async_gen_function(self):
            assert inspect.isasyncgenfunction(provider_cls.stream_with_messages)

        @pytest.mark.asyncio
        async def test_stream_yields_tokens(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_stream(_chunk("Tok1"), _chunk("Tok2"))
            )
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                tokens = [t async for t in llm.stream("hi")]
            assert tokens == ["Tok1", "Tok2"]

        @pytest.mark.asyncio
        async def test_stream_skips_none_content(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_stream(_chunk(None), _chunk("valid"))
            )
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                tokens = [t async for t in llm.stream("hi")]
            assert None not in tokens

        @pytest.mark.asyncio
        async def test_stream_with_messages_yields_tokens(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_stream(_chunk("msg"))
            )
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                tokens = [
                    t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
                ]
            assert tokens == ["msg"]

        @pytest.mark.asyncio
        async def test_stream_passes_temperature(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                [t async for t in llm.stream("hi", temperature=0.77)]
            _, kw = mock_client.chat.completions.create.call_args
            assert kw["temperature"] == 0.77

        @pytest.mark.asyncio
        async def test_stream_passes_max_tokens(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                [t async for t in llm.stream("hi", max_tokens=99)]
            _, kw = mock_client.chat.completions.create.call_args
            assert kw["max_tokens"] == 99

        @pytest.mark.asyncio
        async def test_generate_joins_tokens(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_stream(_chunk("hello"), _chunk(" there"))
            )
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                result = await llm.generate("hi")
            assert result == "hello there"

        @pytest.mark.asyncio
        async def test_token_counting(self):
            llm = self._make()
            mock_client = MagicMock()
            if token_counting == "usage_chunk":
                usage_c = _usage_chunk(prompt_tokens=10, completion_tokens=5)
                mock_client.chat.completions.create = AsyncMock(
                    return_value=_async_stream(_chunk("a"), _chunk("b"), usage_c)
                )
            else:
                mock_client.chat.completions.create = AsyncMock(
                    return_value=_async_stream(_chunk("a"), _chunk("b"), _chunk("c"))
                )
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                [t async for t in llm.stream("hi")]
            if token_counting == "usage_chunk":
                assert llm._output_tokens == 5
            else:
                assert llm._output_tokens == 3

        def test_client_is_cached(self):
            mock_client = MagicMock()
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                llm = self._make()
                c1 = llm._get_client()
                c2 = llm._get_client()
            assert c1 is c2

        @pytest.mark.asyncio
        async def test_call_with_tools_returns_tool_call(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                result = await llm.call_with_tools(
                    [{"role": "user", "content": "use tool"}],
                    [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
                )
            assert result["tool_calls"] is not None
            assert result["tool_calls"][0]["name"] == "my_tool"
            assert result["tool_calls"][0]["arguments"] == {"x": 1}

        @pytest.mark.asyncio
        async def test_call_with_tools_returns_content(self):
            llm = self._make()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
            mock_mod = _mock_openai_module(mock_client)
            with patch.dict("sys.modules", {"openai": mock_mod}):
                result = await llm.call_with_tools(
                    [{"role": "user", "content": "hi"}],
                    [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
                )
            assert result["tool_calls"] is None
            assert result["content"] == "Just text"

    return _Tests


# ---------------------------------------------------------------------------
# MoonshotLLM
# ---------------------------------------------------------------------------


_MoonshotBase = _run_basic_openai_compat_tests(MoonshotLLM, token_counting="per_chunk")


class TestMoonshotLLM(_MoonshotBase):
    pass


# ---------------------------------------------------------------------------
# ZhipuLLM
# ---------------------------------------------------------------------------


_ZhipuBase = _run_basic_openai_compat_tests(ZhipuLLM, token_counting="per_chunk")


class TestZhipuLLM(_ZhipuBase):
    pass


# ---------------------------------------------------------------------------
# SambaNovaLLM
# ---------------------------------------------------------------------------


_SambaNovaBase = _run_basic_openai_compat_tests(SambaNovaLLM, token_counting="per_chunk")


class TestSambaNovaLLM(_SambaNovaBase):
    pass


# ---------------------------------------------------------------------------
# XaiLLM
# ---------------------------------------------------------------------------


_XaiBase = _run_basic_openai_compat_tests(XaiLLM, token_counting="per_chunk")


class TestXaiLLM(_XaiBase):
    pass


# ---------------------------------------------------------------------------
# NovitaLLM
# ---------------------------------------------------------------------------


_NovitaBase = _run_basic_openai_compat_tests(NovitaLLM, token_counting="per_chunk")


class TestNovitaLLM(_NovitaBase):
    pass


# ---------------------------------------------------------------------------
# WriterLLM
# ---------------------------------------------------------------------------


_WriterBase = _run_basic_openai_compat_tests(WriterLLM, token_counting="per_chunk")


class TestWriterLLM(_WriterBase):
    pass


# ---------------------------------------------------------------------------
# PerplexityLLM
# ---------------------------------------------------------------------------


_PerplexityBase = _run_basic_openai_compat_tests(PerplexityLLM, token_counting="per_chunk")


class TestPerplexityLLM(_PerplexityBase):
    pass


# ---------------------------------------------------------------------------
# CerebrasLLM
# ---------------------------------------------------------------------------


_CerebrasBase = _run_basic_openai_compat_tests(CerebrasLLM, token_counting="per_chunk")


class TestCerebrasLLM(_CerebrasBase):
    pass


# ---------------------------------------------------------------------------
# LMStudioLLM — api_key defaults to "lm-studio"
# ---------------------------------------------------------------------------


class TestLMStudioLLM:
    def _make(self):
        from synapsekit.llm.lmstudio import LMStudioLLM

        return LMStudioLLM(LLMConfig(model="test-model", api_key="", provider="lmstudio"))

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.lmstudio import LMStudioLLM

        assert inspect.isasyncgenfunction(LMStudioLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.lmstudio import LMStudioLLM

        assert inspect.isasyncgenfunction(LMStudioLLM.stream_with_messages)

    def test_api_key_defaults_to_lm_studio(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            llm._get_client()
        _, kw = mock_mod.AsyncOpenAI.call_args
        assert kw["api_key"] == "lm-studio"

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("LM"), _chunk("Studio"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["LM", "Studio"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("valid"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("local")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["local"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.2)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=77)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 77

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("lm"), _chunk("studio"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "lmstudio"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        usage_c = _usage_chunk(prompt_tokens=5, completion_tokens=8)
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), usage_c)
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 8

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "use tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"


# ---------------------------------------------------------------------------
# VLLMLLM — api_key defaults to "vllm", uses stream_options include_usage
# ---------------------------------------------------------------------------


class TestVLLMLLM:
    def _make(self):
        from synapsekit.llm.vllm import VLLMLLM

        return VLLMLLM(LLMConfig(model="test-model", api_key="", provider="vllm"))

    def test_missing_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError):
                llm._get_client()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.vllm import VLLMLLM

        assert inspect.isasyncgenfunction(VLLMLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.vllm import VLLMLLM

        assert inspect.isasyncgenfunction(VLLMLLM.stream_with_messages)

    def test_api_key_defaults_to_vllm(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            llm._get_client()
        _, kw = mock_mod.AsyncOpenAI.call_args
        assert kw["api_key"] == "vllm"

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("vLLM"), _chunk("fast"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["vLLM", "fast"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk(None), _chunk("real"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [t async for t in llm.stream("hi")]
        assert None not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream(_chunk("local")))
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            tokens = [
                t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])
            ]
        assert tokens == ["local"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", temperature=0.15)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["temperature"] == 0.15

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_async_stream())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi", max_tokens=333)]
        _, kw = mock_client.chat.completions.create.call_args
        assert kw["max_tokens"] == 333

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("vl"), _chunk("lm"))
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.generate("hi")
        assert result == "vllm"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_client = MagicMock()
        usage_c = _usage_chunk(prompt_tokens=12, completion_tokens=7)
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_stream(_chunk("a"), _chunk("b"), usage_c)
        )
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            [t async for t in llm.stream("hi")]
        assert llm._input_tokens == 12
        assert llm._output_tokens == 7

    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            llm = self._make()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_tool_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "use tool"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["tool_calls"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        llm = self._make()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_content_response())
        mock_mod = _mock_openai_module(mock_client)
        with patch.dict("sys.modules", {"openai": mock_mod}):
            result = await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "my_tool", "description": "", "parameters": {}}}],
            )
        assert result["content"] == "Just text"
