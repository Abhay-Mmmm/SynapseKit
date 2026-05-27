from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.anthropic import AnthropicLLM
from synapsekit.llm.base import LLMConfig


def make_config() -> LLMConfig:
    return LLMConfig(model="claude-3-5-sonnet-20241022", api_key="sk-test", provider="anthropic")


def make_llm() -> AnthropicLLM:
    return AnthropicLLM(make_config())


def _make_stream_context(tokens: list[str], input_tokens: int = 10, output_tokens: int = 5):
    async def _text_stream():
        for t in tokens:
            yield t

    mock_stream = MagicMock()
    mock_stream.text_stream = _text_stream()
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    final_msg = MagicMock()
    final_msg.usage = usage
    mock_stream.get_final_message = AsyncMock(return_value=final_msg)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


class TestAnthropicLLMMissingDep:
    def test_missing_anthropic_dep_raises(self):
        llm = make_llm()
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package required"):
                llm._get_client()


class TestAnthropicLLMAsyncGenFunctions:
    def test_stream_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(AnthropicLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(AnthropicLLM.stream_with_messages)


class TestAnthropicLLMStreamYields:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context(["Hello", " world"])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context(["Tok1", "Tok2"])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            messages = [{"role": "user", "content": "hello"}]
            tokens = [t async for t in llm.stream_with_messages(messages)]
        assert tokens == ["Tok1", "Tok2"]

    @pytest.mark.asyncio
    async def test_stream_captures_token_usage(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context(
            ["hi"], input_tokens=7, output_tokens=3
        )
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            [t async for t in llm.stream("hi")]
        assert llm._input_tokens == 7
        assert llm._output_tokens == 3

    @pytest.mark.asyncio
    async def test_stream_passes_temperature_kwarg(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context([])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            [t async for t in llm.stream("hi", temperature=0.7)]
        _, kw = mock_client.messages.stream.call_args
        assert kw["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens_kwarg(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context([])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            [t async for t in llm.stream("hi", max_tokens=256)]
        _, kw = mock_client.messages.stream.call_args
        assert kw["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_stream_with_messages_strips_system_from_messages(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context([])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            messages = [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
            ]
            [t async for t in llm.stream_with_messages(messages)]
        _, kw = mock_client.messages.stream.call_args
        user_messages = kw["messages"]
        roles = [m["role"] for m in user_messages]
        assert "system" not in roles
        assert kw["system"] == "Be concise."


class TestAnthropicLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_context(["Hello", " world"])
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = await llm.generate("hi")
        assert result == "Hello world"


class TestAnthropicLLMClientCaching:
    def test_client_is_cached(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            llm = make_llm()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2
        assert mock_anthropic.AsyncAnthropic.call_count == 1


class TestAnthropicLLMToolCalling:
    def _make_tool_response(self, has_tool: bool):
        if has_tool:
            tool_use = MagicMock()
            tool_use.type = "tool_use"
            tool_use.id = "call_1"
            tool_use.name = "get_weather"
            tool_use.input = {"city": "London"}
            content = [tool_use]
        else:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "The weather is fine."
            content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        response = MagicMock()
        response.content = content
        response.usage = usage
        return response

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_call(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=self._make_tool_response(has_tool=True)
        )
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = await llm.call_with_tools([{"role": "user", "content": "weather?"}], tools)
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"] == {"city": "London"}
        assert result["content"] is None

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content_when_no_tools(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=self._make_tool_response(has_tool=False)
        )
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {},
                }
            }
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = await llm.call_with_tools([{"role": "user", "content": "hi"}], tools)
        assert result["tool_calls"] is None
        assert result["content"] == "The weather is fine."

    @pytest.mark.asyncio
    async def test_call_with_tools_converts_schema(self):
        llm = make_llm()
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=self._make_tool_response(has_tool=False)
        )
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            }
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            await llm.call_with_tools([{"role": "user", "content": "search"}], tools)
        _, kw = mock_client.messages.create.call_args
        anthropic_tools = kw["tools"]
        assert len(anthropic_tools) == 1
        t = anthropic_tools[0]
        assert t["name"] == "search"
        assert t["description"] == "Search the web"
        assert "input_schema" in t
        assert t["input_schema"]["type"] == "object"


class TestAnthropicLLMPrepareMessages:
    def test_prepare_messages_handles_system_role(self):
        llm = make_llm()
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
        system, user_msgs = llm._prepare_messages(messages)
        assert system == "Be brief."
        assert all(m["role"] != "system" for m in user_msgs)

    def test_prepare_messages_handles_tool_role(self):
        llm = make_llm()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "4",
            },
        ]
        _, user_msgs = llm._prepare_messages(messages)
        tool_results = [
            m for m in user_msgs if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_results) == 1
        block = tool_results[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_abc"
        assert block["content"] == "4"
