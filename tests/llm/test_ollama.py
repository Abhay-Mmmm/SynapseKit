from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.ollama import OllamaLLM


def make_config() -> LLMConfig:
    return LLMConfig(model="llama3", api_key="", provider="ollama")


def make_llm() -> OllamaLLM:
    return OllamaLLM(make_config())


def make_mock_chat(contents: list[str]):
    async def _chat(**kw):
        async def _gen():
            for c in contents:
                yield {"message": {"content": c}}

        return _gen()

    return _chat


class TestOllamaLLMMissingDep:
    def test_missing_ollama_dep_raises(self):
        llm = make_llm()
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError, match="ollama required"):
                llm._get_client()


class TestOllamaLLMAsyncGenFunctions:
    def test_stream_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(OllamaLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(OllamaLLM.stream_with_messages)


class TestOllamaLLMStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = make_llm()
        mock_client = MagicMock()
        mock_client.chat = make_mock_chat(["Hello", " world"])
        llm._client = mock_client

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self):
        llm = make_llm()
        mock_client = MagicMock()
        mock_client.chat = make_mock_chat(["Hello", "", " world"])
        llm._client = mock_client

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]
        assert "" not in tokens

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        llm = make_llm()
        mock_client = MagicMock()
        mock_client.chat = make_mock_chat(["Token1", "Token2"])
        llm._client = mock_client

        messages = [{"role": "user", "content": "hello"}]
        tokens = [t async for t in llm.stream_with_messages(messages)]
        assert tokens == ["Token1", "Token2"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature_kwarg(self):
        llm = make_llm()
        mock_client = MagicMock()
        captured_kw: dict = {}

        async def _chat(**kw):
            captured_kw.update(kw)

            async def _gen():
                return
                yield

            return _gen()

        mock_client.chat = _chat
        llm._client = mock_client

        [t async for t in llm.stream("hi", temperature=0.9)]
        assert captured_kw["options"]["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens_kwarg(self):
        llm = make_llm()
        mock_client = MagicMock()
        captured_kw: dict = {}

        async def _chat(**kw):
            captured_kw.update(kw)

            async def _gen():
                return
                yield

            return _gen()

        mock_client.chat = _chat
        llm._client = mock_client

        [t async for t in llm.stream("hi", max_tokens=512)]
        assert captured_kw["options"]["num_predict"] == 512

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_messages(self):
        config = make_config()
        config.system_prompt = "You are a pirate."
        llm = OllamaLLM(config)
        mock_client = MagicMock()
        captured_kw: dict = {}

        async def _chat(**kw):
            captured_kw.update(kw)

            async def _gen():
                return
                yield

            return _gen()

        mock_client.chat = _chat
        llm._client = mock_client

        [t async for t in llm.stream("hello")]
        msgs = captured_kw["messages"]
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a pirate."


class TestOllamaLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = make_llm()
        mock_client = MagicMock()
        mock_client.chat = make_mock_chat(["Hello", " world"])
        llm._client = mock_client

        result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = make_llm()
        mock_client = MagicMock()
        mock_client.chat = make_mock_chat(["a", "b", "c"])
        llm._client = mock_client

        [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3


class TestOllamaLLMClientCaching:
    def test_client_is_cached(self):
        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.return_value = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            llm = make_llm()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2
        assert mock_ollama.AsyncClient.call_count == 1


class TestOllamaLLMRegression:
    @pytest.mark.asyncio
    async def test_regression_chat_called_with_await(self):
        """
        Regression: client.chat must be awaited (not iterated directly).
        The ollama AsyncClient.chat is a coroutine that returns an async generator.
        Without `await`, iterating would fail because the coroutine object is not
        an async iterable.
        """
        llm = make_llm()
        mock_client = MagicMock()

        async def mock_chat(**kw):
            async def _gen():
                yield {"message": {"content": "Hello"}}
                yield {"message": {"content": " world"}}

            return _gen()

        mock_client.chat = mock_chat
        llm._client = mock_client

        tokens = [t async for t in llm.stream("test")]
        assert tokens == ["Hello", " world"]
