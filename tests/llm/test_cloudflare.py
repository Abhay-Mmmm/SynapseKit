from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.cloudflare import CloudflareLLM


def make_config() -> LLMConfig:
    return LLMConfig(
        model="@cf/meta/llama-3.1-8b-instruct",
        api_key="cf-test-token",
        provider="cloudflare",
    )


def make_llm(account_id: str | None = "acc123") -> CloudflareLLM:
    return CloudflareLLM(make_config(), account_id=account_id)


def _mock_httpx_stream(lines: list[str]):
    async def mock_aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_cm
    return mock_client


class TestCloudflareDepAndInit:
    def test_missing_httpx_dep_raises(self):
        llm = make_llm()
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx required"):
                llm._get_client()

    def test_missing_account_id_raises(self):
        llm = make_llm(account_id=None)
        mock_httpx = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(ValueError, match="account_id"):
                llm._get_client()


class TestCloudflareAsyncGenFunctions:
    def test_stream_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(CloudflareLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        assert inspect.isasyncgenfunction(CloudflareLLM.stream_with_messages)


class TestCloudflareStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                'data: {"response": " world"}',
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_response(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                'data: {"response": ""}',
                'data: {"response": " world"}',
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            tokens = [t async for t in llm.stream("hi")]
        assert "" not in tokens
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                "data: [DONE]",
                'data: {"response": "should not appear"}',
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello"]
        assert "should not appear" not in tokens

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                ": keep-alive",
                'data: {"response": "Hello"}',
                "",
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello"]

    @pytest.mark.asyncio
    async def test_stream_skips_malformed_json(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                "data: {not valid json}",
                'data: {"response": "valid"}',
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["valid"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(["data: [DONE]"])
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            [t async for t in llm.stream("hi", temperature=0.6)]
        _, kw = mock_client.stream.call_args
        assert kw["json"]["temperature"] == 0.6

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(["data: [DONE]"])
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            [t async for t in llm.stream("hi", max_tokens=512)]
        _, kw = mock_client.stream.call_args
        assert kw["json"]["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                'data: {"response": " world"}',
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = make_llm()
        mock_client = _mock_httpx_stream(
            [
                'data: {"response": "a"}',
                'data: {"response": "b"}',
                'data: {"response": "c"}',
                "data: [DONE]",
            ]
        )
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3


class TestCloudflareClientCaching:
    def test_client_is_cached(self):
        mock_client = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = make_llm()
            c1 = llm._get_client()
            c2 = llm._get_client()
        assert c1 is c2
        assert mock_httpx.AsyncClient.call_count == 1


class TestCloudflareToolsAndEndpoint:
    @pytest.mark.asyncio
    async def test_call_with_tools_not_supported(self):
        llm = make_llm()
        with pytest.raises(NotImplementedError):
            await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "tool", "description": "", "parameters": {}}}],
            )

    def test_correct_endpoint_constructed(self):
        llm = CloudflareLLM(
            make_config(),
            account_id="my-account-123",
        )
        endpoint = llm._endpoint()
        assert "my-account-123" in endpoint
        assert "@cf/meta/llama-3.1-8b-instruct" in endpoint
