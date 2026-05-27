from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def make_config(**kw) -> LLMConfig:
    return LLMConfig(model="test-model", api_key="", provider="local", **kw)


# ---------------------------------------------------------------------------
# LlamaCppLLM
# ---------------------------------------------------------------------------


class TestLlamaCppLLM:
    def _make(self, model_path: str = "/models/test.gguf") -> object:
        from synapsekit.llm.llamacpp import LlamaCppLLM

        return LlamaCppLLM(make_config(), model_path=model_path)

    def test_missing_llamacpp_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python required"):
                llm._get_model()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.llamacpp import LlamaCppLLM

        assert inspect.isasyncgenfunction(LlamaCppLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.llamacpp import LlamaCppLLM

        assert inspect.isasyncgenfunction(LlamaCppLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": ""}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            tokens = [t async for t in llm.stream("hi")]
        assert "" not in tokens
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self):
        llm = self._make()
        mock_model = MagicMock()
        captured: dict = {}

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            captured["temperature"] = temperature
            return iter([])

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            [t async for t in llm.stream("hi", temperature=0.88)]
        assert captured["temperature"] == 0.88

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self):
        llm = self._make()
        mock_model = MagicMock()
        captured: dict = {}

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            captured["max_tokens"] = max_tokens
            return iter([])

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            [t async for t in llm.stream("hi", max_tokens=256)]
        assert captured["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_completion(messages, stream, temperature, max_tokens, top_p):
            yield {"choices": [{"delta": {"content": "a"}}]}
            yield {"choices": [{"delta": {"content": "b"}}]}
            yield {"choices": [{"delta": {"content": "c"}}]}

        mock_model.create_chat_completion = mock_completion

        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3

    def test_model_is_cached(self):
        mock_model = MagicMock()
        mock_llamacpp = MagicMock()
        mock_llamacpp.Llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": mock_llamacpp}):
            llm = self._make()
            m1 = llm._get_model()
            m2 = llm._get_model()
        assert m1 is m2
        assert mock_llamacpp.Llama.call_count == 1

    @pytest.mark.asyncio
    async def test_call_with_tools_not_supported(self):
        llm = self._make()
        with pytest.raises(NotImplementedError):
            await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "tool", "description": "", "parameters": {}}}],
            )


# ---------------------------------------------------------------------------
# GPT4AllLLM
# ---------------------------------------------------------------------------


class TestGPT4AllLLM:
    def _make(self) -> object:
        from synapsekit.llm.gpt4all import GPT4AllLLM

        return GPT4AllLLM(make_config())

    def test_missing_gpt4all_dep_raises(self):
        llm = self._make()
        with patch.dict("sys.modules", {"gpt4all": None}):
            with pytest.raises(ImportError, match="gpt4all required"):
                llm._get_model()

    def test_stream_is_async_gen_function(self):
        from synapsekit.llm.gpt4all import GPT4AllLLM

        assert inspect.isasyncgenfunction(GPT4AllLLM.stream)

    def test_stream_with_messages_is_async_gen_function(self):
        from synapsekit.llm.gpt4all import GPT4AllLLM

        assert inspect.isasyncgenfunction(GPT4AllLLM.stream_with_messages)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_generate(prompt, max_tokens, temp, top_p, streaming, callback):
            callback(0, "Hello")
            callback(1, " world")
            callback(2, "")
            return None

        mock_model.generate = mock_generate

        mock_gpt4all = MagicMock()
        mock_gpt4all.GPT4All.return_value = mock_model
        with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
            tokens = [t async for t in llm.stream("hi")]
        assert "Hello" in tokens
        assert " world" in tokens

    @pytest.mark.asyncio
    async def test_stream_skips_empty_callback_response(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_generate(prompt, max_tokens, temp, top_p, streaming, callback):
            callback(0, "")
            callback(1, "real")
            return None

        mock_model.generate = mock_generate

        mock_gpt4all = MagicMock()
        mock_gpt4all.GPT4All.return_value = mock_model
        with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
            tokens = [t async for t in llm.stream("hi")]
        assert "" not in tokens
        assert "real" in tokens

    @pytest.mark.asyncio
    async def test_generate_joins_tokens(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_generate(prompt, max_tokens, temp, top_p, streaming, callback):
            callback(0, "Hello")
            callback(1, " world")
            return None

        mock_model.generate = mock_generate

        mock_gpt4all = MagicMock()
        mock_gpt4all.GPT4All.return_value = mock_model
        with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
            result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_token_counting(self):
        llm = self._make()
        mock_model = MagicMock()

        def mock_generate(prompt, max_tokens, temp, top_p, streaming, callback):
            callback(0, "a")
            callback(1, "b")
            callback(2, "c")
            callback(3, "d")
            return None

        mock_model.generate = mock_generate

        mock_gpt4all = MagicMock()
        mock_gpt4all.GPT4All.return_value = mock_model
        with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
            [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 4

    def test_model_is_cached(self):
        mock_model = MagicMock()
        mock_gpt4all = MagicMock()
        mock_gpt4all.GPT4All.return_value = mock_model
        with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
            llm = self._make()
            m1 = llm._get_model()
            m2 = llm._get_model()
        assert m1 is m2
        assert mock_gpt4all.GPT4All.call_count == 1

    @pytest.mark.asyncio
    async def test_call_with_tools_not_supported(self):
        llm = self._make()
        with pytest.raises(NotImplementedError):
            await llm.call_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"function": {"name": "tool", "description": "", "parameters": {}}}],
            )

    def test_flatten_messages(self):
        from synapsekit.llm.gpt4all import GPT4AllLLM

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = GPT4AllLLM._flatten_messages(messages)
        assert "System: You are helpful." in result
        assert "User: Hello!" in result
        assert "Assistant: Hi there!" in result
        lines = result.split("\n")
        assert len(lines) == 3
