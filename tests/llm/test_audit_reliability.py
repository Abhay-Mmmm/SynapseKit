"""Regression tests for the LLM reliability/correctness audit.

Covers issues #773 #774 #775 #776 #777 #778 #779. Each test group fails on the
pre-fix code and passes on the fixed code.

Testing standards: pytest-only, no MagicMock/Mock/patch. Real SQLite/tmp_path,
hand-written fakes, and (where used) real httpx exception objects.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from synapsekit.llm._retry import _is_retryable, retry_async
from synapsekit.llm._sqlite_cache import SQLiteLLMCache
from synapsekit.llm.base import BaseLLM, LLMConfig

# ------------------------------------------------------------------ #
# Shared fakes
# ------------------------------------------------------------------ #


class _ScriptedLLM(BaseLLM):
    """LLM whose stream yields a fixed response and counts calls."""

    def __init__(self, config: LLMConfig, response: str = "hello") -> None:
        super().__init__(config)
        self.response = response
        self.calls = 0

    async def stream(self, prompt: str, **kw: Any):
        self.calls += 1
        yield self.response


# ================================================================== #
# #779: LLMConfig validation
# ================================================================== #


class TestConfigValidation:
    def test_temperature_too_high_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(model="m", api_key="k", provider="p", temperature=2.5)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(model="m", api_key="k", provider="p", temperature=-0.1)

    def test_temperature_boundaries_ok(self):
        LLMConfig(model="m", api_key="k", provider="p", temperature=0.0)
        LLMConfig(model="m", api_key="k", provider="p", temperature=2.0)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(model="m", api_key="k", provider="p", max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(model="m", api_key="k", provider="p", max_tokens=-5)

    def test_top_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            LLMConfig(model="m", api_key="k", provider="p", top_p=1.5)

    def test_top_p_none_ok(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p")
        assert cfg.top_p is None

    def test_top_p_boundaries_ok(self):
        LLMConfig(model="m", api_key="k", provider="p", top_p=0.0)
        LLMConfig(model="m", api_key="k", provider="p", top_p=1.0)

    def test_empty_api_key_allowed_for_local_providers(self):
        # Local providers need no api_key — must not raise.
        cfg = LLMConfig(model="m", api_key="", provider="ollama")
        assert cfg.api_key == ""

    def test_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            LLMConfig(model="m", api_key="k", provider="p", timeout=-1.0)


# ================================================================== #
# #775: timeout field + max_retries default of 2
# ================================================================== #


class TestConfigDefaults:
    def test_max_retries_defaults_to_two(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p")
        assert cfg.max_retries == 2

    def test_timeout_field_present_and_stored(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p", timeout=12.5)
        assert cfg.timeout == 12.5

    def test_timeout_default_none(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p")
        assert cfg.timeout is None


# ================================================================== #
# #774: cache key isolates by system_prompt and provider
# ================================================================== #


class TestCacheKeyIsolation:
    async def test_different_system_prompts_do_not_collide(self):
        from synapsekit.llm._cache import AsyncLRUCache

        shared = AsyncLRUCache(maxsize=128)

        cfg_a = LLMConfig(
            model="m", api_key="k", provider="p", system_prompt="You are Alice.", cache=True
        )
        cfg_b = LLMConfig(
            model="m", api_key="k", provider="p", system_prompt="You are Bob.", cache=True
        )
        alice = _ScriptedLLM(cfg_a, response="alice-answer")
        bob = _ScriptedLLM(cfg_b, response="bob-answer")
        # Force them to share one cache backend (the bug scenario).
        alice._cache = shared
        bob._cache = shared

        r1 = await alice.generate("same prompt")
        r2 = await bob.generate("same prompt")
        assert r1 == "alice-answer"
        assert r2 == "bob-answer"  # must NOT return alice's cached answer

    async def test_different_providers_do_not_collide(self):
        from synapsekit.llm._cache import AsyncLRUCache

        shared = AsyncLRUCache(maxsize=128)
        cfg_a = LLMConfig(model="m", api_key="k", provider="openai", cache=True)
        cfg_b = LLMConfig(model="m", api_key="k", provider="anthropic", cache=True)
        a = _ScriptedLLM(cfg_a, response="openai-answer")
        b = _ScriptedLLM(cfg_b, response="anthropic-answer")
        a._cache = shared
        b._cache = shared

        assert await a.generate("hi") == "openai-answer"
        assert await b.generate("hi") == "anthropic-answer"

    async def test_same_config_still_caches(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p", cache=True)
        llm = _ScriptedLLM(cfg, response="cached")
        r1 = await llm.generate("hi")
        r2 = await llm.generate("hi")
        assert r1 == r2 == "cached"
        assert llm.calls == 1

    async def test_messages_path_isolates_by_system_prompt(self):
        from synapsekit.llm._cache import AsyncLRUCache

        shared = AsyncLRUCache(maxsize=128)
        cfg_a = LLMConfig(
            model="m", api_key="k", provider="p", system_prompt="A", cache=True
        )
        cfg_b = LLMConfig(
            model="m", api_key="k", provider="p", system_prompt="B", cache=True
        )
        a = _ScriptedLLM(cfg_a, response="from-A")
        b = _ScriptedLLM(cfg_b, response="from-B")
        a._cache = shared
        b._cache = shared

        msgs = [{"role": "user", "content": "hi"}]
        assert await a.generate_with_messages(msgs) == "from-A"
        assert await b.generate_with_messages(msgs) == "from-B"


# ================================================================== #
# #776: retry classification by type / HTTP status + jitter + Retry-After
# ================================================================== #


class TestRetryClassification:
    async def test_retries_timeout(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise TimeoutError("slow")
            return "ok"

        assert await retry_async(fn, max_retries=3, delay=0.001) == "ok"
        assert calls == 2

    async def test_retries_connection_error(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("reset")
            return "ok"

        assert await retry_async(fn, max_retries=3, delay=0.001) == "ok"
        assert calls == 3

    async def test_does_not_retry_value_error(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise ValueError("bad request payload")

        with pytest.raises(ValueError):
            await retry_async(fn, max_retries=3, delay=0.001)
        assert calls == 1  # 4xx-style logic errors must not be retried

    async def test_does_not_retry_runtime_error(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise RuntimeError("some non-transient failure")

        with pytest.raises(RuntimeError):
            await retry_async(fn, max_retries=3, delay=0.001)
        assert calls == 1

    async def test_retries_httpx_429(self):
        import httpx

        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                req = httpx.Request("GET", "https://example.com")
                resp = httpx.Response(429, request=req)
                raise httpx.HTTPStatusError("rate limited", request=req, response=resp)
            return "ok"

        assert await retry_async(fn, max_retries=3, delay=0.001) == "ok"
        assert calls == 2

    async def test_does_not_retry_httpx_400(self):
        import httpx

        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            req = httpx.Request("GET", "https://example.com")
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad request", request=req, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            await retry_async(fn, max_retries=3, delay=0.001)
        assert calls == 1

    async def test_does_not_retry_httpx_401(self):
        import httpx

        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            req = httpx.Request("GET", "https://example.com")
            resp = httpx.Response(401, request=req)
            raise httpx.HTTPStatusError("unauthorized", request=req, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            await retry_async(fn, max_retries=3, delay=0.001)
        assert calls == 1

    async def test_retries_httpx_503(self):
        import httpx

        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                req = httpx.Request("GET", "https://example.com")
                resp = httpx.Response(503, request=req)
                raise httpx.HTTPStatusError("unavailable", request=req, response=resp)
            return "ok"

        assert await retry_async(fn, max_retries=3, delay=0.001) == "ok"
        assert calls == 2

    def test_is_retryable_classification(self):
        assert _is_retryable(TimeoutError()) is True
        assert _is_retryable(ConnectionError()) is True
        assert _is_retryable(ValueError()) is False
        assert _is_retryable(RuntimeError()) is False

    async def test_respects_retry_after_header(self):
        import httpx

        calls = 0
        slept: list[float] = []
        real_sleep = asyncio.sleep

        async def spy_sleep(seconds: float) -> None:
            slept.append(seconds)
            await real_sleep(0)  # do not actually wait

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                req = httpx.Request("GET", "https://example.com")
                resp = httpx.Response(429, headers={"Retry-After": "7"}, request=req)
                raise httpx.HTTPStatusError("rate limited", request=req, response=resp)
            return "ok"

        orig = asyncio.sleep
        asyncio.sleep = spy_sleep  # type: ignore[assignment]
        try:
            assert await retry_async(fn, max_retries=3, delay=0.001) == "ok"
        finally:
            asyncio.sleep = orig  # type: ignore[assignment]
        assert slept == [7.0]  # honored the header instead of computed backoff

    async def test_backoff_has_jitter(self):
        # Full jitter → sleep values must vary and stay within [0, base].
        slept: list[float] = []
        real_sleep = asyncio.sleep

        async def spy_sleep(seconds: float) -> None:
            slept.append(seconds)
            await real_sleep(0)

        async def fn():
            raise ConnectionError("always")

        orig = asyncio.sleep
        asyncio.sleep = spy_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ConnectionError):
                await retry_async(fn, max_retries=5, delay=1.0)
        finally:
            asyncio.sleep = orig  # type: ignore[assignment]

        # attempt i backoff base = 1.0 * 2**i; jittered sleep in [0, base].
        assert len(slept) == 5
        for i, s in enumerate(slept):
            assert 0.0 <= s <= 1.0 * (2**i) + 1e-9
        # Extremely unlikely all identical if jitter is applied.
        assert len(set(slept)) > 1


# ================================================================== #
# #778: SQLite cache thread-safety / concurrency
# ================================================================== #


class TestSQLiteCacheConcurrency:
    def test_wal_mode_enabled(self, tmp_path):
        cache = SQLiteLLMCache(db_path=str(tmp_path / "c.db"))
        try:
            mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            cache.close()

    def test_check_same_thread_disabled(self, tmp_path):
        # Using the connection from another thread must not raise.
        cache = SQLiteLLMCache(db_path=str(tmp_path / "c.db"))
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                cache.put(f"k{n}", f"v{n}")
                assert cache.get(f"k{n}") == f"v{n}"
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        import threading

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        try:
            assert errors == []
            assert len(cache) == 8
        finally:
            cache.close()

    async def test_concurrent_async_writers_no_lock_error(self, tmp_path):
        cache = SQLiteLLMCache(db_path=str(tmp_path / "c.db"))

        async def writer(n: int) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cache.put, f"k{n}", f"v{n}")

        try:
            await asyncio.gather(*(writer(i) for i in range(20)))
            assert len(cache) == 20
        finally:
            cache.close()


# ================================================================== #
# #777: async cache path offloads blocking I/O to an executor
# ================================================================== #


class _BlockingCache:
    """A cache whose get/put block the calling thread (simulates sync redis).

    Records the thread it runs on so the test can assert the async path did not
    execute it on the event loop thread.
    """

    def __init__(self) -> None:
        import threading

        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.get_threads: list[int] = []
        self.put_threads: list[int] = []
        self._main_thread = threading.get_ident()

    def get(self, key: str) -> Any | None:
        import threading
        import time

        self.get_threads.append(threading.get_ident())
        time.sleep(0.02)  # simulate blocking socket I/O
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        import threading
        import time

        self.put_threads.append(threading.get_ident())
        time.sleep(0.02)
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)


class TestAsyncCacheOffload:
    async def test_blocking_cache_runs_off_event_loop_thread(self):
        import threading

        cfg = LLMConfig(model="m", api_key="k", provider="p", cache=True)
        llm = _ScriptedLLM(cfg, response="answer")
        blocking = _BlockingCache()
        llm._cache = blocking

        loop_thread = threading.get_ident()
        result = await llm.generate("hi")
        assert result == "answer"
        # get (miss) then put both happened off the event-loop thread.
        assert blocking.get_threads and all(t != loop_thread for t in blocking.get_threads)
        assert blocking.put_threads and all(t != loop_thread for t in blocking.put_threads)

    async def test_event_loop_stays_responsive_during_cache_io(self):
        cfg = LLMConfig(model="m", api_key="k", provider="p", cache=True)
        llm = _ScriptedLLM(cfg, response="answer")
        llm._cache = _BlockingCache()

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.005)
                ticks += 1

        # If cache I/O blocked the loop, the ticker could not advance concurrently.
        await asyncio.gather(llm.generate("hi"), ticker())
        assert ticks == 5


# ================================================================== #
# #777 support: RedisLLMCache constructs client with a socket timeout
# ================================================================== #


class _FakeRedisModule:
    """Hand-written stand-in for the ``redis`` package.

    Captures the kwargs passed to ``Redis.from_url`` so the test can assert the
    socket timeout is wired without a live Redis server or redis-py internals.
    """

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        module = self

        class _Redis:
            @staticmethod
            def from_url(url: str, **kwargs: Any) -> Any:
                module.captured = {"url": url, **kwargs}
                return object()

        self.Redis = _Redis


class TestRedisSocketTimeout:
    def test_redis_client_built_with_timeout(self, monkeypatch):
        import sys

        fake = _FakeRedisModule()
        # Inject a controlled fake so we assert exactly what RedisLLMCache passes,
        # independent of whether the real redis-py is installed.
        monkeypatch.setitem(sys.modules, "redis", fake)

        from synapsekit.llm._redis_cache import RedisLLMCache

        RedisLLMCache(url="redis://localhost:6379", socket_timeout=3.0)
        assert fake.captured["socket_timeout"] == 3.0
        assert fake.captured["socket_connect_timeout"] == 3.0
        assert fake.captured["url"] == "redis://localhost:6379"


# ================================================================== #
# #775: config.timeout is plumbed into providers that build httpx clients
# ================================================================== #


class TestTimeoutPlumbing:
    def test_cloudflare_uses_config_timeout(self):
        pytest.importorskip("httpx")
        from synapsekit.llm.cloudflare import CloudflareLLM

        cfg = LLMConfig(
            model="@cf/meta/llama-3.1-8b-instruct",
            api_key="k",
            provider="cloudflare",
            timeout=7.0,
        )
        llm = CloudflareLLM(cfg, account_id="acc")
        client = llm._get_client()
        try:
            assert client.timeout.read == 7.0
        finally:
            # Close the underlying transport to avoid unclosed-client warnings.
            asyncio.run(client.aclose())

    def test_cloudflare_defaults_when_no_timeout(self):
        pytest.importorskip("httpx")
        from synapsekit.llm.cloudflare import CloudflareLLM

        cfg = LLMConfig(
            model="@cf/meta/llama-3.1-8b-instruct",
            api_key="k",
            provider="cloudflare",
        )
        llm = CloudflareLLM(cfg, account_id="acc")
        client = llm._get_client()
        try:
            assert client.timeout.read == 120.0
        finally:
            asyncio.run(client.aclose())


# ================================================================== #
# #773: HuggingFace must not clobber temperature=0.0 via truthiness
# ================================================================== #


class _CapturingHFClient:
    """Hand-written stand-in for AsyncInferenceClient that records params."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs.get("model")
        self.token = kwargs.get("token")
        self.last_params: dict[str, Any] = {}

    async def text_generation(self, prompt: str, **params: Any) -> str:
        self.last_params = params
        return "captured"


@pytest.fixture
def hf_module_with_fake_client(monkeypatch):
    """Inject the capturing client into the huggingface module.

    Works whether or not the real huggingface-hub is installed, since we swap the
    module-level symbol and availability flag directly (no MagicMock).
    """
    from synapsekit.llm import huggingface as hf

    monkeypatch.setattr(hf, "AsyncInferenceClient", _CapturingHFClient, raising=False)
    monkeypatch.setattr(hf, "HUGGINGFACE_AVAILABLE", True, raising=False)
    return hf


class TestHuggingFaceTemperatureZero:
    async def test_temperature_zero_not_overridden(self, hf_module_with_fake_client):
        hf = hf_module_with_fake_client
        cfg = LLMConfig(
            provider="huggingface",
            model="test-model",
            api_key="k",
            temperature=0.0,
            max_tokens=32,
        )
        llm = hf.HuggingFaceLLM(cfg)
        result = await llm._generate("hi")
        assert result == "captured"
        # Pre-fix `temperature or 0.7` would have turned 0.0 into 0.7.
        assert llm.client.last_params["temperature"] == 0.0
        assert llm.client.last_params["max_new_tokens"] == 32

    async def test_stream_temperature_zero_not_overridden(self, hf_module_with_fake_client):
        hf = hf_module_with_fake_client

        captured: dict[str, Any] = {}

        class _StreamingClient(_CapturingHFClient):
            async def text_generation(self, prompt: str, **params: Any):
                captured.update(params)

                async def _gen():
                    for tok in ["a", "b"]:
                        yield tok

                return _gen()

        cfg = LLMConfig(
            provider="huggingface",
            model="test-model",
            api_key="k",
            temperature=0.0,
            max_tokens=16,
        )
        llm = hf.HuggingFaceLLM(cfg)
        llm.client = _StreamingClient(model="test-model", token="k")
        chunks = [c async for c in llm.stream("hi")]
        assert chunks == ["a", "b"]
        assert captured["temperature"] == 0.0
        assert captured["max_new_tokens"] == 16
