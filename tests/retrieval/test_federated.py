"""Tests for FederatedRetriever."""

from __future__ import annotations

import asyncio
import json

import pytest

from synapsekit.retrieval.federated import FederatedRetriever


class _MockRetriever:
    def __init__(self, results):
        self._results = results

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        return self._results[:top_k]

    async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
        return [
            {"text": t, "score": s, "metadata": {"id": i}}
            for i, (t, s) in enumerate(self._results[:top_k])
        ]


class _DocumentOnlyRetriever:
    def __init__(self, docs):
        self._docs = docs

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        return self._docs[:top_k]


@pytest.mark.asyncio
async def test_federated_rrf_fusion_dedup():
    a = _MockRetriever([("doc a", 0.9), ("doc b", 0.8), ("doc c", 0.7)])
    b = _MockRetriever([("doc a", 0.95), ("doc d", 0.6)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "a", "retriever": a},
            {"name": "b", "retriever": b},
        ],
        fusion="rrf",
        top_k=3,
    )

    results = await retriever.retrieve_with_scores("query")
    texts = [r["text"] for r in results]
    assert texts[0] == "doc a"
    assert len(texts) == 3
    assert len(set(texts)) == 3


@pytest.mark.asyncio
async def test_federated_interleave():
    a = _MockRetriever([("doc a", 0.9), ("doc b", 0.8)])
    b = _MockRetriever([("doc c", 0.7), ("doc d", 0.6)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "a", "retriever": a},
            {"name": "b", "retriever": b},
        ],
        fusion="interleave",
        top_k=3,
    )

    results = await retriever.retrieve("query")
    assert results == ["doc a", "doc c", "doc b"]


@pytest.mark.asyncio
async def test_federated_score_fusion():
    a = _MockRetriever([("shared", 10.0), ("local a", 5.0)])
    b = _MockRetriever([("shared", 0.8), ("local b", 0.2)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "a", "retriever": a},
            {"name": "b", "retriever": b},
        ],
        fusion="score",
        top_k=3,
    )

    results = await retriever.retrieve("query")
    assert results == ["shared", "local a", "local b"]


@pytest.mark.asyncio
async def test_federated_accepts_document_only_retrievers():
    from synapsekit.loaders.base import Document

    docs = [
        Document(text="hr policy", metadata={"dept": "hr"}),
        Document(text="eng policy", metadata={"dept": "eng"}),
    ]
    retriever = FederatedRetriever(
        sources=[{"name": "docs", "retriever": _DocumentOnlyRetriever(docs)}],
        fusion="rrf",
        top_k=2,
    )

    results = await retriever.retrieve_with_scores("query")
    assert [r["text"] for r in results] == ["hr policy", "eng policy"]
    assert results[0]["metadata"]["dept"] == "hr"


@pytest.mark.asyncio
async def test_federated_accepts_tuple_results():
    class _TupleRetriever:
        async def retrieve(self, query, top_k=5, metadata_filter=None):
            return [("tuple doc", 0.42), ("other", 0.1)]

    retriever = FederatedRetriever(
        sources=[{"name": "tuple", "retriever": _TupleRetriever()}],
        fusion="rrf",
        top_k=1,
    )

    results = await retriever.retrieve_with_scores("query")
    assert results == [{"text": "tuple doc", "score": 0.42, "metadata": {}, "source": "tuple"}]


@pytest.mark.asyncio
async def test_federated_timeout_returns_partial():
    class _SlowRetriever:
        async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
            await asyncio.sleep(0.05)
            return [{"text": "late", "score": 0.1, "metadata": {}}]

    fast = _MockRetriever([("doc a", 0.9)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "fast", "retriever": fast},
            {"name": "slow", "retriever": _SlowRetriever(), "timeout_ms": 1},
        ],
        fusion="rrf",
        top_k=2,
    )

    results = await retriever.retrieve("query")
    assert results == ["doc a"]


@pytest.mark.asyncio
async def test_federated_remote_source_standard_protocol(monkeypatch):
    import socket

    httpx = pytest.importorskip("httpx")

    # The SSRF guard (fail-closed) resolves the host before the request; stub
    # DNS so the placeholder host validates as a public address.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))
        ],
    )

    captured: dict[str, object] = {}

    def handler(request):
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "text": "remote policy",
                        "score": 0.85,
                        "metadata": {"dept": "hr"},
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        return original_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    retriever = FederatedRetriever(
        sources=[
            {
                "name": "remote",
                "url": "https://federated.invalid/retrieve",
                "api_key": "sekret",
            }
        ],
        fusion="score",
        top_k=1,
    )

    results = await retriever.retrieve_with_scores(
        "vacation policy",
        metadata_filter={"department": "hr"},
    )

    assert captured["authorization"] == "Bearer sekret"
    assert captured["body"] == {
        "query": "vacation policy",
        "top_k": 1,
        "metadata_filter": {"department": "hr"},
    }
    assert results == [
        {
            "text": "remote policy",
            "score": 0.85,
            "metadata": {"dept": "hr"},
            "source": "remote",
        }
    ]
