"""Audit sinks persist already-signed batches; they never touch signing."""

from __future__ import annotations

import json

import pytest

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy
from synapsekit.audit.merkle import MerkleHasher
from synapsekit.audit.sink import FileAuditSink, KafkaAuditSink, MultiSink, S3AuditSink, SignedBatch


def _make_batch(records):
    policy = SigningPolicy.ed25519()
    leaves = [r.hash for r in records]
    root = MerkleHasher.root(leaves)
    signature = policy.sign_batch(root, start_index=0, end_index=len(records) - 1)
    return SignedBatch(records=records, signature=signature, leaves=leaves)


class TestFileAuditSink:
    def test_appends_one_line_per_batch(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        sink = FileAuditSink(path)

        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        batch1 = _make_batch(tracer.drain())
        sink.write(batch1)

        tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        batch2 = _make_batch(tracer.drain())
        sink.write(batch2)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["signature"]["merkle_root"] == batch1.signature.merkle_root

    def test_sink_receives_a_signature_it_never_computed_itself(self, tmp_path):
        # The sink has no signing key and no SigningPolicy reference at all —
        # it only ever sees the already-computed Signature dataclass.
        path = tmp_path / "audit.jsonl"
        sink = FileAuditSink(path)
        assert not hasattr(sink, "provider")
        assert not hasattr(sink, "sign")


class TestMultiSink:
    def test_fans_out_to_all_sinks(self, tmp_path):
        path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        multi = MultiSink([FileAuditSink(path_a), FileAuditSink(path_b)])
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        multi.write(_make_batch(tracer.drain()))
        assert path_a.exists() and path_b.exists()

    def test_raises_if_any_sink_fails(self, tmp_path):
        class BrokenSink:
            def write(self, batch):
                raise RuntimeError("disk full")

        multi = MultiSink([FileAuditSink(tmp_path / "ok.jsonl"), BrokenSink()])
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        try:
            multi.write(_make_batch(tracer.drain()))
            raised = False
        except RuntimeError:
            raised = True
        assert raised


class TestAsyncSinkSupport:
    """Sinks must not block the event loop — see the module docstring."""

    @pytest.mark.asyncio
    async def test_file_sink_awrite_works(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        sink = FileAuditSink(path)
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        await sink.awrite(_make_batch(tracer.drain()))
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    @pytest.mark.asyncio
    async def test_multi_sink_awrite_fans_out_concurrently(self, tmp_path):
        path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        multi = MultiSink([FileAuditSink(path_a), FileAuditSink(path_b)])
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        await multi.awrite(_make_batch(tracer.drain()))
        assert path_a.exists() and path_b.exists()

    @pytest.mark.asyncio
    async def test_multi_sink_awrite_raises_if_any_sink_fails(self, tmp_path):
        class BrokenAsyncSink:
            async def awrite(self, batch):
                raise RuntimeError("disk full")

        multi = MultiSink([FileAuditSink(tmp_path / "ok.jsonl"), BrokenAsyncSink()])
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        with pytest.raises(RuntimeError):
            await multi.awrite(_make_batch(tracer.drain()))


class TestLazyClientConstruction:
    """Instantiating a network-backed sink must never itself open a connection."""

    def test_s3_sink_does_not_construct_a_client_at_init(self):
        sink = S3AuditSink("my-bucket")
        assert sink._client is None

    def test_kafka_sink_does_not_construct_a_producer_at_init(self):
        sink = KafkaAuditSink("my-topic")
        assert sink._producer is None
