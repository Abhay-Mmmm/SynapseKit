"""Audit sinks — persistence backends that receive already-signed batches.

Sinks never see a signing key and never decide when something is signed;
:mod:`synapsekit.audit.signer` produces a :class:`Signature` over a batch
*before* any sink is invoked. A sink's only job is durable storage of
bytes it's handed.

Every sink exposes both a blocking :meth:`~AuditSink.write` and an
:meth:`~AuditSink.awrite` coroutine. The base class implements
``awrite`` by running ``write`` in a worker thread
(``asyncio.to_thread``) so none of the blocking disk/network I/O below
(file ``fsync``, S3/Kafka network calls) ever stalls the event loop when
called from SynapseKit's async-native code paths — override ``awrite``
directly in a subclass if a truly async client (e.g. ``aioboto3``) is
available.
"""

from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import AuditRecord, Signature


@dataclass
class SignedBatch:
    """A batch of records plus the signature already computed over its Merkle root."""

    records: list[AuditRecord]
    signature: Signature
    leaves: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "signature": self.signature.to_dict(),
            "leaves": self.leaves,
        }


class AuditSink(ABC):
    """A durable store for signed audit batches."""

    @abstractmethod
    def write(self, batch: SignedBatch) -> None: ...

    async def awrite(self, batch: SignedBatch) -> None:
        """Async-safe write — off-loads the (potentially blocking) :meth:`write` to a thread."""
        await asyncio.to_thread(self.write, batch)


class FileAuditSink(AuditSink):
    """Append-only newline-delimited JSON file. Never truncates or rewrites."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, batch: SignedBatch) -> None:
        import os

        line = json.dumps(batch.to_dict(), sort_keys=True)
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


class S3AuditSink(AuditSink):
    """Writes one object per batch to S3 (or an S3-compatible store).

    The ``boto3`` client is constructed lazily on first use (not in
    ``__init__``), so simply instantiating this sink — e.g. while
    validating configuration — never attempts a network/credential
    round-trip.
    """

    def __init__(
        self, bucket: str, *, prefix: str = "synapsekit-audit", client: Any = None
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def write(self, batch: SignedBatch) -> None:
        client = self._ensure_client()
        key = f"{self._prefix}/{batch.signature.start_index:012d}-{batch.signature.end_index:012d}.json"
        body = json.dumps(batch.to_dict(), sort_keys=True).encode("utf-8")
        client.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType="application/json")


class KafkaAuditSink(AuditSink):
    """Publishes one message per batch to a Kafka topic.

    The ``kafka-python`` producer is constructed lazily on first use
    (not in ``__init__``), so simply instantiating this sink never
    attempts a broker connection.
    """

    def __init__(
        self, topic: str, *, producer: Any = None, bootstrap_servers: str | None = None
    ) -> None:
        self._topic = topic
        self._producer = producer
        self._bootstrap_servers = bootstrap_servers

    def _ensure_producer(self) -> Any:
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers or "localhost:9092",
                value_serializer=lambda v: json.dumps(v, sort_keys=True).encode("utf-8"),
            )
        return self._producer

    def write(self, batch: SignedBatch) -> None:
        producer = self._ensure_producer()
        future = producer.send(self._topic, batch.to_dict())
        # Surface send-time errors immediately rather than dropping them —
        # a swallowed publish failure would silently break the audit trail.
        if hasattr(future, "get"):
            future.get(timeout=30)


class OTelAuditSink(AuditSink):
    """Forwards each record in a batch to an OTel exporter as a span/event."""

    def __init__(self, exporter: Any) -> None:
        self._exporter = exporter

    def write(self, batch: SignedBatch) -> None:
        from .otel import export_records

        export_records(self._exporter, batch.records)


class MultiSink(AuditSink):
    """Fans a batch out to several sinks; raises if any of them fails."""

    def __init__(self, sinks: list[AuditSink]) -> None:
        self._sinks = sinks

    def write(self, batch: SignedBatch) -> None:
        errors = []
        for sink in self._sinks:
            try:
                sink.write(batch)
            except Exception as exc:
                errors.append(f"{type(sink).__name__}: {exc}")
        if errors:
            raise RuntimeError(f"{len(errors)} sink(s) failed to write batch: {'; '.join(errors)}")

    async def awrite(self, batch: SignedBatch) -> None:
        # Concurrent fan-out rather than the base class's single
        # to_thread(write) — each sink's own I/O overlaps instead of
        # being serialized behind one one worker thread.
        results = await asyncio.gather(
            *(sink.awrite(batch) for sink in self._sinks), return_exceptions=True
        )
        errors = [
            f"{type(sink).__name__}: {result}"
            for sink, result in zip(self._sinks, results, strict=True)
            if isinstance(result, Exception)
        ]
        if errors:
            raise RuntimeError(f"{len(errors)} sink(s) failed to write batch: {'; '.join(errors)}")
