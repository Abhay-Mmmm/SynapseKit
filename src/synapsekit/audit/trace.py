"""Hash-chained trace recorder — the append-only spine of the audit system."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .serializer import hash_value
from .types import (
    GENESIS_HASH,
    SCHEMA_VERSION,
    AuditRecord,
    EventKind,
    RedactionStatus,
    deep_freeze,
)

if TYPE_CHECKING:
    from .redact import PIIRedactor


def _new_id() -> str:
    return uuid.uuid4().hex


def compute_payload_hash(payload: Any) -> str:
    """Hash a record's (already redacted, already frozen) payload on its own.

    Keeping this separate from :func:`compute_record_hash` — which
    commits to ``payload_hash`` rather than the raw payload directly —
    means a future selective-disclosure mode can drop a record's payload
    entirely while keeping ``payload_hash`` as evidence of what it
    originally committed to, without changing the hash-chain format.
    """
    return hash_value(payload)


def compute_record_hash(
    *,
    event_id: str,
    parent_id: str | None,
    run_id: str,
    kind: str,
    actor: str,
    payload_hash: str,
    timestamp: datetime,
    prev_hash: str,
    schema_version: str = SCHEMA_VERSION,
    redaction_policy_hash: str | None = None,
    redaction_status: RedactionStatus = "none",
) -> str:
    """Hash everything about a record except its own ``hash`` field.

    Commits to ``payload_hash`` rather than the raw ``payload`` — see
    :func:`compute_payload_hash`. This is the single source of truth for
    "what a record hash means" — :mod:`trace`, :mod:`bundle`, and
    :mod:`verifier` all call equivalent logic so a record produced by
    SynapseKit and one recomputed by an external verifier are
    byte-for-byte comparable.
    """
    return hash_value(
        {
            "event_id": event_id,
            "parent_id": parent_id,
            "run_id": run_id,
            "kind": kind,
            "actor": actor,
            "payload_hash": payload_hash,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
            "schema_version": schema_version,
            "redaction_policy_hash": redaction_policy_hash,
            "redaction_status": redaction_status,
        }
    )


class ChainIntegrityError(Exception):
    """Raised when a record sequence fails hash-chain verification."""


class AuditTracer:
    """Append-only, hash-chained recorder for one run.

    Every :meth:`record` call links to the previous record's hash, so the
    chain behaves like a blockchain: retroactively editing any record
    changes its hash and invalidates every record after it. Thread-safe;
    an :class:`AuditTracer` is meant to be flushed periodically (see
    :mod:`synapsekit.audit.sink`) rather than kept forever in memory.
    """

    def __init__(self, run_id: str | None = None, *, redactor: PIIRedactor | None = None) -> None:
        self.run_id = run_id or _new_id()
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._last_hash = GENESIS_HASH
        self._redactor = redactor

    def record(
        self,
        kind: EventKind | str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        parent_id: str | None = None,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> AuditRecord:
        """Append a new record to the chain and return it.

        If this tracer was constructed with a ``redactor``, PII is
        redacted here — before hashing — regardless of whether the
        caller already redacted it, and ``redaction_status``/
        ``redaction_policy_hash`` are stamped onto the record (metadata
        only in v2.0 — no enforcement yet). The stored payload is then
        deep-frozen (see :func:`deep_freeze`) so it can never drift out
        of sync with the hash already computed over it.
        """
        kind_value = kind.value if isinstance(kind, EventKind) else str(kind)
        eid = event_id or _new_id()
        ts = timestamp or datetime.now(timezone.utc)

        redaction_status: RedactionStatus = "none"
        redaction_policy_hash: str | None = None
        if self._redactor is not None:
            redacted = self._redactor.redact_payload(payload)
            redaction_policy_hash = self._redactor.policy_fingerprint()
            redaction_status = "redacted" if redacted != payload else "none"
            payload = redacted
        frozen_payload = deep_freeze(payload)
        payload_hash = compute_payload_hash(frozen_payload)

        with self._lock:
            prev_hash = self._last_hash
            record_hash = compute_record_hash(
                event_id=eid,
                parent_id=parent_id,
                run_id=self.run_id,
                kind=kind_value,
                actor=actor,
                payload_hash=payload_hash,
                timestamp=ts,
                prev_hash=prev_hash,
                redaction_policy_hash=redaction_policy_hash,
                redaction_status=redaction_status,
            )
            record = AuditRecord(
                event_id=eid,
                parent_id=parent_id,
                run_id=self.run_id,
                kind=kind_value,
                actor=actor,
                payload=frozen_payload,
                payload_hash=payload_hash,
                timestamp=ts,
                schema_version=SCHEMA_VERSION,
                prev_hash=prev_hash,
                hash=record_hash,
                redaction_policy_hash=redaction_policy_hash,
                redaction_status=redaction_status,
            )
            self._records.append(record)
            self._last_hash = record_hash
        return record

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """A read-only snapshot of the chain so far."""
        with self._lock:
            return tuple(self._records)

    def drain(self) -> list[AuditRecord]:
        """Atomically pop all buffered records (for flushing to a sink)."""
        with self._lock:
            records, self._records = self._records, []
            return records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @staticmethod
    def verify_chain(records: list[AuditRecord], *, expect_genesis: bool = True) -> None:
        """Recompute every hash and linkage; raise :class:`ChainIntegrityError` on any break."""
        prev_hash = GENESIS_HASH
        for i, rec in enumerate(records):
            if i == 0 and expect_genesis and rec.prev_hash != GENESIS_HASH:
                raise ChainIntegrityError(
                    f"record 0 ({rec.event_id}) does not start from the genesis hash"
                )
            if rec.prev_hash != prev_hash:
                raise ChainIntegrityError(
                    f"record {i} ({rec.event_id}) prev_hash does not match "
                    f"the hash of the preceding record — chain broken or reordered"
                )
            recomputed_payload_hash = compute_payload_hash(rec.payload)
            if recomputed_payload_hash != rec.payload_hash:
                raise ChainIntegrityError(
                    f"record {i} ({rec.event_id}) payload_hash does not match its payload — tampered payload"
                )
            recomputed = compute_record_hash(
                event_id=rec.event_id,
                parent_id=rec.parent_id,
                run_id=rec.run_id,
                kind=rec.kind,
                actor=rec.actor,
                payload_hash=rec.payload_hash,
                timestamp=rec.timestamp,
                prev_hash=rec.prev_hash,
                schema_version=rec.schema_version,
                redaction_policy_hash=rec.redaction_policy_hash,
                redaction_status=rec.redaction_status,
            )
            if recomputed != rec.hash:
                raise ChainIntegrityError(
                    f"record {i} ({rec.event_id}) hash does not match its content — tampered record"
                )
            prev_hash = rec.hash
