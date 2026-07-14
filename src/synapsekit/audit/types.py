"""Immutable audit record model — the unit of the verifiable trace chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal


def deep_freeze(value: Any) -> Any:
    """Recursively convert dict/list/tuple/set into read-only equivalents.

    ``@dataclass(frozen=True)`` on :class:`AuditRecord` only stops
    *rebinding* ``record.payload`` — it does nothing to stop
    ``record.payload["x"] = "y"`` in place, which would silently desync
    the live object from the hash already computed over its original
    content. Freezing the payload's containers (not just the outer
    field) closes that gap: mutation attempts now raise instead of
    silently succeeding. Primitive leaves and any other object type are
    returned unchanged.
    """
    if isinstance(value, dict):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, set | frozenset):
        return frozenset(deep_freeze(v) for v in value)
    return value


def deep_unfreeze(value: Any) -> Any:
    """Inverse of :func:`deep_freeze` — plain, JSON-friendly dict/list output."""
    if isinstance(value, MappingProxyType):
        return {k: deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple | frozenset):
        return [deep_unfreeze(v) for v in value]
    return value


#: Bundle/record schema version. Bump on any breaking change to the
#: canonical hash inputs, the bundle layout, or the manifest fields —
#: verifiers must refuse to process a version they don't recognize.
#:
#: - 1.1 switched the Merkle construction from naive last-leaf duplication
#:   (the CVE-2012-2459 class of ambiguity) to the RFC 6962 recursive-split
#:   algorithm with domain-separated leaf/node hashing, and added signed
#:   manifest metadata.
#: - 1.2 renamed ``step_id`` to ``event_id``, added ``actor`` and a
#:   two-level ``payload_hash``/``hash`` commitment (see
#:   :mod:`synapsekit.audit.trace`), replaced the ad hoc ``kind`` set with
#:   the stable :class:`EventKind` taxonomy, reserved
#:   ``redaction_policy_hash``/``redaction_status`` fields for future
#:   compliance enforcement, and added the RFC 6962 0x00 leaf-hash domain
#:   prefix (see :func:`synapsekit.audit.merkle.hash_leaf`) so leaf hashes
#:   can never be replayed as internal node hashes. Bundles from 1.1 and
#:   earlier are not compatible with 1.2 verifiers. 1.2 has not shipped
#:   yet, so this leaf-hashing change lands within the same version
#:   rather than bumping to 1.3.
SCHEMA_VERSION = "1.2"

#: prev_hash of the first record in a run — there is nothing before it.
GENESIS_HASH = "0" * 64

#: Redaction lifecycle for a record's payload — reserved for future
#: compliance enforcement (HIPAA/GDPR/EU AI Act redaction policies). v2.0
#: only stores this metadata; it does not yet enforce anything based on it.
RedactionStatus = Literal["none", "redacted", "withheld"]


class EventKind(str, Enum):
    """The stable, public event taxonomy — arbitrary strings are not allowed.

    Renaming or removing a value is a breaking change to the bundle
    format; adding a new value is not (old verifiers simply don't
    recognize the new kind string, they don't reject the record for it).
    """

    USER_INPUT = "USER_INPUT"
    LLM_CALL = "LLM_CALL"
    LLM_RESPONSE = "LLM_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    RETRIEVAL = "RETRIEVAL"
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    STATE_CHANGE = "STATE_CHANGE"
    DECISION = "DECISION"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AuditRecord:
    """A single, immutable, hash-chained audit event.

    ``hash`` commits to every other field (including ``prev_hash`` and
    ``payload_hash``, but not the raw ``payload`` directly — see
    :mod:`synapsekit.audit.trace`), so mutating any prior record changes
    its hash and breaks the chain for every record that follows it —
    this is what makes the trace tamper-evident rather than merely
    append-only.

    ``redaction_policy_hash``/``redaction_status`` are reserved for
    future compliance enforcement (HIPAA/GDPR/EU AI Act) — they exist in
    the schema now specifically so that turning on real enforcement
    later never requires a breaking bundle-format change. v2.0 only
    records this metadata; it does not enforce anything based on it.
    """

    event_id: str
    parent_id: str | None
    run_id: str
    kind: str
    actor: str
    payload: dict[str, Any]
    payload_hash: str
    timestamp: datetime
    schema_version: str
    prev_hash: str
    hash: str

    # Reserved for future compliance support — do not remove.
    redaction_policy_hash: str | None = None
    redaction_status: RedactionStatus = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "actor": self.actor,
            "payload": deep_unfreeze(self.payload),
            "payload_hash": self.payload_hash,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "schema_version": self.schema_version,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "redaction_policy_hash": self.redaction_policy_hash,
            "redaction_status": self.redaction_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            event_id=data["event_id"],
            parent_id=data.get("parent_id"),
            run_id=data["run_id"],
            kind=data["kind"],
            actor=data.get("actor", "unknown"),
            payload=deep_freeze(data["payload"]),
            payload_hash=data["payload_hash"],
            timestamp=ts,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            prev_hash=data["prev_hash"],
            hash=data["hash"],
            redaction_policy_hash=data.get("redaction_policy_hash"),
            redaction_status=data.get("redaction_status", "none"),
        )


@dataclass(frozen=True)
class Signature:
    """A single Ed25519/KMS signature over a Merkle root (a signed batch)."""

    algorithm: str
    key_id: str
    public_key_b64: str
    signature_b64: str
    merkle_root: str
    signed_at: datetime
    start_index: int
    end_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "public_key_b64": self.public_key_b64,
            "signature_b64": self.signature_b64,
            "merkle_root": self.merkle_root,
            "signed_at": self.signed_at.astimezone(timezone.utc).isoformat(),
            "start_index": self.start_index,
            "end_index": self.end_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signature:
        signed_at = data["signed_at"]
        if isinstance(signed_at, str):
            signed_at = datetime.fromisoformat(signed_at)
        return cls(
            algorithm=data["algorithm"],
            key_id=data["key_id"],
            public_key_b64=data["public_key_b64"],
            signature_b64=data["signature_b64"],
            merkle_root=data["merkle_root"],
            signed_at=signed_at,
            start_index=data["start_index"],
            end_index=data["end_index"],
        )


class Verdict(str, Enum):
    """The three possible outcomes of :func:`synapsekit.audit.verifier.verify`.

    This is deliberately not a bool: a verifier can fail to reach a
    conclusion (``UNVERIFIABLE`` — e.g. a corrupted bundle, an
    unsupported schema version, or a key it has no way to check against)
    without that meaning the evidence actively *contradicts* what's
    claimed (``DRIFT`` — a broken hash chain, an invalid signature, a
    tampered Merkle root). Collapsing that distinction into a single
    boolean would make "we couldn't check" indistinguishable from "we
    checked and it's wrong".
    """

    MATCH = "MATCH"
    DRIFT = "DRIFT"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class VerificationResult:
    """Outcome of verifying a bundle.

    ``trust_anchor`` distinguishes two very different claims:

    - ``"none"`` — only internal self-consistency was checked (hash
      chain, Merkle tree, and signatures all verify against public keys
      *sourced from the bundle itself*). This proves the bundle wasn't
      tampered with after it was produced, but NOT that it was produced
      by any particular party — anyone can generate a self-consistent
      bundle from scratch with their own keypair.
    - ``"pinned"`` — every signature (batch and manifest) was checked
      against a caller-supplied ``trusted_keys`` map obtained
      independently of the bundle, giving real non-repudiation.
    """

    verdict: Verdict
    errors: list[str] = field(default_factory=list)
    record_count: int = 0
    batch_count: int = 0
    schema_version: str | None = None
    trust_anchor: str = "none"

    @property
    def ok(self) -> bool:
        """Convenience alias for ``verdict == Verdict.MATCH``."""
        return self.verdict == Verdict.MATCH

    def raise_if_invalid(self) -> None:
        if self.verdict != Verdict.MATCH:
            raise AuditVerificationError(
                f"{self.verdict.value}: " + ("; ".join(self.errors) or "verification failed")
            )


class AuditVerificationError(Exception):
    """Raised when a bundle fails cryptographic or structural verification."""
