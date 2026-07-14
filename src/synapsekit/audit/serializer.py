"""Deterministic canonical serialization for hashing.

Hashing must never depend on dict ordering, whitespace, float repr, the
Python version, or the platform. We never hash ``repr()`` of a Python
object — everything that gets hashed first passes through
:func:`canonical_json`, which produces the same bytes for the same
logical value on every machine.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any


def _normalize_strings(value: Any) -> Any:
    """Recursively apply Unicode NFC normalization to every string leaf.

    Two payloads that render identically to a human can be byte-for-byte
    different Python strings — e.g. an "e" followed by a combining acute
    accent (``"e\\u0301"``) versus the single precomposed "é"
    (``"\\u00e9"``). Without normalizing to one canonical form first,
    those would hash differently even though they commit to the same
    logical value, which is exactly what :func:`canonical_json` promises
    not to happen. This must run *before* ``json.dumps`` — dict keys are
    strings too and need the same treatment so two logically-identical
    keys don't collide or fail to collide inconsistently.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict | MappingProxyType):
        return {_normalize_strings(k): _normalize_strings(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_strings(v) for v in value]
    if isinstance(value, set | frozenset):
        return {_normalize_strings(v) for v in value}
    return value


def _default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, bytes):
        import base64

        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, MappingProxyType):
        # Read-only view over an already-frozen record payload (see
        # trace.py's deep-freeze) — unwrap one level; the encoder recurses
        # into the result and will call back into `_default` for any
        # nested MappingProxyType/frozenset values it still contains.
        return dict(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=repr)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not canonically serializable")


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` to canonical (deterministic) JSON bytes.

    Uses sorted keys, minimal separators, and ASCII-only output so the
    same logical value always produces byte-identical output regardless
    of dict insertion order, locale, or platform. String values (and
    dict keys) are also normalized to Unicode NFC first, so visually
    identical text using different Unicode encodings (combining vs
    precomposed characters) hashes to the same value.
    """
    normalized = _normalize_strings(value)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_default,
    )
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_value(value: Any) -> str:
    """Canonicalize ``value`` and return its hex SHA-256 digest."""
    return sha256_hex(canonical_json(value))


class DeterministicSerializer:
    """Namespace wrapper around the canonical-serialization functions.

    Kept as a class (rather than bare functions) so callers can swap in
    an alternate serializer via dependency injection without changing
    call sites — e.g. a stricter variant that rejects floats entirely.
    """

    @staticmethod
    def canonicalize(value: Any) -> bytes:
        return canonical_json(value)

    @staticmethod
    def hash(value: Any) -> str:
        return hash_value(value)
