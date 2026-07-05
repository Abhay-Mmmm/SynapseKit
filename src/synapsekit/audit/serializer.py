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
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any


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
    of dict insertion order, locale, or platform.
    """
    text = json.dumps(
        value,
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
