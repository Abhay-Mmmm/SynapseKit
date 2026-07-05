"""Deterministic canonical serialization must not depend on ordering/whitespace/platform quirks."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synapsekit.audit.serializer import DeterministicSerializer, canonical_json, hash_value


class TestCanonicalJSON:
    def test_key_order_does_not_affect_output(self):
        a = {"b": 1, "a": 2, "c": 3}
        b = {"c": 3, "a": 2, "b": 1}
        assert canonical_json(a) == canonical_json(b)

    def test_no_whitespace_variance(self):
        assert canonical_json({"a": 1}) == b'{"a":1}'

    def test_nested_structures_are_deterministic(self):
        a = {"x": {"z": 1, "y": 2}, "list": [3, 2, 1]}
        b = {"list": [3, 2, 1], "x": {"y": 2, "z": 1}}
        assert canonical_json(a) == canonical_json(b)

    def test_datetime_is_normalized_to_utc_isoformat(self):
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        out = canonical_json({"ts": dt})
        assert b"2026-01-01T12:00:00+00:00" in out

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})

    def test_hash_is_stable_across_key_order(self):
        assert hash_value({"a": 1, "b": 2}) == hash_value({"b": 2, "a": 1})

    def test_hash_changes_with_content(self):
        assert hash_value({"a": 1}) != hash_value({"a": 2})

    def test_deterministic_serializer_matches_module_functions(self):
        value = {"z": 1, "a": [1, 2, 3]}
        assert DeterministicSerializer.canonicalize(value) == canonical_json(value)
        assert DeterministicSerializer.hash(value) == hash_value(value)
