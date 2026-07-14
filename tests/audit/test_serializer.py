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


class TestUnicodeNormalization:
    """Regression test: payloads that render identically to a human but
    use different Unicode encodings (combining vs precomposed
    characters) must hash to the SAME value -- otherwise the "hash
    commits to the logical value" claim in the module docstring is
    false. On the buggy code these two forms of the same string hashed
    differently; on the fixed code they must be equal.
    """

    # "cafe" + precomposed U+00E9 (LATIN SMALL LETTER E WITH ACUTE) --
    # a single code point that renders as an accented e.
    _PRECOMPOSED = "café"
    # "cafe" + plain "e" (U+0065) + combining U+0301 (COMBINING ACUTE
    # ACCENT) -- two code points that render identically to the
    # precomposed form above, but are a different Python string.
    _COMBINING = "café"

    def test_precomposed_and_combining_forms_are_different_python_strings(self):
        # Sanity check the fixture itself is testing what it claims to.
        assert self._PRECOMPOSED != self._COMBINING
        assert len(self._PRECOMPOSED) != len(self._COMBINING)

    def test_combining_and_precomposed_accents_hash_the_same(self):
        assert hash_value({"name": self._PRECOMPOSED}) == hash_value({"name": self._COMBINING})

    def test_combining_and_precomposed_accents_produce_identical_canonical_json(self):
        assert canonical_json(self._PRECOMPOSED) == canonical_json(self._COMBINING)

    def test_normalization_applies_to_dict_keys_too(self):
        precomposed_key = {self._PRECOMPOSED: 1}
        combining_key = {self._COMBINING: 1}
        assert hash_value(precomposed_key) == hash_value(combining_key)

    def test_normalization_applies_inside_nested_lists(self):
        a = {"items": [self._PRECOMPOSED, "normal"]}
        b = {"items": [self._COMBINING, "normal"]}
        assert hash_value(a) == hash_value(b)

    def test_genuinely_different_strings_still_hash_differently(self):
        # Negative case: normalization must not collapse distinct content.
        assert hash_value({"x": "cafe"}) != hash_value({"x": self._PRECOMPOSED})
