"""Coverage for the v2.0 schema additions: EventKind taxonomy, actor,
payload_hash, reserved redaction fields, and the three-valued Verdict.
"""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from synapsekit.audit import (
    AuditTracer,
    EventKind,
    PIIRedactor,
    SigningPolicy,
    Verdict,
    export_audit_bundle,
)
from synapsekit.audit.types import AuditRecord
from synapsekit.audit.verifier import load_bundle, verify

from .conftest import dump_trace_lines, load_trace_lines, read_zip_entries, write_zip_entries


class TestEventKindTaxonomy:
    def test_all_spec_kinds_are_present(self):
        expected = {
            "USER_INPUT",
            "LLM_CALL",
            "LLM_RESPONSE",
            "TOOL_CALL",
            "TOOL_RESULT",
            "RETRIEVAL",
            "MEMORY_READ",
            "MEMORY_WRITE",
            "STATE_CHANGE",
            "DECISION",
            "SYSTEM_EVENT",
            "ERROR",
        }
        assert {k.value for k in EventKind} == expected

    def test_record_rejects_nothing_but_stores_whatever_string_is_given(self):
        # AuditTracer.record accepts EventKind OR str — the taxonomy is
        # enforced by convention/callers (e.g. VerifiableAgent), not by
        # a hard runtime check inside record() itself.
        tracer = AuditTracer()
        rec = tracer.record(EventKind.USER_INPUT, {"text": "hello"})
        assert rec.kind == "USER_INPUT"


class TestActorField:
    def test_default_actor_is_system(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.DECISION, {"x": 1})
        assert rec.actor == "system"

    def test_custom_actor_is_recorded(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.DECISION, {"x": 1}, actor="user:alice")
        assert rec.actor == "user:alice"

    def test_actor_is_committed_to_the_hash(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.DECISION, {"x": 1}, actor="user:alice")
        tracer2 = AuditTracer(run_id=tracer.run_id)
        r2 = tracer2.record(
            EventKind.DECISION,
            {"x": 1},
            actor="user:bob",
            event_id=r1.event_id,
            timestamp=r1.timestamp,
        )
        assert r1.hash != r2.hash


class TestPayloadHash:
    def test_payload_hash_is_deterministic_for_identical_payloads(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.DECISION, {"a": 1, "b": 2})
        tracer2 = AuditTracer()
        r2 = tracer2.record(EventKind.DECISION, {"b": 2, "a": 1})
        assert r1.payload_hash == r2.payload_hash

    def test_payload_hash_changes_with_payload(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.DECISION, {"a": 1})
        r2 = tracer.record(EventKind.DECISION, {"a": 2})
        assert r1.payload_hash != r2.payload_hash

    def test_record_hash_commits_to_payload_hash(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.DECISION, {"a": 1})
        assert rec.payload_hash in rec.to_dict().values()

    def test_tampering_payload_hash_alone_is_detected_as_drift(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["payload_hash"] = "f" * 64
        dump_trace_lines(entries, records)
        out = tmp_path / "bad_payload_hash.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert result.verdict == Verdict.DRIFT

    def test_tampering_payload_without_updating_payload_hash_is_detected(
        self, tmp_path, bundle_path
    ):
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["payload"] = {**records[0]["payload"], "prompt": "INJECTED"}
        dump_trace_lines(entries, records)
        out = tmp_path / "bad_payload.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert result.verdict == Verdict.DRIFT
        assert any("payload_hash" in e for e in result.errors)


class TestReservedRedactionFields:
    def test_defaults_are_none_and_not_redacted(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.DECISION, {"x": 1})
        assert rec.redaction_status == "none"
        assert rec.redaction_policy_hash is None

    def test_redactor_stamps_redacted_status_and_policy_hash(self):
        redactor = PIIRedactor()
        tracer = AuditTracer(redactor=redactor)
        rec = tracer.record(EventKind.TOOL_CALL, {"note": "contact alice@example.com"})
        assert rec.redaction_status == "redacted"
        assert rec.redaction_policy_hash == redactor.policy_fingerprint()

    def test_redactor_present_but_nothing_to_redact_stays_none(self):
        redactor = PIIRedactor()
        tracer = AuditTracer(redactor=redactor)
        rec = tracer.record(EventKind.TOOL_CALL, {"note": "nothing sensitive here"})
        assert rec.redaction_status == "none"

    def test_reserved_fields_round_trip_through_to_dict_and_from_dict(self):
        from synapsekit.audit import AuditRecord

        redactor = PIIRedactor()
        tracer = AuditTracer(redactor=redactor)
        rec = tracer.record(EventKind.TOOL_CALL, {"note": "contact alice@example.com"})
        rebuilt = AuditRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
        assert rebuilt.redaction_status == rec.redaction_status
        assert rebuilt.redaction_policy_hash == rec.redaction_policy_hash

    def test_tampering_redaction_status_is_detected(self, tmp_path, sample_records):
        redactor = PIIRedactor()
        tracer = AuditTracer(redactor=redactor)
        tracer.record(EventKind.TOOL_CALL, {"note": "contact alice@example.com"})
        path = export_audit_bundle(tracer.drain(), SigningPolicy.ed25519(), tmp_path / "b.zip")

        entries = read_zip_entries(path)
        records = load_trace_lines(entries)
        records[0]["redaction_status"] = "withheld"
        dump_trace_lines(entries, records)
        out = tmp_path / "tampered_redaction_status.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert result.verdict == Verdict.DRIFT


class TestVerdictCategorization:
    def test_valid_pinned_bundle_is_match(self, bundle_path):
        # MATCH now requires a pinned trust anchor; pin the bundle's own
        # advertised keys since this test only asserts structural validity.
        from .conftest import manifest_keys_as_trusted

        result = verify(bundle_path, trusted_keys=manifest_keys_as_trusted(bundle_path))
        assert result.verdict == Verdict.MATCH
        assert result.ok is True

    def test_valid_unpinned_bundle_is_unverifiable(self, bundle_path):
        # Without pinning, a would-be MATCH is capped at UNVERIFIABLE.
        result = verify(bundle_path)
        assert result.verdict == Verdict.UNVERIFIABLE
        assert result.ok is False
        assert result.trust_anchor == "none"

    def test_tampered_hash_is_drift_not_unverifiable(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["hash"] = "0" * 64
        dump_trace_lines(entries, records)
        out = tmp_path / "bad_hash.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert result.verdict == Verdict.DRIFT
        assert result.ok is False

    def test_unsupported_schema_version_is_unverifiable_not_drift(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        manifest["schema_version"] = "99.0"
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "bad_schema.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert result.verdict == Verdict.UNVERIFIABLE

    def test_corrupted_bundle_is_unverifiable(self, tmp_path):
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"not a zip file")
        result = verify(bad)
        assert result.verdict == Verdict.UNVERIFIABLE

    def test_missing_trusted_key_is_unverifiable_not_drift(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519(key_id="release-key")
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip")

        # Pin a *different* key_id entirely, so the bundle's signatures
        # reference a key we simply have no evidence about — that's
        # "can't tell" (UNVERIFIABLE), not "proven wrong" (DRIFT).
        result = verify(path, trusted_keys={"some-other-key-id": b"\x00" * 32})
        assert result.verdict == Verdict.UNVERIFIABLE

    def test_wrong_key_bytes_for_the_right_key_id_is_drift(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519(key_id="release-key")
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip")

        wrong_key = SigningPolicy.ed25519().provider.public_key_bytes()
        result = verify(path, trusted_keys={"release-key": wrong_key})
        assert result.verdict == Verdict.DRIFT

    def test_raise_if_invalid_includes_verdict_in_message(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["hash"] = "0" * 64
        dump_trace_lines(entries, records)
        out = tmp_path / "bad_hash.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        try:
            result.raise_if_invalid()
            raised = False
        except Exception as exc:
            raised = True
            assert "DRIFT" in str(exc)
        assert raised


class TestFromDictFreezesPayload:
    """Regression test: AuditRecord.from_dict must deep_freeze the payload
    it reconstructs, exactly like AuditTracer.record does — otherwise a
    record reconstructed from a loaded bundle (verifier/replay path) has
    a mutable payload dict, contradicting the class docstring's claim
    that "mutation raises instead of silently succeeding". On the buggy
    code, record.payload["x"] = ... succeeds silently; on the fixed code
    it raises TypeError because the payload is a MappingProxyType.
    """

    def test_from_dict_payload_is_frozen(self):
        rec = AuditRecord.from_dict(
            {
                "event_id": "e1",
                "parent_id": None,
                "run_id": "run-1",
                "kind": "SYSTEM_EVENT",
                "actor": "system",
                "payload": {"x": 1, "nested": {"y": 2}},
                "payload_hash": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "schema_version": "1.2",
                "prev_hash": "0" * 64,
                "hash": "1" * 64,
            }
        )
        assert isinstance(rec.payload, MappingProxyType)
        with pytest.raises(TypeError):
            rec.payload["x"] = 999  # type: ignore[index]

    def test_from_dict_nested_containers_are_also_frozen(self):
        rec = AuditRecord.from_dict(
            {
                "event_id": "e1",
                "parent_id": None,
                "run_id": "run-1",
                "kind": "SYSTEM_EVENT",
                "actor": "system",
                "payload": {"items": [1, 2, 3]},
                "payload_hash": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "schema_version": "1.2",
                "prev_hash": "0" * 64,
                "hash": "1" * 64,
            }
        )
        assert isinstance(rec.payload["items"], tuple)

    def test_records_reloaded_from_a_bundle_have_frozen_payloads(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")
        loaded = load_bundle(path)
        for rec in loaded.records:
            assert isinstance(rec.payload, MappingProxyType)
            with pytest.raises(TypeError):
                rec.payload["tampered"] = "value"  # type: ignore[index]
