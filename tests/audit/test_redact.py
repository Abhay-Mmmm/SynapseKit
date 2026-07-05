"""PII redaction: PII removed, hashes deterministic, and signatures remain valid post-redaction."""

from __future__ import annotations

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.redact import PIIRedactor
from synapsekit.audit.serializer import hash_value
from synapsekit.audit.verifier import verify


class TestPIIRedactor:
    def test_email_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("contact me at alice@example.com please")
        assert "alice@example.com" not in out
        assert "[REDACTED:EMAIL]" in out

    def test_phone_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("call 555-123-4567 now")
        assert "555-123-4567" not in out

    def test_ssn_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("ssn: 123-45-6789")
        assert "123-45-6789" not in out
        assert "[REDACTED:SSN]" in out

    def test_credit_card_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("card 4111111111111111 expires soon")
        assert "4111111111111111" not in out

    def test_ip_address_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("connect to 192.168.1.100 now")
        assert "192.168.1.100" not in out

    def test_text_without_pii_is_unchanged(self):
        redactor = PIIRedactor()
        text = "just a normal sentence with no secrets"
        assert redactor.redact_text(text) == text

    def test_redact_payload_recurses_through_nested_structures(self):
        redactor = PIIRedactor()
        payload = {"user": {"email": "bob@example.com", "notes": ["call 555-987-6543"]}}
        redacted = redactor.redact_payload(payload)
        assert "bob@example.com" not in redacted["user"]["email"]
        assert "555-987-6543" not in redacted["user"]["notes"][0]

    def test_redaction_count_increments(self):
        redactor = PIIRedactor()
        redactor.redact_text("a@b.com and c@d.com")
        assert redactor.redaction_count == 2

    def test_pluggable_ml_detector_supplements_regex_detectors(self):
        def fake_ner(text: str):
            idx = text.find("Bob Smith")
            return [(idx, idx + len("Bob Smith"), "PERSON")] if idx >= 0 else []

        redactor = PIIRedactor(ml_detector=fake_ner)
        out = redactor.redact_text("My name is Bob Smith and my email is x@y.com")
        assert "Bob Smith" not in out
        assert "x@y.com" not in out


class TestRedactionBeforeHashing:
    """Redaction must run BEFORE hashing — this is a documented, deliberate tradeoff."""

    def test_hashing_the_redacted_payload_is_deterministic(self):
        redactor = PIIRedactor()
        payload = {"note": "email me at test@example.com"}
        redacted_once = redactor.redact_payload(payload)
        redacted_twice = PIIRedactor().redact_payload(payload)
        assert hash_value(redacted_once) == hash_value(redacted_twice)

    def test_bundle_built_from_redacted_payloads_verifies_normally(self, tmp_path):
        redactor = PIIRedactor()
        tracer = AuditTracer()
        raw_payload = {"tool": "email_tool", "input": {"to": "alice@example.com"}, "output": "sent"}
        tracer.record(EventKind.TOOL_CALL, redactor.redact_payload(raw_payload))

        path = export_audit_bundle(
            tracer.drain(), SigningPolicy.ed25519(), tmp_path / "redacted.zip"
        )
        result = verify(path)
        assert result.ok

        from .conftest import read_zip_entries

        entries = read_zip_entries(path)
        assert b"alice@example.com" not in entries["trace.jsonl"]
