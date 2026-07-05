"""Trust-anchor pinning — closes the "self-certifying bundle" gap.

Without ``trusted_keys``, ``verify()`` can only prove a bundle wasn't
edited after export, since it reads public keys from the bundle's own
manifest. These tests exercise the actual attack that gap allows (an
attacker forging a complete, internally-consistent bundle from scratch)
and confirm pinning closes it.
"""

from __future__ import annotations

import base64

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.verifier import verify


def _forge_bundle(tmp_path):
    """Simulate an attacker: fabricate records from scratch, sign them
    with a freshly generated keypair they control, and export a bundle
    that is fully self-consistent."""
    forged_tracer = AuditTracer()
    forged_tracer.record(EventKind.DECISION, {"decision": "approve loan", "amount": 1_000_000})
    attacker_policy = SigningPolicy.ed25519()
    forged_path = export_audit_bundle(
        forged_tracer.drain(), attacker_policy, tmp_path / "forged.zip"
    )
    return forged_path, attacker_policy


class TestTrustAnchorPinning:
    def test_forged_bundle_passes_self_consistency_check_without_pinning(self, tmp_path):
        # This documents the known, unavoidable limitation of
        # self-consistency-only verification: a fully self-consistent
        # forgery is indistinguishable from a real bundle unless the
        # verifier pins an independently-known key.
        forged_path, _ = _forge_bundle(tmp_path)
        result = verify(forged_path)
        assert result.ok
        assert result.trust_anchor == "none"

    def test_forged_bundle_is_rejected_when_the_real_key_is_pinned(self, tmp_path):
        forged_path, _attacker_policy = _forge_bundle(tmp_path)

        real_org_policy = SigningPolicy.ed25519(key_id="real-org-key")
        trusted_keys = {"real-org-key": real_org_policy.provider.public_key_bytes()}

        result = verify(forged_path, trusted_keys=trusted_keys)
        assert not result.ok
        assert result.trust_anchor == "pinned"
        assert any("trusted key set" in e for e in result.errors)

    def test_genuine_bundle_verifies_when_its_real_key_is_pinned(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519(key_id="real-org-key")
        path = export_audit_bundle(sample_records, policy, tmp_path / "genuine.zip")

        trusted_keys = {"real-org-key": policy.provider.public_key_bytes()}
        result = verify(path, trusted_keys=trusted_keys)
        assert result.ok
        assert result.trust_anchor == "pinned"

    def test_pinning_ignores_a_manifest_key_masquerading_under_a_trusted_key_id(self, tmp_path):
        """Even if an attacker labels their forged key with the SAME key_id
        as the real signer, pinning must use the caller-supplied bytes —
        never the manifest's own claimed public key for that id."""
        forged_path, attacker_policy = _forge_bundle(tmp_path)

        import json
        import zipfile

        with zipfile.ZipFile(forged_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        attacker_key_id = next(iter(manifest["keys"]))

        real_key_bytes = SigningPolicy.ed25519().provider.public_key_bytes()
        assert real_key_bytes != attacker_policy.provider.public_key_bytes()

        result = verify(forged_path, trusted_keys={attacker_key_id: real_key_bytes})
        assert not result.ok


class TestManifestMetadataAuthentication:
    def test_manifest_created_at_tampering_is_detected(self, tmp_path, bundle_path):
        import json

        from .conftest import read_zip_entries, write_zip_entries

        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        manifest["created_at"] = "1999-01-01T00:00:00+00:00"
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "tampered_created_at.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert not result.ok
        assert any("manifest_hash" in e or "manifest" in e.lower() for e in result.errors)

    def test_manifest_run_ids_tampering_is_detected(self, tmp_path, bundle_path):
        import json

        from .conftest import read_zip_entries, write_zip_entries

        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        manifest["run_ids"] = ["not-the-real-run-id"]
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "tampered_run_ids.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert not result.ok

    def test_missing_manifest_signature_is_rejected(self, tmp_path, bundle_path):
        import json

        from .conftest import read_zip_entries, write_zip_entries

        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        del manifest["manifest_hash"]
        del manifest["manifest_signatures"]
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "unsigned_manifest.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert not result.ok
        assert any("not signed" in e for e in result.errors)

    def test_valid_manifest_signature_present_in_a_fresh_export(self, bundle_path):
        import json
        import zipfile

        with zipfile.ZipFile(bundle_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert "manifest_hash" in manifest
        assert manifest["manifest_signatures"]
        for sig in manifest["manifest_signatures"]:
            assert base64.b64decode(sig["signature_b64"])
