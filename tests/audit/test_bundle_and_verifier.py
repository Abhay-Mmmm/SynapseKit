"""Bundle export + standalone verification — positive and negative (tamper) cases."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from synapsekit.audit import SigningPolicy, export_audit_bundle
from synapsekit.audit.verifier import verify

from .conftest import dump_trace_lines, load_trace_lines, read_zip_entries, write_zip_entries


class TestValidBundle:
    def test_valid_bundle_verifies_ok(self, bundle_path: Path):
        result = verify(bundle_path)
        assert result.ok
        assert result.errors == []
        assert result.record_count == 4

    def test_batch_count_matches_number_of_signed_batches(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip", batch_size=2)
        result = verify(path)
        assert result.ok
        assert result.batch_count == 2

    def test_empty_record_set_produces_a_verifiable_empty_bundle(self, tmp_path):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle([], policy, tmp_path / "empty.zip")
        result = verify(path)
        assert result.ok
        assert result.record_count == 0


class TestNegativeVerification:
    """Every listed tamper scenario MUST fail verification — never silently pass."""

    def _mutate(self, tmp_path: Path, bundle_path: Path, mutate_records) -> Path:
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records = mutate_records(records)
        dump_trace_lines(entries, records)
        out = tmp_path / "mutated.zip"
        write_zip_entries(out, entries)
        return out

    def test_remove_a_record(self, tmp_path, bundle_path):
        mutated = self._mutate(tmp_path, bundle_path, lambda recs: recs[:-1])
        result = verify(mutated)
        assert not result.ok

    def test_reorder_records(self, tmp_path, bundle_path):
        def swap(recs):
            recs = list(recs)
            recs[0], recs[1] = recs[1], recs[0]
            return recs

        mutated = self._mutate(tmp_path, bundle_path, swap)
        result = verify(mutated)
        assert not result.ok

    def test_modify_payload(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["payload"] = {**recs[0]["payload"], "prompt": "INJECTED"}
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_modify_timestamp(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["timestamp"] = "2099-01-01T00:00:00+00:00"
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_modify_hash(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["hash"] = "0" * 64
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_invalid_parent_id(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[1]["parent_id"] = "does-not-exist"
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok
        assert any("parent_id" in e for e in result.errors)

    def test_invalid_merkle_root(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        hashes_doc["batches"][0]["merkle_root"] = "f" * 64
        entries["hashes.merkle"] = json.dumps(hashes_doc).encode("utf-8")
        out = tmp_path / "bad_merkle.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_invalid_manifest_missing_record_count(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        del manifest["record_count"]
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "bad_manifest.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok


class TestMerkleLeafDomainSeparation:
    """Regression test: the exported bundle's hashes.merkle leaves must be
    the RFC 6962 domain-separated leaf hashes (hash_leaf(record.hash)),
    not the raw record.hash values. On the buggy code, leaves ==
    [r.hash for r in records] -- indistinguishable from an internal node
    input. A valid bundle must still verify end to end with the fix.
    """

    def test_recorded_leaves_are_not_raw_record_hashes(self, bundle_path, sample_records):
        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        recorded_leaves = hashes_doc["batches"][0]["leaves"]
        raw_hashes = [r.hash for r in sample_records]
        assert recorded_leaves != raw_hashes

    def test_recorded_leaves_match_hash_leaf_of_record_hash(self, bundle_path, sample_records):
        from synapsekit.audit.merkle import hash_leaf

        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        recorded_leaves = hashes_doc["batches"][0]["leaves"]
        expected = [hash_leaf(r.hash) for r in sample_records]
        assert recorded_leaves == expected

    def test_valid_bundle_with_domain_separated_leaves_still_verifies(self, bundle_path):
        result = verify(bundle_path)
        assert result.ok
        assert result.errors == []

    def test_replaying_raw_record_hash_as_a_leaf_is_rejected(self, tmp_path, bundle_path):
        # Simulate an attacker who tries to pass a raw record.hash off as
        # a pre-domain-separated leaf -- must fail Merkle verification.
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        hashes_doc = json.loads(entries["hashes.merkle"])
        hashes_doc["batches"][0]["leaves"][0] = records[0]["hash"]
        entries["hashes.merkle"] = json.dumps(hashes_doc).encode("utf-8")
        out = tmp_path / "raw_hash_as_leaf.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_schema_version_mismatch(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        manifest["schema_version"] = "99.0"
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "bad_schema.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok
        assert "unsupported schema version" in result.errors[0]

    def test_wrong_public_key(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        other = SigningPolicy.ed25519()
        for key_id in manifest["keys"]:
            manifest["keys"][key_id]["public_key_b64"] = base64.b64encode(
                other.provider.public_key_bytes()
            ).decode("ascii")
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "wrong_key.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_missing_zip_entry_is_reported_not_crashed(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        del entries["signatures.json"]
        out = tmp_path / "missing_entry.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok
        assert result.errors  # never silently succeeds

    def test_corrupted_zip_is_reported_not_crashed(self, tmp_path):
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"not a zip file at all")
        result = verify(bad)
        assert not result.ok

    def test_missing_bundle_file_is_reported_not_crashed(self, tmp_path):
        result = verify(tmp_path / "does_not_exist.zip")
        assert not result.ok


class TestKeyRotation:
    def test_verification_succeeds_across_rotated_keys_in_one_bundle(
        self, tmp_path, sample_records
    ):
        key_a = SigningPolicy.ed25519(key_id="key-a")
        key_b = SigningPolicy.ed25519(key_id="key-b")
        batch_a = sample_records[:2]
        batch_b = sample_records[2:]

        path = export_audit_bundle(
            [],
            [(batch_a, key_a), (batch_b, key_b)],
            tmp_path / "rotated.zip",
        )
        result = verify(path)
        assert result.ok
        assert result.batch_count == 2

    def test_tampering_after_rotation_still_caught(self, tmp_path, sample_records):
        key_a = SigningPolicy.ed25519(key_id="key-a")
        key_b = SigningPolicy.ed25519(key_id="key-b")
        batch_a = sample_records[:2]
        batch_b = sample_records[2:]
        path = export_audit_bundle(
            [], [(batch_a, key_a), (batch_b, key_b)], tmp_path / "rotated2.zip"
        )

        entries = read_zip_entries(path)
        records = load_trace_lines(entries)
        records[3]["payload"] = {**records[3]["payload"], "decision": "TAMPERED"}
        dump_trace_lines(entries, records)
        out = tmp_path / "rotated2_tampered.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert not result.ok
