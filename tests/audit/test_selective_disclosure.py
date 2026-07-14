"""Selective disclosure — exporting a subset of records must stay verifiable.

A subset export can't recompute the full original Merkle batch (most
leaves are intentionally missing), so each kept record carries its own
Merkle inclusion proof against the original signed root instead.
"""

from __future__ import annotations

from synapsekit.audit import EventKind, export_selective_bundle
from synapsekit.audit.verifier import load_bundle, verify


class TestSelectiveDisclosure:
    def test_only_tool_calls_are_kept(self, tmp_path, bundle_path):
        out = export_selective_bundle(
            bundle_path, tmp_path / "tool_only.zip", kinds=[EventKind.TOOL_CALL.value]
        )
        loaded = load_bundle(out)
        assert all(r.kind == EventKind.TOOL_CALL.value for r in loaded.records)
        assert len(loaded.records) == 1

    def test_only_retrieval_events_are_kept(self, tmp_path, bundle_path):
        out = export_selective_bundle(
            bundle_path, tmp_path / "retrieval_only.zip", kinds=[EventKind.RETRIEVAL.value]
        )
        loaded = load_bundle(out)
        assert all(r.kind == EventKind.RETRIEVAL.value for r in loaded.records)

    def test_selective_bundle_still_verifies(self, tmp_path, bundle_path):
        from .conftest import manifest_keys_as_trusted

        out = export_selective_bundle(
            bundle_path, tmp_path / "subset.zip", kinds=[EventKind.LLM_CALL.value]
        )
        # MATCH requires pinning; pin the original signer's keys carried in
        # the selective manifest.
        result = verify(out, trusted_keys=manifest_keys_as_trusted(out))
        assert result.ok
        assert result.errors == []
        assert result.record_count == 1

    def test_selective_bundle_unpinned_is_unverifiable(self, tmp_path, bundle_path):
        from synapsekit.audit.types import Verdict

        out = export_selective_bundle(
            bundle_path, tmp_path / "subset.zip", kinds=[EventKind.LLM_CALL.value]
        )
        result = verify(out)
        assert result.verdict == Verdict.UNVERIFIABLE
        assert result.trust_anchor == "none"

    def test_original_signatures_are_carried_through_unchanged(self, tmp_path, bundle_path):
        original = load_bundle(bundle_path)
        out = export_selective_bundle(
            bundle_path, tmp_path / "subset.zip", kinds=[EventKind.LLM_CALL.value]
        )
        subset = load_bundle(out)
        assert subset.signatures_doc == original.signatures_doc

    def test_manifest_flags_selective_disclosure(self, tmp_path, bundle_path):
        out = export_selective_bundle(
            bundle_path, tmp_path / "flagged.zip", kinds=[EventKind.LLM_CALL.value]
        )
        loaded = load_bundle(out)
        assert loaded.manifest["selective_disclosure"] is True
        assert loaded.manifest["record_count"] == 1

    def test_tampering_a_disclosed_record_is_still_detected(self, tmp_path, bundle_path):
        from .conftest import (
            dump_trace_lines,
            load_trace_lines,
            read_zip_entries,
            write_zip_entries,
        )

        out = export_selective_bundle(
            bundle_path, tmp_path / "subset.zip", kinds=[EventKind.LLM_CALL.value]
        )
        entries = read_zip_entries(out)
        records = load_trace_lines(entries)
        records[0]["payload"] = {**records[0]["payload"], "prompt": "INJECTED"}
        dump_trace_lines(entries, records)
        tampered = tmp_path / "subset_tampered.zip"
        write_zip_entries(tampered, entries)

        result = verify(tampered)
        assert not result.ok

    def test_multi_batch_bundle_selective_disclosure_still_verifies(self, tmp_path, sample_records):
        from synapsekit.audit import SigningPolicy, export_audit_bundle

        from .conftest import manifest_keys_as_trusted

        path = export_audit_bundle(
            sample_records, SigningPolicy.ed25519(), tmp_path / "multi.zip", batch_size=2
        )
        out = export_selective_bundle(
            path, tmp_path / "multi_subset.zip", kinds=[EventKind.DECISION.value]
        )
        result = verify(out, trusted_keys=manifest_keys_as_trusted(out))
        assert result.ok
        assert result.record_count == 1
