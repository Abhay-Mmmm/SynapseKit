"""Hash-chain integrity (trace.py) and Merkle tree correctness (merkle.py)."""

from __future__ import annotations

import dataclasses
import hashlib
from types import MappingProxyType

import pytest

from synapsekit.audit.merkle import MerkleHasher
from synapsekit.audit.redact import PIIRedactor
from synapsekit.audit.trace import AuditTracer, ChainIntegrityError
from synapsekit.audit.types import GENESIS_HASH, EventKind


class TestAuditTracer:
    def test_first_record_chains_from_genesis(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        assert rec.prev_hash == GENESIS_HASH

    def test_records_chain_sequentially(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        r2 = tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        assert r2.prev_hash == r1.hash

    def test_same_payload_produces_different_hash_due_to_chain_position(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        r2 = tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        assert r1.hash != r2.hash  # prev_hash differs even though payload is identical

    def test_drain_empties_and_returns_records(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        drained = tracer.drain()
        assert len(drained) == 1
        assert len(tracer) == 0

    def test_verify_chain_passes_for_untampered_chain(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        AuditTracer.verify_chain(list(tracer.records))  # should not raise

    def test_verify_chain_detects_payload_tamper(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        records = list(tracer.records)
        tampered = dataclasses.replace(records[0], payload={"x": 999})
        with pytest.raises(ChainIntegrityError):
            AuditTracer.verify_chain([tampered])

    def test_verify_chain_detects_reordering(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        records = list(tracer.records)
        with pytest.raises(ChainIntegrityError):
            AuditTracer.verify_chain([records[1], records[0]])

    def test_verify_chain_detects_direct_hash_tamper(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        records = list(tracer.records)
        tampered = dataclasses.replace(records[0], hash="0" * 64)
        with pytest.raises(ChainIntegrityError):
            AuditTracer.verify_chain([tampered])

    def test_verify_chain_detects_bad_parent_prev_hash_link(self):
        tracer = AuditTracer()
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        records = list(tracer.records)
        tampered_second = dataclasses.replace(records[1], prev_hash="f" * 64)
        with pytest.raises(ChainIntegrityError):
            AuditTracer.verify_chain([records[0], tampered_second])

    def test_verify_chain_rejects_unfiltered_multi_run_record_list(self):
        """Regression test: verify_chain had no run_id awareness — given
        an unfiltered multi-run record list, a record from run B whose
        prev_hash was forged to equal run A's last hash would previously
        be accepted as a legitimate cross-run link (the old code only
        checked prev_hash linkage, never run_id). It must now reject
        mixed-run input outright rather than silently validating a bogus
        cross-run chain.
        """
        run_a = AuditTracer(run_id="run-a")
        run_a.record(EventKind.SYSTEM_EVENT, {"x": 1})
        rec_a = run_a.records[0]

        run_b = AuditTracer(run_id="run-b")
        run_b.record(EventKind.SYSTEM_EVENT, {"x": 2})
        rec_b = run_b.records[0]
        # Forge run B's first record to chain from run A's last hash —
        # on the old code this "links" cleanly and passes.
        forged_rec_b = dataclasses.replace(rec_b, prev_hash=rec_a.hash)

        with pytest.raises(ChainIntegrityError):
            AuditTracer.verify_chain([rec_a, forged_rec_b], expect_genesis=True)

    def test_verify_chain_accepts_single_run_records(self):
        tracer = AuditTracer(run_id="run-only")
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 1})
        tracer.record(EventKind.SYSTEM_EVENT, {"x": 2})
        AuditTracer.verify_chain(list(tracer.records))  # should not raise

    def test_verify_chain_accepts_empty_list(self):
        AuditTracer.verify_chain([])  # should not raise


class TestPayloadImmutability:
    """A record's payload must never be able to drift from the content its hash commits to."""

    def test_payload_is_frozen_into_a_mappingproxy(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.SYSTEM_EVENT, {"a": 1, "nested": {"b": [1, 2, 3]}})
        assert isinstance(rec.payload, MappingProxyType)
        assert isinstance(rec.payload["nested"], MappingProxyType)
        assert rec.payload["nested"]["b"] == (1, 2, 3)

    def test_mutating_stored_payload_raises(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.SYSTEM_EVENT, {"a": 1})
        with pytest.raises(TypeError):
            rec.payload["a"] = 999

    def test_mutating_nested_stored_payload_raises(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.SYSTEM_EVENT, {"nested": {"b": 1}})
        with pytest.raises(TypeError):
            rec.payload["nested"]["b"] = 999

    def test_mutating_the_caller_supplied_dict_after_record_does_not_affect_the_stored_copy(self):
        tracer = AuditTracer()
        payload = {"a": 1}
        rec = tracer.record(EventKind.SYSTEM_EVENT, payload)
        payload["a"] = 999
        payload["new_key"] = "sneaky"
        assert rec.payload["a"] == 1
        assert "new_key" not in rec.payload

    def test_to_dict_yields_plain_json_friendly_containers(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.SYSTEM_EVENT, {"nested": {"b": [1, 2]}})
        d = rec.to_dict()
        assert type(d["payload"]) is dict
        assert type(d["payload"]["nested"]) is dict
        assert type(d["payload"]["nested"]["b"]) is list

    def test_frozen_payload_still_hashes_deterministically(self):
        tracer = AuditTracer()
        r1 = tracer.record(EventKind.SYSTEM_EVENT, {"a": 1, "b": {"c": 2}})
        tracer2 = AuditTracer(run_id=tracer.run_id)
        r2 = tracer2.record(
            EventKind.SYSTEM_EVENT,
            {"a": 1, "b": {"c": 2}},
            event_id=r1.event_id,
            timestamp=r1.timestamp,
        )
        assert r1.hash == r2.hash


class TestTracerLevelRedaction:
    """Redaction can be enforced at the AuditTracer itself, not only via VerifiableAgent."""

    def test_tracer_with_redactor_redacts_before_hashing(self):
        tracer = AuditTracer(redactor=PIIRedactor())
        rec = tracer.record(EventKind.TOOL_CALL, {"note": "email alice@example.com now"})
        assert "alice@example.com" not in rec.payload["note"]
        assert "[REDACTED:EMAIL]" in rec.payload["note"]

    def test_tracer_without_redactor_does_not_redact(self):
        tracer = AuditTracer()
        rec = tracer.record(EventKind.TOOL_CALL, {"note": "email alice@example.com now"})
        assert "alice@example.com" in rec.payload["note"]


class TestMerkleHasher:
    def test_empty_root_is_stable(self):
        assert MerkleHasher.root([]) == MerkleHasher.EMPTY_ROOT

    def test_single_leaf_root_equals_leaf(self):
        assert MerkleHasher.root(["a" * 64]) == "a" * 64

    def test_root_is_order_sensitive(self):
        leaves = ["1" * 64, "2" * 64, "3" * 64]
        assert MerkleHasher.root(leaves) != MerkleHasher.root(list(reversed(leaves)))

    def test_root_handles_odd_leaf_counts(self):
        leaves = ["1" * 64, "2" * 64, "3" * 64]
        # Should not raise, and should be deterministic.
        assert MerkleHasher.root(leaves) == MerkleHasher.root(leaves)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 16])
    def test_proof_verifies_for_every_leaf(self, n):
        leaves = [f"{i:064x}" for i in range(n)]
        root = MerkleHasher.root(leaves)
        for i in range(n):
            proof = MerkleHasher.proof(leaves, i)
            assert MerkleHasher.verify(proof, root)

    def test_proof_fails_against_wrong_root(self):
        leaves = [f"{i:064x}" for i in range(5)]
        proof = MerkleHasher.proof(leaves, 2)
        assert not MerkleHasher.verify(proof, "0" * 64)

    def test_proof_out_of_range_raises(self):
        with pytest.raises(IndexError):
            MerkleHasher.proof(["a" * 64], 5)

    def test_proof_on_empty_tree_raises(self):
        with pytest.raises(ValueError):
            MerkleHasher.proof([], 0)


class TestMerkleDuplicateLeafAmbiguity:
    """Regression test for the CVE-2012-2459 class of bug: a tree must
    never produce the same root for two structurally different leaf
    sets just because an odd count got padded with a duplicate.
    """

    def test_three_leaves_and_a_duplicated_fourth_leaf_produce_different_roots(self):
        leaves3 = ["1" * 64, "2" * 64, "3" * 64]
        leaves4_with_duplicate = [*leaves3, "3" * 64]
        assert MerkleHasher.root(leaves3) != MerkleHasher.root(leaves4_with_duplicate)

    def test_internal_node_hash_is_domain_separated_from_a_raw_leaf(self):
        # An internal node hash must not collide with a value that could
        # plausibly be presented as a leaf: sha256(leaf_a || leaf_b)
        # without the 0x01 node-domain prefix would be indistinguishable
        # from "just another leaf hash".
        a, b = "1" * 64, "2" * 64
        node_hash = MerkleHasher.root([a, b])
        undomained = hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()
        assert node_hash != undomained


class TestLeafHashDomainSeparation:
    """Regression test: before the fix, only internal Merkle nodes got
    domain separation (0x01 prefix, RFC 6962-style) -- leaf hashes were
    used raw, un-prefixed. Full RFC 6962 also domain-separates leaves
    with a 0x00 prefix so a leaf hash can never be replayed as if it
    were an internal node hash's input (or vice versa). This exercises
    synapsekit.audit.merkle.hash_leaf directly.
    """

    def test_hash_leaf_is_not_the_raw_value(self):
        from synapsekit.audit.merkle import hash_leaf

        raw = "a" * 64
        assert hash_leaf(raw) != raw

    def test_hash_leaf_uses_the_0x00_domain_prefix(self):
        from synapsekit.audit.merkle import hash_leaf

        raw = "ab" * 32
        expected = hashlib.sha256(b"\x00" + bytes.fromhex(raw)).hexdigest()
        assert hash_leaf(raw) == expected

    def test_hash_leaf_is_deterministic(self):
        from synapsekit.audit.merkle import hash_leaf

        raw = "c" * 64
        assert hash_leaf(raw) == hash_leaf(raw)

    def test_hash_leaf_output_is_domain_separated_from_hash_pair_output(self):
        # A leaf hash (0x00 prefix) must never equal an internal node
        # hash (0x01 prefix) computed over the same underlying bytes.
        from synapsekit.audit.merkle import hash_leaf

        a, b = "1" * 64, "2" * 64
        node_hash = MerkleHasher.root([a, b])
        leaf_hash_of_a = hash_leaf(a)
        assert leaf_hash_of_a != node_hash

    def test_single_record_leaf_hash_differs_from_raw_record_hash_used_as_root(self):
        # A single-leaf tree's root is the leaf itself in MerkleHasher's
        # generic API (see test_single_leaf_root_equals_leaf) -- but the
        # actual leaf fed in must be the domain-separated hash_leaf(...)
        # of the record hash, not the record hash directly, or leaves
        # and internal nodes share an un-separated hash space.
        from synapsekit.audit.merkle import hash_leaf

        record_hash = "7" * 64
        leaf = hash_leaf(record_hash)
        assert MerkleHasher.root([leaf]) == leaf
        assert MerkleHasher.root([leaf]) != record_hash
