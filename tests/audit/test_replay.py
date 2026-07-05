"""Replay: verifies provenance (inputs/tool outputs/retrieval refs/policy hashes),
never requires deterministic LLM output.
"""

from __future__ import annotations

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.replay import ReplayEngine
from synapsekit.audit.serializer import hash_value


class TestReplaySucceeds:
    def test_replay_succeeds_on_a_clean_bundle(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")
        report = ReplayEngine().replay(path)
        assert report.ok
        assert report.record_count == 4
        assert report.checked_tool_calls == 1
        assert report.checked_retrievals == 1
        assert report.checked_decisions == 1
        assert report.skipped_llm_calls == 1  # LLM determinism is optional, never enforced

    def test_tool_output_matches_when_executor_reexecutes_it(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")

        def calculator(inp):
            left, _, right = inp["expr"].partition("+")
            return int(left) + int(right)

        report = ReplayEngine(tool_executors={"calculator": calculator}).replay(path)
        assert report.ok

    def test_retrieval_refs_match_against_a_live_corpus(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")
        corpus = {"doc-1": "Paris is the capital of France.", "doc-2": "France is in Europe."}
        report = ReplayEngine(retrieval_resolver=corpus.get).replay(path)
        assert report.ok

    def test_policy_hash_matches_recorded_policy(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")
        report = ReplayEngine().replay(path)
        assert not any(m.kind == EventKind.DECISION.value for m in report.mismatches)


class TestReplayMismatches:
    def test_tool_output_mismatch_is_detected(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")

        def wrong_calculator(inp):
            return 999999

        report = ReplayEngine(tool_executors={"calculator": wrong_calculator}).replay(path)
        assert not report.ok
        assert any("different output" in m.reason for m in report.mismatches)

    def test_missing_retrieval_ref_is_detected(self, tmp_path, sample_records):
        path = export_audit_bundle(sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip")
        corpus = {"doc-1": "only doc-1 exists"}
        report = ReplayEngine(retrieval_resolver=corpus.get).replay(path)
        assert not report.ok
        assert any("no longer resolves" in m.reason for m in report.mismatches)

    def test_policy_hash_mismatch_is_detected(self, tmp_path):
        tracer = AuditTracer()
        tracer.record(
            EventKind.DECISION,
            {
                "policy": {"rule": "a"},
                "policy_hash": hash_value({"rule": "different"}),
                "decision": "x",
            },
        )
        path = export_audit_bundle(tracer.drain(), SigningPolicy.ed25519(), tmp_path / "b.zip")
        report = ReplayEngine().replay(path)
        assert not report.ok
        assert any("policy_hash" in m.reason for m in report.mismatches)

    def test_llm_output_is_never_required_to_match(self, tmp_path):
        # No executor for llm calls at all — replay must not fail on this basis.
        tracer = AuditTracer()
        tracer.record(
            EventKind.LLM_CALL,
            {"model": "m", "prompt": "p", "response": "any nondeterministic text"},
        )
        path = export_audit_bundle(tracer.drain(), SigningPolicy.ed25519(), tmp_path / "b.zip")
        report = ReplayEngine().replay(path)
        assert report.ok
        assert report.skipped_llm_calls == 1
