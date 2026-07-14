"""AuditMetrics must actually fire as a side effect of real operations —
not just exist as an importable, disconnected class.
"""

from __future__ import annotations

import pytest

from synapsekit.audit import (
    AuditMetrics,
    AuditTracer,
    EventKind,
    SigningPolicy,
    export_audit_bundle,
)
from synapsekit.audit.redact import PIIRedactor
from synapsekit.audit.verifier import verify

prometheus_client = pytest.importorskip("prometheus_client")


def _counter_total(metrics: AuditMetrics, attr: str) -> float:
    metric = getattr(metrics, attr)
    assert metric is not None, f"{attr} metric was never initialized"
    total = 0.0
    for sample_family in metric.collect():
        for sample in sample_family.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return total


class TestSigningMetrics:
    def test_export_audit_bundle_increments_records_signed(self, tmp_path, sample_records):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)
        policy = SigningPolicy.ed25519(metrics=metrics)

        export_audit_bundle(sample_records, policy, tmp_path / "b.zip", metrics=metrics)

        assert _counter_total(metrics, "_records_signed") == len(sample_records)

    def test_sign_batch_directly_increments_the_counter_regardless_of_caller(self):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)
        policy = SigningPolicy.ed25519(metrics=metrics)

        policy.sign_batch("ab" * 32, start_index=0, end_index=4)

        assert _counter_total(metrics, "_records_signed") == 5


class TestBundleExportMetrics:
    def test_successful_export_increments_bundle_exports_total(self, tmp_path, sample_records):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)

        export_audit_bundle(
            sample_records, SigningPolicy.ed25519(), tmp_path / "b.zip", metrics=metrics
        )

        assert _counter_total(metrics, "_bundle_exports") == 1


class TestVerificationMetrics:
    def test_failed_verification_increments_verification_failures_total(
        self, tmp_path, bundle_path
    ):
        from .conftest import (
            dump_trace_lines,
            load_trace_lines,
            read_zip_entries,
            write_zip_entries,
        )

        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["hash"] = "0" * 64
        dump_trace_lines(entries, records)
        tampered = tmp_path / "tampered.zip"
        write_zip_entries(tampered, entries)

        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)

        result = verify(tampered, metrics=metrics)

        assert not result.ok
        assert _counter_total(metrics, "_verification_failures") >= 1

    def test_successful_verification_does_not_increment_failures(self, bundle_path):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)

        result = verify(bundle_path, metrics=metrics)

        assert result.ok
        assert _counter_total(metrics, "_verification_failures") == 0


class TestRedactionMetrics:
    def test_redaction_increments_redactions_total(self):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)
        redactor = PIIRedactor(metrics=metrics)

        redactor.redact_text("email me at a@b.com or c@d.com")

        assert _counter_total(metrics, "_redactions") == 2

    def test_tracer_level_redaction_also_feeds_the_metric(self):
        registry = prometheus_client.CollectorRegistry()
        metrics = AuditMetrics(registry=registry)
        redactor = PIIRedactor(metrics=metrics)
        tracer = AuditTracer(redactor=redactor)

        tracer.record(EventKind.TOOL_CALL, {"note": "contact a@b.com"})

        assert _counter_total(metrics, "_redactions") == 1
