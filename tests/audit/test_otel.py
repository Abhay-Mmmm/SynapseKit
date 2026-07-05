"""OTel mapping: every AuditRecord maps to a span/event; a record dropped
before export means its span is simply missing — callers must reconcile
record counts against exported spans themselves.
"""

from __future__ import annotations

from synapsekit.audit import AuditTracer, EventKind
from synapsekit.audit.otel import _OTEL_AVAILABLE, OTelAuditExporter, record_to_span_dict


class TestRecordToSpanDict:
    def test_maps_run_id_and_event_id_to_trace_and_span_id(self):
        tracer = AuditTracer(run_id="run-42")
        rec = tracer.record(EventKind.TOOL_CALL, {"tool": "calc"})
        span = record_to_span_dict(rec)
        assert span["trace_id"] == "run-42"
        assert span["span_id"] == rec.event_id
        assert span["name"] == "audit.TOOL_CALL"

    def test_parent_child_relationship_is_preserved(self):
        tracer = AuditTracer()
        parent = tracer.record(EventKind.DECISION, {"x": 1})
        child = tracer.record(EventKind.TOOL_CALL, {"x": 2}, parent_id=parent.event_id)
        span = record_to_span_dict(child)
        assert span["parent_span_id"] == parent.event_id


class TestOTelAuditExporter:
    def test_exporter_reflects_whether_the_opentelemetry_sdk_is_installed(self):
        # Degrades gracefully either way: `available` mirrors whether the
        # optional `opentelemetry` package is importable in this env,
        # and construction never raises regardless of which it is.
        exporter = OTelAuditExporter(service_name="test-service")
        assert exporter.available is _OTEL_AVAILABLE

    def test_export_records_returns_span_dicts_for_every_record(self):
        tracer = AuditTracer()
        tracer.record(EventKind.LLM_CALL, {"a": 1})
        tracer.record(EventKind.TOOL_CALL, {"b": 2})
        records = tracer.drain()
        exporter = OTelAuditExporter()
        spans = exporter.export_records(records)
        assert len(spans) == len(records) == 2

    def test_a_record_missing_from_the_export_set_has_no_span(self):
        """ "Missing OTEL span" case: if a record never reaches export_records
        (e.g. dropped by a filter upstream), no span is produced for it —
        exporting is not implicitly complete just because *some* spans exist.
        """
        tracer = AuditTracer()
        kept = tracer.record(EventKind.LLM_CALL, {"a": 1})
        dropped = tracer.record(EventKind.TOOL_CALL, {"b": 2})
        exporter = OTelAuditExporter()

        spans = exporter.export_records([kept])  # simulate `dropped` never being forwarded

        span_ids = {s["span_id"] for s in spans}
        assert kept.event_id in span_ids
        assert dropped.event_id not in span_ids
