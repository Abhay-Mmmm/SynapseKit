"""OpenTelemetry mapping — every AuditRecord maps cleanly to a span/event.

Falls back to a no-op when the ``opentelemetry`` packages aren't
installed, matching the lazy-optional-dependency pattern used by
:mod:`synapsekit.observability.otel`. Works with any OTLP-compatible
backend (Jaeger, Honeycomb, Tempo, ...).
"""

from __future__ import annotations

from typing import Any

from .serializer import canonical_json
from .types import AuditRecord

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _OTEL_AVAILABLE = False


def _span_attribute_value(value: Any) -> Any:
    """OTel span attributes must be a primitive or a homogeneous sequence
    of primitives — anything else (nested dict/list/None) is JSON-encoded
    to a string rather than silently dropped, so no payload data is lost
    on export.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return canonical_json(value).decode("utf-8")


def record_to_span_dict(record: AuditRecord) -> dict[str, Any]:
    """Map one AuditRecord to an OTel-shaped span dict.

    ``trace_id`` is the run id, ``span_id`` the event id, and
    ``parent_span_id`` the parent event id — audit runs are already a
    causal DAG, so this mapping is structure-preserving in both
    directions (a span-shaped view can be traced back to the exact
    record it came from via ``span_id``). The audit bundle remains
    authoritative — OTel is an export layer only, never the source of
    truth (a later verifier can confirm an exported span's
    ``audit.hash``/``audit.payload_hash`` attributes still match the
    signed record with that ``span_id``).
    """
    return {
        "name": f"audit.{record.kind}",
        "trace_id": record.run_id,
        "span_id": record.event_id,
        "parent_span_id": record.parent_id,
        "start_time": record.timestamp.isoformat(),
        "attributes": {
            "audit.kind": record.kind,
            "audit.actor": record.actor,
            "audit.hash": record.hash,
            "audit.payload_hash": record.payload_hash,
            "audit.prev_hash": record.prev_hash,
            "audit.schema_version": record.schema_version,
            "audit.redaction_status": record.redaction_status,
            **{
                f"audit.payload.{k}": _span_attribute_value(v)
                for k, v in record.payload.items()
                if v is not None
            },
        },
    }


class OTelAuditExporter:
    """Exports audit records as OTel spans when the SDK is available.

    Usage::

        exporter = OTelAuditExporter(service_name="my-agent")
        exporter.export_records(records)
    """

    def __init__(self, service_name: str = "synapsekit-audit", tracer_provider: Any = None) -> None:
        self.service_name = service_name
        self._tracer = None
        if _OTEL_AVAILABLE:
            provider = tracer_provider or _otel_trace.get_tracer_provider()
            self._tracer = _otel_trace.get_tracer(service_name, tracer_provider=provider)

    @property
    def available(self) -> bool:
        return self._tracer is not None

    def export_record(self, record: AuditRecord) -> dict[str, Any]:
        span_dict = record_to_span_dict(record)
        if self._tracer is not None:
            with self._tracer.start_as_current_span(span_dict["name"]) as span:
                for key, value in span_dict["attributes"].items():
                    span.set_attribute(key, value)
        return span_dict

    def export_records(self, records: list[AuditRecord]) -> list[dict[str, Any]]:
        return [self.export_record(r) for r in records]


def export_records(exporter: OTelAuditExporter, records: list[AuditRecord]) -> list[dict[str, Any]]:
    return exporter.export_records(records)
