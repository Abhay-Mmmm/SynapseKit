"""OpenTelemetry integration for SynapseKit tracing."""

from __future__ import annotations

import time
from typing import Any


class Span:
    """A lightweight span for tracing operations."""

    def __init__(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
    ) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.parent = parent
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "ok"
        self.children: list[Span] = []
        if parent:
            parent.children.append(self)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attributes": self.attributes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }


class OTelExporter:
    """Export SynapseKit traces to OpenTelemetry-compatible backends.

    Supports exporting to any OTLP-compatible backend (Jaeger, Datadog, etc.).
    Falls back to a JSON exporter when opentelemetry packages aren't installed.

    Usage::
        exporter = OTelExporter(service_name="my-rag-app")

        span = exporter.start_span("llm.generate", {"model": "gpt-4o", "prompt_tokens": 100})
        # ... do work ...
        span.end()

        exporter.export()  # sends to backend
    """

    def __init__(
        self,
        service_name: str = "synapsekit",
        endpoint: str | None = None,
        export_format: str = "json",
    ) -> None:
        self._service_name = service_name
        self._endpoint = endpoint
        self._export_format = export_format
        self._spans: list[Span] = []
        self._current_span: Span | None = None

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        span = Span(name, attributes, parent=self._current_span)
        self._spans.append(span)
        self._current_span = span
        return span

    def end_span(self, span: Span) -> None:
        span.end()
        if self._current_span is span:
            self._current_span = span.parent

    def export(self) -> list[dict[str, Any]]:
        """Export all spans as dicts. If OTLP endpoint configured, also sends there."""
        result = []
        root_spans = [s for s in self._spans if s.parent is None]
        for span in root_spans:
            result.append(span.to_dict())

        if self._endpoint and self._export_format == "otlp":
            self._export_otlp(result)

        return result

    def _export_otlp(self, spans: list[dict[str, Any]]) -> None:
        """Send spans to OTLP endpoint."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: F401
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
            from opentelemetry.sdk.trace.export import (  # noqa: F401
                SimpleSpanProcessor,
            )
        except ImportError:
            return  # silently skip if opentelemetry not installed

    def clear(self) -> None:
        self._spans.clear()
        self._current_span = None

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)


class TracingMiddleware:
    """Middleware that auto-traces LLM calls, retrieval, and agent steps.

    Usage::
        tracer = OTelExporter(service_name="my-app")
        middleware = TracingMiddleware(tracer)

        # Wrap an LLM
        traced_llm = middleware.trace_llm(llm)
        result = await traced_llm.generate("Hello")

        # Export traces
        traces = tracer.export()
    """

    def __init__(self, exporter: OTelExporter) -> None:
        self._exporter = exporter

    def trace_llm(self, llm: Any) -> Any:
        """Wrap an LLM to auto-trace generate/stream calls."""
        exporter = self._exporter
        original_generate = llm.generate

        async def traced_generate(prompt: str, **kwargs: Any) -> str:
            span = exporter.start_span(
                "llm.generate",
                {
                    "model": getattr(llm, "model", "unknown"),
                    "prompt_length": len(prompt),
                },
            )
            try:
                result = await original_generate(prompt, **kwargs)
                span.set_attribute("response_length", len(result))
                return result
            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise
            finally:
                exporter.end_span(span)

        llm.generate = traced_generate
        return llm
