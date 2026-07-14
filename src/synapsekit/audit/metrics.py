"""Prometheus metrics for the audit subsystem.

Mirrors the lazy-optional pattern in
:class:`synapsekit.observability.metrics.PrometheusMetrics` — a separate,
small counter set rather than bolting more labels onto the general LLM
cost/latency metrics, since audit events (signing, export, verification,
redaction) aren't per-model.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry as PromCollectorRegistry
    from prometheus_client import Counter as PromCounter

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PromCollectorRegistry = None  # type: ignore[assignment,misc]
    PromCounter = None  # type: ignore[assignment,misc]
    _PROMETHEUS_AVAILABLE = False


class AuditMetrics:
    """Exposes:

    - ``audit_records_signed_total``
    - ``audit_bundle_exports_total``
    - ``audit_verification_failures_total``
    - ``audit_redactions_total``
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        namespace: str = "synapsekit",
        registry: Any | None = None,
    ) -> None:
        self.enabled = bool(enabled) and _PROMETHEUS_AVAILABLE
        self._namespace = namespace
        self._registry = registry
        if self._registry is None and self.enabled and PromCollectorRegistry is not None:
            self._registry = PromCollectorRegistry()

        self._records_signed: Any | None = None
        self._bundle_exports: Any | None = None
        self._verification_failures: Any | None = None
        self._redactions: Any | None = None

        if self.enabled:
            self._records_signed = PromCounter(
                "audit_records_signed_total",
                "Total audit records covered by a completed batch signature.",
                ["algorithm"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._bundle_exports = PromCounter(
                "audit_bundle_exports_total",
                "Total audit bundles exported.",
                ["result"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._verification_failures = PromCounter(
                "audit_verification_failures_total",
                "Total bundle verification failures, by reason category.",
                ["reason"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._redactions = PromCounter(
                "audit_redactions_total",
                "Total PII redactions applied before hashing.",
                ["label"],
                namespace=self._namespace,
                registry=self._registry,
            )

    def record_signed_batch(self, *, algorithm: str, count: int) -> None:
        if not self.enabled or self._records_signed is None:
            return
        with suppress(Exception):
            self._records_signed.labels(algorithm=str(algorithm)).inc(int(count))

    def record_bundle_export(self, *, ok: bool = True) -> None:
        if not self.enabled or self._bundle_exports is None:
            return
        with suppress(Exception):
            self._bundle_exports.labels(result="success" if ok else "failure").inc()

    def record_verification_failure(self, *, reason: str) -> None:
        if not self.enabled or self._verification_failures is None:
            return
        with suppress(Exception):
            self._verification_failures.labels(reason=str(reason)).inc()

    def record_redaction(self, *, label: str, count: int = 1) -> None:
        if not self.enabled or self._redactions is None:
            return
        with suppress(Exception):
            self._redactions.labels(label=str(label)).inc(int(count))


#: Shared instance used by :mod:`signer`, :mod:`bundle`, :mod:`verifier`,
#: and :mod:`redact` whenever a caller doesn't supply their own —
#: metrics fire automatically without every call site needing to wire
#: one up explicitly. Pass an explicit ``metrics=`` argument anywhere
#: this is used as a default to override it (e.g. a custom registry).
default_metrics = AuditMetrics()
