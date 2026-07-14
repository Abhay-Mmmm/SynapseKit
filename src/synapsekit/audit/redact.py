"""PII redaction — MUST run before hashing.

Redaction is a lossy, irreversible rewrite of a record's payload. It has
to happen *before* :func:`synapsekit.audit.trace.compute_record_hash`
runs, not after: the hash chain commits to whatever payload it was given,
so if you redact after hashing, the recorded hash no longer matches the
(redacted) content anyone can actually inspect, and the chain becomes
unverifiable. The tradeoff this creates is deliberate and permanent —
once a record is hashed on the redacted payload, the pre-redaction
content is gone forever and cannot be recovered even by SynapseKit
itself. Decide your detectors before you start tracing, not after.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from .metrics import AuditMetrics, default_metrics
from .serializer import hash_value


class Detector(Protocol):
    """A PII detector: scans text and returns spans to redact."""

    label: str

    def find(self, text: str) -> list[tuple[int, int]]: ...


class RegexDetector:
    """Detects PII via a compiled regular expression."""

    def __init__(self, label: str, pattern: str) -> None:
        self.label = label
        self._pattern = re.compile(pattern)

    def find(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._pattern.finditer(text)]


def _luhn_checksum_valid(digits: str) -> bool:
    """True if ``digits`` (a numeric string, no separators) passes the Luhn check.

    Real card numbers are Luhn-valid by construction; an arbitrary 13-16
    digit numeric ID (order number, tracking number, phone extension...)
    passes the Luhn check only ~1 in 10 times by chance, so this is an
    effective filter against over-redacting ordinary numeric IDs that
    merely happen to be the right length.
    """
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class LuhnValidatedDetector:
    """Detects credit-card-shaped digit runs that also pass a Luhn checksum.

    The regex alone (13-16 digits, optional space/dash separators) would
    match any numeric ID of that length — order numbers, tracking
    numbers, database IDs. Real card numbers are Luhn-valid by
    construction, so requiring the checksum to pass before redacting
    keeps the detector from over-redacting ordinary numeric IDs while
    still catching real card numbers regardless of separator style.
    """

    def __init__(self, label: str, pattern: str) -> None:
        self.label = label
        self._pattern = re.compile(pattern)

    def find(self, text: str) -> list[tuple[int, int]]:
        spans = []
        for m in self._pattern.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if _luhn_checksum_valid(digits):
                spans.append(m.span())
        return spans


EmailDetector = RegexDetector("EMAIL", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PhoneDetector = RegexDetector(
    "PHONE", r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
SSNDetector = RegexDetector("SSN", r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CreditCardDetector = LuhnValidatedDetector("CREDIT_CARD", r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
IPAddressDetector = RegexDetector(
    "IP_ADDRESS",
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?!\d)",
)

DEFAULT_DETECTORS: list[Detector] = [
    EmailDetector,
    PhoneDetector,
    SSNDetector,
    CreditCardDetector,
    IPAddressDetector,
]


class PIIRedactor:
    """Redacts PII from audit payloads before they enter the hash chain.

    ``ml_detector``, if given, is any callable ``(text) -> list[(start,
    end, label)]`` — a spot for a statistical/NER-based detector (e.g. a
    spaCy or Presidio pipeline) to supplement the regex detectors. It's
    optional and lazily used only when provided, so the base package
    never requires an ML dependency.
    """

    def __init__(
        self,
        detectors: list[Detector] | None = None,
        *,
        ml_detector: Callable[[str], list[tuple[int, int, str]]] | None = None,
        replacement: str = "[REDACTED:{label}]",
        metrics: AuditMetrics | None = None,
    ) -> None:
        self.detectors = detectors if detectors is not None else list(DEFAULT_DETECTORS)
        self.ml_detector = ml_detector
        self.replacement = replacement
        self.redaction_count = 0
        self._metrics = metrics if metrics is not None else default_metrics

    def policy_fingerprint(self) -> str:
        """A stable hash identifying which detectors/config produced a redaction.

        Recorded on each record as ``redaction_policy_hash`` (see
        :class:`~synapsekit.audit.types.AuditRecord`) so a future
        compliance verifier can confirm *which* policy was applied to a
        given record without needing the original unredacted content —
        v2.0 only stores this metadata, it doesn't enforce anything
        based on it yet.
        """
        return hash_value(
            {
                "detectors": sorted(d.label for d in self.detectors),
                "ml_detector": self.ml_detector is not None,
                "replacement": self.replacement,
            }
        )

    def redact_text(self, text: str) -> str:
        spans: list[tuple[int, int, str]] = []
        for detector in self.detectors:
            for start, end in detector.find(text):
                spans.append((start, end, detector.label))
        if self.ml_detector is not None:
            spans.extend(self.ml_detector(text))
        if not spans:
            return text

        # Resolve overlaps by taking the earliest, longest span first.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str]] = []
        cursor = -1
        for start, end, label in spans:
            if start < cursor:
                continue
            merged.append((start, end, label))
            cursor = end

        out: list[str] = []
        pos = 0
        for start, end, label in merged:
            out.append(text[pos:start])
            out.append(self.replacement.format(label=label))
            pos = end
            self.redaction_count += 1
            self._metrics.record_redaction(label=label)
        out.append(text[pos:])
        return "".join(out)

    def redact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Recursively redact all string values in a payload dict."""
        return self._redact_value(payload)  # type: ignore[no-any-return]

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(v) for v in value)
        return value
