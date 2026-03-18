"""Base classes for evaluation metrics."""

from __future__ import annotations

from typing import Any


class MetricResult:
    """Result of a metric evaluation."""

    def __init__(
        self, score: float, reason: str = "", details: dict[str, Any] | None = None
    ) -> None:
        self.score = score
        self.reason = reason
        self.details = details or {}

    def __repr__(self) -> str:
        return f"MetricResult(score={self.score:.2f}, reason={self.reason!r})"
