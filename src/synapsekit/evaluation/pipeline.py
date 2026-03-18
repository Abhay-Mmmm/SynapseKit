"""Evaluation pipeline — run multiple metrics on RAG outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import MetricResult


@dataclass
class EvaluationResult:
    """Aggregated evaluation result."""

    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, MetricResult] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def __repr__(self) -> str:
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        return f"EvaluationResult(mean={self.mean_score:.2f}, {scores_str})"


class EvaluationPipeline:
    """Run multiple evaluation metrics on RAG outputs.

    Usage::
        pipeline = EvaluationPipeline(metrics=[
            FaithfulnessMetric(llm),
            RelevancyMetric(llm),
            GroundednessMetric(llm),
        ])
        result = await pipeline.evaluate(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        print(result.mean_score)
        print(result.scores)
    """

    def __init__(self, metrics: list[Any]) -> None:
        self._metrics = metrics

    async def evaluate(
        self,
        question: str = "",
        answer: str = "",
        contexts: list[str] | None = None,
    ) -> EvaluationResult:
        scores: dict[str, float] = {}
        details: dict[str, MetricResult] = {}

        for metric in self._metrics:
            result = await metric.evaluate(
                question=question,
                answer=answer,
                contexts=contexts or [],
            )
            scores[metric.name] = result.score
            details[metric.name] = result

        return EvaluationResult(scores=scores, details=details)

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of samples."""
        results = []
        for sample in samples:
            result = await self.evaluate(**sample)
            results.append(result)
        return results
