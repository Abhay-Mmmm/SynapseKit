"""Relevancy metric — measures if retrieved documents are relevant to the query."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .base import MetricResult


class RelevancyMetric:
    """Evaluates whether retrieved documents are relevant to the query.

    Usage::
        metric = RelevancyMetric(llm)
        score = await metric.evaluate(
            question="What is Python?",
            contexts=["Python is a programming language.", "Java is also a language."],
        )
    """

    name = "relevancy"

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def evaluate(
        self,
        question: str,
        contexts: list[str],
        answer: str = "",
    ) -> MetricResult:
        if not contexts:
            return MetricResult(score=0.0, reason="No contexts provided.")

        relevant_count = 0
        relevancy_scores: list[bool] = []

        for _i, context in enumerate(contexts):
            prompt = (
                f"Is the following document relevant to answering the question? "
                f"Answer YES or NO only.\n\n"
                f"Question: {question}\n\n"
                f"Document: {context}"
            )
            response = await self._llm.generate(prompt)
            is_relevant = "YES" in response.upper()
            relevancy_scores.append(is_relevant)
            if is_relevant:
                relevant_count += 1

        score = relevant_count / len(contexts)

        return MetricResult(
            score=score,
            reason=f"{relevant_count}/{len(contexts)} documents relevant.",
            details={"relevancy_scores": relevancy_scores},
        )
