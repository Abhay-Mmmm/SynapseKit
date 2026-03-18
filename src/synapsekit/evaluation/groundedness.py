"""Groundedness metric — measures if the response is grounded in facts."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .base import MetricResult


class GroundednessMetric:
    """Evaluates whether the response is grounded in the provided context.

    Detects hallucinations by checking if each statement in the answer
    can be traced back to the source documents.

    Usage::
        metric = GroundednessMetric(llm)
        score = await metric.evaluate(
            answer="Python was created in 1991 by Guido.",
            contexts=["Python was created by Guido van Rossum and first released in 1991."],
        )
    """

    name = "groundedness"

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def evaluate(
        self,
        answer: str,
        contexts: list[str],
        question: str = "",
    ) -> MetricResult:
        if not contexts:
            return MetricResult(score=0.0, reason="No contexts to ground against.")

        if not answer.strip():
            return MetricResult(score=1.0, reason="Empty answer.")

        context_text = "\n\n".join(f"[Source {i + 1}]: {c}" for i, c in enumerate(contexts))

        prompt = (
            f"Evaluate whether the following answer is grounded in the provided sources. "
            f"An answer is grounded if every claim can be traced to the sources.\n\n"
            f"Sources:\n{context_text}\n\n"
            f"Answer: {answer}\n\n"
            f"Rate the groundedness from 0 to 10 where:\n"
            f"0 = completely ungrounded / hallucinated\n"
            f"10 = fully grounded in sources\n\n"
            f"Respond with just the number."
        )

        response = await self._llm.generate(prompt)

        # Parse score
        try:
            raw_score = float(response.strip().split()[0])
            score = max(0.0, min(1.0, raw_score / 10.0))
        except (ValueError, IndexError):
            score = 0.5  # default if parsing fails

        return MetricResult(
            score=score,
            reason=f"Groundedness score: {score:.2f}",
            details={"raw_response": response.strip()},
        )
