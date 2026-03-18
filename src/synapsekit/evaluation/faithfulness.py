"""Faithfulness metric — measures if the answer is faithful to the context."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .base import MetricResult


class FaithfulnessMetric:
    """Evaluates whether an LLM answer is faithful to the retrieved context.

    Uses an LLM judge to extract claims from the answer and verify each
    against the source documents. Returns a score from 0.0 to 1.0.

    Usage::
        metric = FaithfulnessMetric(llm)
        score = await metric.evaluate(
            question="What is Python?",
            answer="Python is a programming language created by Guido.",
            contexts=["Python is a programming language created by Guido van Rossum."],
        )
        # score.score → 1.0 (all claims supported)
    """

    name = "faithfulness"

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> MetricResult:
        # Step 1: Extract claims from the answer
        claims_prompt = (
            f"Extract all factual claims from this answer as a numbered list.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"List each distinct factual claim on its own line, numbered (1. 2. 3. etc).\n"
            f"If there are no factual claims, write: NONE"
        )
        claims_response = await self._llm.generate(claims_prompt)

        if "NONE" in claims_response.upper():
            return MetricResult(
                score=1.0,
                reason="No factual claims to verify.",
                details={"claims": [], "supported": []},
            )

        claims = [
            line.strip()
            for line in claims_response.strip().split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

        if not claims:
            return MetricResult(
                score=1.0,
                reason="No claims extracted.",
                details={"claims": [], "supported": []},
            )

        # Step 2: Check each claim against contexts
        context_text = "\n\n".join(f"[Source {i + 1}]: {c}" for i, c in enumerate(contexts))
        supported = []

        for claim in claims:
            check_prompt = (
                f"Is the following claim supported by the provided sources? "
                f"Answer YES or NO only.\n\n"
                f"Sources:\n{context_text}\n\n"
                f"Claim: {claim}"
            )
            response = await self._llm.generate(check_prompt)
            is_supported = "YES" in response.upper()
            supported.append(is_supported)

        score = sum(supported) / len(supported) if supported else 1.0

        return MetricResult(
            score=score,
            reason=f"{sum(supported)}/{len(supported)} claims supported by context.",
            details={"claims": claims, "supported": supported},
        )
