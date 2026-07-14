"""Hybrid Search Retriever: combines BM25 keyword matching with vector similarity."""

from __future__ import annotations

import heapq
import logging

from rank_bm25 import BM25Okapi

from .retriever import Retriever

logger = logging.getLogger(__name__)


class HybridSearchRetriever:
    """Combines BM25 keyword matching with vector similarity using RRF fusion.

    Usage::

        hybrid = HybridSearchRetriever(retriever=retriever)
        hybrid.add_documents(["doc1 text", "doc2 text", ...])
        results = await hybrid.retrieve("search query", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._retriever = retriever
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._rrf_k = rrf_k
        self._documents: list[str] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, texts: list[str]) -> None:
        """Add texts to the BM25 index, accumulating with any prior documents.

        Incremental calls extend the index rather than replacing it, so previously
        added documents remain searchable. BM25Okapi has no incremental update, so
        the index is rebuilt over the full accumulated corpus.
        """
        self._documents.extend(texts)
        tokenized = [doc.lower().split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve using RRF fusion of BM25 + vector scores.

        If the vector store fails, fall back to BM25-only results rather than
        failing the whole query.
        """
        # Vector retrieval — tolerate a downed vector store by degrading to BM25.
        vector_results: list[str] = []
        try:
            vector_results = await self._retriever.retrieve(
                query, top_k=top_k * 2, metadata_filter=metadata_filter
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed (%s); falling back to BM25-only results.", exc)

        # BM25 scoring — nlargest avoids a full O(n log n) sort of every score.
        bm25_ranked: list[str] = []
        if self._bm25 is not None and self._documents:
            scores = self._bm25.get_scores(query.lower().split())
            top_pairs = heapq.nlargest(top_k * 2, enumerate(scores), key=lambda x: x[1])
            bm25_ranked = [self._documents[i] for i, _ in top_pairs]

        # RRF fusion
        fused_scores: dict[str, float] = {}

        for rank, doc in enumerate(vector_results):
            fused_scores[doc] = fused_scores.get(doc, 0.0) + self._vector_weight / (
                self._rrf_k + rank + 1
            )

        for rank, doc in enumerate(bm25_ranked):
            fused_scores[doc] = fused_scores.get(doc, 0.0) + self._bm25_weight / (
                self._rrf_k + rank + 1
            )

        sorted_docs = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        return sorted_docs[:top_k]
