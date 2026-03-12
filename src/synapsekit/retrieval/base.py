from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract base class for all vector store backends."""

    @abstractmethod
    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]: ...

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Maximal Marginal Relevance search.

        Balances relevance with diversity by penalizing documents similar to
        already-selected ones.

        Args:
            lambda_mult: 0 = max diversity, 1 = max relevance.
            fetch_k: Number of initial candidates to consider.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support search_mmr()")

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support load()")
