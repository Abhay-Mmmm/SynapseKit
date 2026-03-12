from __future__ import annotations

import numpy as np

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class InMemoryVectorStore(VectorStore):
    """
    Numpy-backed in-memory vector store.
    Supports cosine similarity search, save/load via .npz.
    """

    def __init__(self, embedding_backend: SynapsekitEmbeddings) -> None:
        self._embeddings = embedding_backend
        self._vectors: np.ndarray | None = None  # (N, D)
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)

        if self._vectors is None:
            self._vectors = vecs
        else:
            self._vectors = np.concatenate([self._vectors, vecs], axis=0)

        self._texts.extend(texts)
        self._metadata.extend(meta)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Returns top_k results sorted by cosine similarity (desc).

        Args:
            metadata_filter: If provided, only include documents whose metadata
                contains all the specified key-value pairs.
        """
        if self._vectors is None or len(self._texts) == 0:
            return []

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,) cosine sim (vecs are L2-normalised)

        # Apply metadata filter — build candidate indices
        if metadata_filter:
            candidates = [
                i
                for i in range(len(self._texts))
                if all(self._metadata[i].get(k) == v for k, v in metadata_filter.items())
            ]
            if not candidates:
                return []
            candidate_scores = scores[candidates]
            k = min(top_k, len(candidates))
            local_top = np.argpartition(candidate_scores, -k)[-k:]
            local_top = local_top[np.argsort(candidate_scores[local_top])[::-1]]
            top_indices = [candidates[j] for j in local_top]
        else:
            k = min(top_k, len(self._texts))
            _top = np.argpartition(scores, -k)[-k:]
            top_indices = list(_top[np.argsort(scores[_top])[::-1]])

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_indices
        ]

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Maximal Marginal Relevance search.

        Greedily selects documents that maximize:
        ``lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))``
        """
        if self._vectors is None or len(self._texts) == 0:
            return []

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,) cosine sim

        # Build candidate pool
        if metadata_filter:
            candidates = [
                i
                for i in range(len(self._texts))
                if all(self._metadata[i].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            candidates = list(range(len(self._texts)))

        if not candidates:
            return []

        # Take top fetch_k by relevance
        candidate_scores = [(i, float(scores[i])) for i in candidates]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        pool = candidate_scores[: min(fetch_k, len(candidate_scores))]

        # Greedy MMR selection
        selected: list[int] = []

        for _ in range(min(top_k, len(pool))):
            best_idx = -1
            best_score = float("-inf")

            for idx, rel_score in pool:
                if idx in selected:
                    continue

                # Max similarity to already-selected docs
                if selected:
                    sim_to_selected = max(
                        float(self._vectors[idx] @ self._vectors[s]) for s in selected
                    )
                else:
                    sim_to_selected = 0.0

                mmr_score = lambda_mult * rel_score - (1 - lambda_mult) * sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break
            selected.append(best_idx)

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in selected
        ]

    def save(self, path: str) -> None:
        """Persist vectors, texts, and metadata to a .npz file."""
        if self._vectors is None:
            raise ValueError("Nothing to save — store is empty.")
        import json

        np.savez(
            path,
            vectors=self._vectors,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array([json.dumps(m) for m in self._metadata], dtype=object),
        )

    def load(self, path: str) -> None:
        """Load vectors, texts, and metadata from a .npz file."""
        import json

        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"].astype(np.float32)
        self._texts = list(data["texts"])
        self._metadata = [json.loads(s) for s in data["metadata"]]

    def __len__(self) -> int:
        return len(self._texts)
