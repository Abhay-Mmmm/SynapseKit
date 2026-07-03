"""Small local embeddings used by the personal knowledge mesh."""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


class HashingEmbeddings:
    """Deterministic numpy-only embedding backend for offline mesh indexing.

    The mesh should be useful before optional semantic dependencies are
    installed. This backend gives ``InMemoryVectorStore`` and ``SQLiteVecStore``
    a stable local vector representation without downloading a model.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.model = f"synapsekit-hash-{dimensions}"
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into normalized float32 vectors."""

        rows = [self._embed_one_sync(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed one text into a normalized float32 vector."""

        return self._embed_one_sync(text)

    def _embed_one_sync(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = -1.0 if digest[4] & 1 else 1.0
            vec[slot] += sign
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec
