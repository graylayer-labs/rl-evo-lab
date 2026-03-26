from __future__ import annotations

import numpy as np


class EpisodicNovelty:
    """Episodic KNN novelty. Memory resets at the start of each episode."""

    def __init__(self, k: int) -> None:
        self.k = k
        self._memory: list[np.ndarray] = []

    def reset(self) -> None:
        self._memory.clear()

    def score(self, embedding: np.ndarray) -> float:
        """Return mean distance to k nearest neighbours, then add embedding to memory."""
        if len(self._memory) < self.k:
            self._memory.append(embedding.copy())
            return 0.0
        memory = np.stack(self._memory)  # (N, embed_dim)
        dists = np.linalg.norm(memory - embedding, axis=1)
        knn_dists = np.partition(dists, self.k - 1)[: self.k]
        novelty = knn_dists.mean()
        self._memory.append(embedding.copy())
        return float(novelty)
