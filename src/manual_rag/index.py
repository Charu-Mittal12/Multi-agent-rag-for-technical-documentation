"""Hybrid index: BM25 (exact codes / part numbers) + dense vectors (paraphrases), fused with RRF."""
from __future__ import annotations
import json, pickle, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from .parse import Chunk


def tokenize(text: str) -> list[str]:
    # keep error codes / part numbers like "E-203" or "ESP32-C3" as whole tokens plus their parts
    text = text.lower()
    whole = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text)
    parts = re.findall(r"[a-z0-9]+", text)
    return whole + [p for p in parts if p not in whole]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    # --- individual retrievers (return ranked chunk indices) ---
    def manual_mask(self, manual: str) -> np.ndarray:
        """Boolean mask selecting the chunks of one manual (used to restrict retrieval)."""
        return np.array([c.manual == manual for c in self.chunks])

    def bm25_search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        if mask is not None:
            scores = np.where(mask, scores, -1.0)
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]

    def dense_search(self, qvec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[int]:
        qn = qvec / (np.linalg.norm(qvec) or 1.0)
        vn = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        sims = vn @ qn
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        return [int(i) for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])]

    # --- persistence ---
    def save(self, directory: str | Path):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, directory: str | Path):
        d = Path(directory)
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(d / "vectors.npy"))
