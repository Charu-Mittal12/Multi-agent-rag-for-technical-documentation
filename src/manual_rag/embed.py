from __future__ import annotations
import hashlib, re
import numpy as np


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. Only for offline tests / smoke runs."""
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts): return np.stack([self._vec(t) for t in texts])
    def embed_query(self, text): return self._vec(text)


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model)

    def embed_documents(self, texts):
        return np.array(list(self.model.embed(texts, batch_size=16)), dtype=np.float32)

    def embed_query(self, text):
        return np.array(list(self.model.query_embed(text))[0], dtype=np.float32)


def make_embedder(cfg: dict):
    if cfg["backend"] == "hash":
        return HashEmbedder()
    return FastEmbedder(cfg["model"])
