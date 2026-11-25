from __future__ import annotations
from .index import HybridIndex, rrf
from .embed import make_embedder
from .rerank import make_reranker


class Retriever:
    """mode: 'bm25' | 'dense' | 'hybrid'; rerank: bool. Both are ablation switches."""
    def __init__(self, index: HybridIndex, cfg: dict):
        self.index, self.cfg = index, cfg
        self.embedder = make_embedder(cfg["embeddings"])
        self.reranker = make_reranker(cfg["reranker"])

    def search(self, query: str, mode: str = "hybrid", rerank: bool = True, k: int | None = None):
        r = self.cfg["retrieval"]
        k = k or r["final_k"]
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self.index.bm25_search(query, r["bm25_k"]))
        if mode in ("dense", "hybrid"):
            rankings.append(self.index.dense_search(self.embedder.embed_query(query), r["dense_k"]))
        fused = rrf(rankings, r["rrf_k"])
        n = self.cfg["reranker"]["top_n"] if rerank else k
        candidates = [self.index.chunks[i] for i, _ in fused[:n]]
        return self.reranker.rerank(query, candidates, k) if rerank else candidates[:k]
