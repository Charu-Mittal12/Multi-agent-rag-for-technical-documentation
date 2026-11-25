from __future__ import annotations
from collections import defaultdict
from .index import HybridIndex, rrf
from .embed import make_embedder
from .rerank import make_reranker


def manual_of(question: dict) -> str:
    """Manual a benchmark question belongs to (ids look like 'bosch_oven_Q003')."""
    return question["id"].rsplit("_Q", 1)[0]


class Retriever:
    """mode: 'bm25' | 'dense' | 'hybrid'; rerank: bool. Both are ablation switches.
    manual: optionally restrict the search to one manual (oracle, or the router's choice)."""
    def __init__(self, index: HybridIndex, cfg: dict):
        self.index, self.cfg = index, cfg
        self.embedder = make_embedder(cfg["embeddings"])
        self.reranker = make_reranker(cfg["reranker"])
        self._masks: dict[str, object] = {}

    def _mask(self, manual: str | None):
        if manual is None:
            return None
        if manual not in self._masks:
            self._masks[manual] = self.index.manual_mask(manual)
        return self._masks[manual]

    def _fused(self, query: str, mode: str, mask=None):
        r = self.cfg["retrieval"]
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self.index.bm25_search(query, r["bm25_k"], mask))
        if mode in ("dense", "hybrid"):
            rankings.append(self.index.dense_search(self.embedder.embed_query(query), r["dense_k"], mask))
        return rrf(rankings, r["rrf_k"])

    def route(self, query: str, top: int = 10) -> str | None:
        """Cheap, LLM-free manual router: hybrid-search all manuals and let the top hits vote
        (weighted by their fused score) for the manual they come from."""
        votes: dict[str, float] = defaultdict(float)
        for i, score in self._fused(query, "hybrid")[:top]:
            votes[self.index.chunks[i].manual] += score
        return max(votes, key=votes.get) if votes else None

    def search(self, query: str, mode: str = "hybrid", rerank: bool = True, k: int | None = None,
               manual: str | None = None):
        r = self.cfg["retrieval"]
        k = k or r["final_k"]
        fused = self._fused(query, mode, self._mask(manual))
        n = self.cfg["reranker"]["top_n"] if rerank else k
        candidates = [self.index.chunks[i] for i, _ in fused[:n]]
        return self.reranker.rerank(query, candidates, k) if rerank else candidates[:k]
