from __future__ import annotations


class NoReranker:
    def rerank(self, query, chunks, top_k):
        return chunks[:top_k]


class FastEmbedReranker:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model)

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return []
        scores = list(self.model.rerank(query, [c.text for c in chunks], batch_size=8))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]


def make_reranker(cfg: dict):
    if cfg["backend"] == "none":
        return NoReranker()
    return FastEmbedReranker(cfg["model"])
