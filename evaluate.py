"""Retrieval metrics against gold pages. A hit = retrieved chunk is on a gold (manual, page)."""
from __future__ import annotations
import json, math


def load_questions(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _hits(retrieved, gold):
    gold_set = {(g["manual"], g["page"]) for g in gold}
    return [(c.manual, c.page) in gold_set for c in retrieved]


def evaluate(retriever, questions, mode, rerank, ks=(1, 3, 5)):
    kmax = max(max(ks), 10)
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg.update({"mrr": 0.0, "ndcg@10": 0.0})
    answerable = [q for q in questions if q.get("gold")]
    for q in answerable:
        res = retriever.search(q["question"], mode=mode, rerank=rerank, k=kmax)
        # de-duplicate to unique pages, preserving rank
        seen, pages = set(), []
        for c in res:
            if (c.manual, c.page) not in seen:
                seen.add((c.manual, c.page)); pages.append(c)
        h = _hits(pages, q["gold"])
        for k in ks:
            agg[f"recall@{k}"] += float(any(h[:k]))
        agg["mrr"] += next((1.0 / (i + 1) for i, x in enumerate(h) if x), 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(h[:10]) if x)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(q["gold"]), 10)))
        agg["ndcg@10"] += dcg / ideal if ideal else 0.0
    n = max(len(answerable), 1)
    return {k: round(v / n, 3) for k, v in agg.items()} | {"n": len(answerable)}
