"""Retrieval metrics against gold pages (lenient) and against answer-bearing chunks (strict).

Page level : a retrieved chunk is a hit if it lies on a gold (manual, page).  Lenient - a chunk from the right
             page that does NOT contain the answer still counts.
Chunk level: a retrieved chunk is relevant only if it lies on a gold page AND contains the evidence text
             (exact match after normalisation, or >= 70% of the evidence words). Reported when questions carry
             an `evidence` field (scripts/prepare_techmanualqa.py writes it).
"""
from __future__ import annotations
import json, math, re


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _hits(retrieved, gold):
    gold_set = {(g["manual"], g["page"]) for g in gold}
    return [(c.manual, c.page) in gold_set for c in retrieved]


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def chunk_relevant(chunk, q: dict) -> bool | None:
    """True if the chunk is on a gold page and contains the evidence; None if the question has no evidence."""
    ev = q.get("evidence") or q.get("answer")
    if not ev or not q.get("gold"):
        return None
    if (chunk.manual, chunk.page) not in {(g["manual"], g["page"]) for g in q["gold"]}:
        return False
    key = _squash(ev)
    if key and key in _squash(chunk.text):
        return True
    w = _words(ev)
    if len(w) >= 3:
        have = set(_words(chunk.text))
        return sum(x in have for x in w) / len(w) >= 0.7
    return False


def evaluate(retriever, questions, mode, rerank, ks=(1, 3, 5), scope="all"):
    """scope: 'all' = search every manual; 'router' = LLM-free router picks the manual first;
    'oracle' = search only the question's true manual (upper bound)."""
    from .retrieve import manual_of
    kmax = max(max(ks), 10)
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg.update({"mrr": 0.0, "ndcg@10": 0.0})
    strict = {"s_recall@1": 0.0, "s_recall@3": 0.0, "s_recall@5": 0.0, "s_mrr": 0.0, "s_ndcg@10": 0.0,
              "prec@1": 0.0, "prec@3": 0.0, "prec@5": 0.0, "ideal_prec@1": 0.0,
              "ideal_prec@3": 0.0, "ideal_prec@5": 0.0, "page_recall@5_same_qs": 0.0}
    page_prec = {1: 0.0, 3: 0.0, 5: 0.0}
    s_n = unlabelled = 0
    answerable = [q for q in questions if q.get("gold")]
    routed_ok = 0
    for q in answerable:
        manual = None
        if scope == "oracle":
            manual = manual_of(q)
        elif scope == "router":
            manual = retriever.route(q["question"])
            routed_ok += manual == manual_of(q)
        res = retriever.search(q["question"], mode=mode, rerank=rerank, k=kmax, manual=manual)
        # ---- page level: de-duplicate to unique pages, preserving rank
        seen, pages = set(), []
        for c in res:
            if (c.manual, c.page) not in seen:
                seen.add((c.manual, c.page)); pages.append(c)
        h = _hits(pages, q["gold"])
        for k in ks:
            agg[f"recall@{k}"] += float(any(h[:k]))
        for k in page_prec:                      # page precision@k: share of the top-k UNIQUE pages that are gold pages
            page_prec[k] += sum(h[:k]) / k
        agg["mrr"] += next((1.0 / (i + 1) for i, x in enumerate(h) if x), 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(h[:10]) if x)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(q["gold"]), 10)))
        agg["ndcg@10"] += dcg / ideal if ideal else 0.0
        # ---- chunk level (strict)
        if q.get("evidence") or q.get("answer"):
            total_rel = sum(bool(chunk_relevant(c, q)) for c in retriever.index.chunks
                            if (c.manual, c.page) in {(g["manual"], g["page"]) for g in q["gold"]})
            if total_rel == 0:
                unlabelled += 1          # evidence text not found in any chunk of the gold page: cannot score
                continue
            rel = [bool(chunk_relevant(c, q)) for c in res[:10]]
            s_n += 1
            for k in (1, 3, 5):
                strict[f"s_recall@{k}"] += float(any(rel[:k]))
            strict["s_mrr"] += next((1.0 / (i + 1) for i, x in enumerate(rel) if x), 0.0)
            sd = sum(1.0 / math.log2(i + 2) for i, x in enumerate(rel) if x)
            si = sum(1.0 / math.log2(i + 2) for i in range(min(total_rel, 10)))
            strict["s_ndcg@10"] += sd / si if si else 0.0
            for k in (1, 3, 5):                  # precision@k, and the best achievable given how many relevant chunks exist
                strict[f"prec@{k}"] += sum(rel[:k]) / k
                strict[f"ideal_prec@{k}"] += min(k, total_rel) / k
            strict["page_recall@5_same_qs"] += float(any(h[:5]))
    n = max(len(answerable), 1)
    out = {k: round(v / n, 3) for k, v in agg.items()} | {"n": len(answerable)}
    out |= {f"page_prec@{k}": round(v / n, 3) for k, v in page_prec.items()}
    if s_n:
        out |= {k: round(v / s_n, 3) for k, v in strict.items()} | {"s_n": s_n, "unlabelled": unlabelled}
        out["gap@5"] = round(out["page_recall@5_same_qs"] - out["s_recall@5"], 3)
    if scope == "router":
        out["router_acc"] = round(routed_ok / n, 3)
    return out
