"""Retrieval failure analysis: list answerable questions whose gold page is NOT in the top-k.

Writes results/retrieval_failures.csv with the question, gold page(s), what was retrieved instead,
the router's pick, and the gold page's rank (if it appears at all within the top 30).
Read these rows and tag each failure (wrong manual, table/figure, vague question, gold page doubtful...).

  python scripts/failures.py [--questions data/eval/techmanualqa.jsonl] [--mode hybrid] [--no-rerank] [--top-k 5]
"""
import argparse, csv, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever, manual_of
from manual_rag.evaluate import load_questions

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--mode", default="hybrid")
ap.add_argument("--no-rerank", action="store_true")
ap.add_argument("--top-k", type=int, default=5)
a = ap.parse_args()

cfg = load_config()
qs = [q for q in load_questions(ROOT / a.questions) if q.get("gold")]
r = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)
rows, by_cat, by_manual, wrong_manual = [], Counter(), Counter(), 0
for q in qs:
    res = r.search(q["question"], mode=a.mode, rerank=not a.no_rerank, k=30)
    pages = []
    for c in res:
        if (c.manual, c.page) not in pages:
            pages.append((c.manual, c.page))
    gold = {(g["manual"], int(g["page"])) for g in q["gold"]}
    rank = next((i + 1 for i, p in enumerate(pages) if p in gold), None)
    if rank is not None and rank <= a.top_k:
        continue
    true_m = manual_of(q)
    top = pages[:a.top_k]
    wm = sum(m != true_m for m, _ in top)
    wrong_manual += wm == len(top)
    by_cat[q.get("category", "?")] += 1; by_manual[true_m] += 1
    rows.append({"id": q["id"], "category": q.get("category", "?"), "question": q["question"],
                 "gold": "; ".join(f"{m} p.{p}" for m, p in sorted(gold)), "gold_rank_in_top30": rank or "",
                 "top_retrieved": "; ".join(f"{m} p.{p}" for m, p in top),
                 "all_top_from_wrong_manual": wm == len(top), "router_pick": r.route(q["question"]),
                 "reference_answer": q.get("answer", "")[:300]})
out = ROOT / "results" / "retrieval_failures.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"]); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} of {len(qs)} answerable questions have no gold page in the top {a.top_k} "
      f"({a.mode}{'' if a.no_rerank else '+rerank'}).")
print(f"  of these, {sum(1 for x in rows if x['gold_rank_in_top30'])} have the gold page at rank {a.top_k + 1}-30; "
      f"{sum(1 for x in rows if not x['gold_rank_in_top30'])} are not in the top 30 at all.")
print(f"  {wrong_manual} had ALL top-{a.top_k} results from the wrong manual.")
print("  by category:", dict(by_cat)); print("  by manual:  ", dict(by_manual))
print(f"Saved {out.relative_to(ROOT)} - read it row by row.")
