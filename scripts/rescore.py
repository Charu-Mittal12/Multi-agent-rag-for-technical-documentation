"""Re-score citation metrics of cached answers WITHOUT calling any LLM.

Needed after the citation parser changes: cached rows keep the answer text, so n_cites / cite_valid /
cited_gold can be recomputed by re-running the (deterministic) retrieval and re-parsing the answer.
Verdicts (the LLM judge) are left untouched. Originals are backed up as *.jsonl.bak.

Usage: python scripts/rescore.py [--questions data/eval/techmanualqa.jsonl] [--top-k 5]
"""
import argparse, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from manual_rag.config import load_config
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever, manual_of
from manual_rag.answer_eval import parse_citations, is_abstention

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--top-k", type=int, default=5)
a = ap.parse_args()

cfg = load_config()
qs = {}
for line in open(ROOT / a.questions, encoding="utf-8"):
    if line.strip():
        q = json.loads(line); qs[q["id"]] = q
retriever = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)

for cache in sorted((ROOT / "results").glob("answers_*.jsonl")):
    name, *extra = cache.stem.removeprefix("answers_").split("__")   # e.g. hybrid_rerank__router__strict
    scope = next((x for x in extra if x in ("router", "oracle")), "all")
    rerank = name.endswith("_rerank")
    mode = name.removesuffix("_rerank")
    rows = [json.loads(l) for l in open(cache, encoding="utf-8") if l.strip()]
    changed = 0
    for r in rows:
        q = qs.get(r["id"])
        if q is None:
            continue
        manual = (manual_of(q) if scope == "oracle"
                  else retriever.route(q["question"]) if scope == "router" else None)
        hits = retriever.search(q["question"], mode=mode, rerank=rerank, k=a.top_k, manual=manual)
        gold = {(g["manual"], int(g["page"])) for g in q.get("gold", [])}
        retrieved = {(c.manual, c.page) for c in hits}
        cites = parse_citations(r["answer"])
        new = {"n_cites": len(cites), "cite_valid": bool(cites) and cites <= retrieved,
               "cited_gold": bool(gold & cites) if gold else None,
               "abstained": is_abstention(r["answer"])}
        if any(r.get(k) != v for k, v in new.items()):
            changed += 1
        r.update(new)
    shutil.copy(cache, cache.with_suffix(".jsonl.bak"))
    with open(cache, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{cache.name}: {len(rows)} rows, {changed} changed")
print("Done. Re-run scripts/stats.py (and eval_answers.py to see the new summary).")
