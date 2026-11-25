"""Ablation: bm25 / dense / hybrid, each with and without reranker, optionally per retrieval scope.
  scope all    = search all manuals (default)
  scope router = LLM-free router picks the manual, then search inside it
  scope oracle = search only the question's true manual (upper bound)
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever
from manual_rag.evaluate import load_questions, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None, help="question file (default: paths.eval_file in config.yaml)")
ap.add_argument("--scopes", default="all", help="comma list from: all, router, oracle")
ap.add_argument("--modes", default="bm25,dense,hybrid")
args = ap.parse_args()
cfg = load_config()
qs = load_questions(ROOT / (args.questions or cfg["paths"]["eval_file"]))
r = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)
cols = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"]
print(f"{'config':<30}" + "".join(f"{m:>11}" for m in cols))
n = 0
strict_rows = []
for scope in [s.strip() for s in args.scopes.split(",") if s.strip()]:
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for rerank in [False, True]:
            s = evaluate(r, qs, mode, rerank, scope=scope)
            name = f"{mode}{'+rerank' if rerank else ''} [{scope}]"
            extra = f"   router_acc={s['router_acc']}" if "router_acc" in s else ""
            print(f"{name:<30}" + "".join(f"{s[m]:>11}" for m in cols) + extra)
            n = s["n"]
            if "s_recall@5" in s:
                strict_rows.append((name, s))
print(f"(n={n} answerable questions)")

if strict_rows:
    sc = ["s_recall@1", "s_recall@3", "s_recall@5", "s_mrr", "s_ndcg@10", "prec@5", "gap@5"]
    print("\nChunk level (strict): a retrieved chunk counts only if it is on the gold page AND contains the answer text.")
    print(f"{'config':<30}" + "".join(f"{m:>12}" for m in sc))
    for name, s in strict_rows:
        print(f"{name:<30}" + "".join(f"{s[m]:>12}" for m in sc))
    pc = ["page_prec@1", "page_prec@3", "page_prec@5", "prec@1", "prec@3", "prec@5", "ideal_prec@1", "ideal_prec@3", "ideal_prec@5"]
    print("\nPrecision@k. page_prec = share of the top-k unique pages that are gold pages; prec = share of the top-k chunks that are "
          "relevant (strict); ideal_prec = the best a perfect retriever could score given how many relevant chunks exist.")
    print(f"{'config':<30}" + "".join(f"{m:>13}" for m in pc))
    for name, s in strict_rows:
        print(f"{name:<30}" + "".join(f"{s[m]:>13}" for m in pc))
    s0 = strict_rows[0][1]
    print(f"(scored on {s0['s_n']} questions; {s0['unlabelled']} skipped because the evidence text was not found "
          f"in any chunk of the gold page)")
    print("gap@5 = page-level recall@5 minus strict recall@5 on the same questions: how much the page-level number flatters.")
