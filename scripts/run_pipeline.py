"""Run the COMPLETE pipeline (retriever -> answer agent -> verifier agent) on the eval set and score it.

  python scripts/run_pipeline.py --provider openai --model openai/gpt-oss-120b \
      --verifier-provider openai --verifier-model qwen/qwen3.8-27b \
      --judge-provider openai --judge-model qwen/qwen3.8-27b --sleep 3 [--shard 1/3] [--limit N]

Per question it stores the final answer, the verifier verdict (faithfulness, answer relevance), whether the
verifier vetoed, and the usual answer metrics (judge correctness, retrieval_hit, cite_valid, cited_gold,
false_abstain / hallucination). Cached in results/pipeline_<tag>.jsonl, so reruns resume after quota resets.
Reports: system-level table + an ablation of the verifier (final vs draft answers, same run).
"""
import argparse, csv, json, sys, time, zlib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever
from manual_rag.evaluate import load_questions
from manual_rag.agents import Assistant
from manual_rag.answer_eval import evaluate_question, summarize, is_abstention

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--provider", required=True); ap.add_argument("--model", default="")
ap.add_argument("--verifier-provider", default="none"); ap.add_argument("--verifier-model", default="")
ap.add_argument("--judge-provider", default=None); ap.add_argument("--judge-model", default=None)
ap.add_argument("--top-k", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="")
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--tag", default="full")
a = ap.parse_args()

cfg = load_config()
qs = load_questions(ROOT / a.questions)
if a.shard:
    k_, n_ = (int(x) for x in a.shard.split("/"))
    qs = [q for q in qs if zlib.crc32(q["id"].encode()) % n_ == k_ - 1]
if a.limit:
    qs = qs[:a.limit]
judge_cfg = {"provider": a.judge_provider or a.provider, "model": a.judge_model or a.model}
bot = Assistant(Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg),
                {"provider": a.provider, "model": a.model},
                {"provider": a.verifier_provider, "model": a.verifier_model}, top_k=a.top_k)
print(f"{len(qs)} questions | answer {a.provider}:{a.model} | verifier {a.verifier_provider}:{a.verifier_model} | judge {judge_cfg['provider']}:{judge_cfg['model']}")

out = ROOT / "results" / f"pipeline_{a.tag}.jsonl"; out.parent.mkdir(exist_ok=True)
done = {}
if out.exists():
    for line in open(out, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); done[r["id"]] = r
t0 = time.time()
with open(out, "a", encoding="utf-8") as f:
    for i, q in enumerate(qs, 1):
        if q["id"] in done:
            continue
        try:
            res = bot.run(q["question"])
            row = evaluate_question(q, res["hits"], res["answer"], judge_cfg)
            if res["trace"]["overridden"]:                 # draft as it would have been without the verifier
                draft_row = evaluate_question(q, res["hits"], res["trace"]["draft"], judge_cfg)
                row["draft_verdict"], row["draft_abstained"] = draft_row.get("verdict"), draft_row["abstained"]
            else:
                row["draft_verdict"], row["draft_abstained"] = row.get("verdict"), row["abstained"]
        except Exception as e:
            print(f"\n  [{q['id']}] failed: {str(e)[:150]}"); continue
        v = res["trace"]["verifier"] or {}
        row.update(id=q["id"], faithfulness=v.get("faithfulness"), answer_relevance=v.get("answer_relevance"),
                   verifier_veto=res["trace"]["overridden"], draft=res["trace"]["draft"])
        done[q["id"]] = row
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(f"  {i}/{len(qs)}", end="\r")
        if a.sleep:
            time.sleep(a.sleep)

rows = [done[q["id"]] for q in qs if q["id"] in done]
print(f"\n== full pipeline: {len(rows)}/{len(qs)} questions ({time.time() - t0:.0f}s this run) ==")
cols = ["group", "n", "correct", "correct+partial", "token_recall", "retrieval_hit", "cite_valid",
        "cited_gold", "false_abstain", "abstain_ok", "hallucination"]
summ = summarize([dict(r) for r in rows])
print("".join(f"{c[:14]:<{27 if i == 0 else 15}}" for i, c in enumerate(cols)))
for s in summ:
    print("".join(f"{str(s.get(c, '-')):<{27 if i == 0 else 15}}" for i, c in enumerate(cols)))

ans = [r for r in rows if r["answerable"]]
nonab = [r for r in rows if not r["abstained"]]
def frac(rs, k, val): return round(sum(r.get(k) == val for r in rs) / len(rs), 3) if rs else None
print(f"\nVerifier (answers that were not abstentions, n={len(nonab)}):")
print("  faithfulness     ", dict(Counter(r.get("faithfulness") for r in nonab)))
print("  answer relevance ", dict(Counter(r.get("answer_relevance") for r in nonab)))
print(f"  vetoes: {sum(r['verifier_veto'] for r in rows)}")
una = [r for r in rows if not r["answerable"]]
if ans and una:
    print("\nVerifier ablation (same run, final vs draft):")
    print(f"  hallucination on unanswerable : draft {round(sum(not r['draft_abstained'] for r in una)/len(una),3)}"
          f" -> final {round(sum(not r['abstained'] for r in una)/len(una),3)}  (n={len(una)})")
    print(f"  correct on answerable         : draft {frac(ans,'draft_verdict','correct')}"
          f" -> final {frac(ans,'verdict','correct')}  (n={len(ans)})")
    print(f"  false abstain on answerable   : draft {round(sum(r['draft_abstained'] for r in ans)/len(ans),3)}"
          f" -> final {round(sum(r['abstained'] for r in ans)/len(ans),3)}")
with open(ROOT / "results" / f"pipeline_{a.tag}_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for s in summ: w.writerow({c: s.get(c, "") for c in cols})
