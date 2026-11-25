"""End-to-end answer evaluation: retrieve -> cited answer -> judge against the reference answer.

  python scripts/eval_answers.py --questions data/eval/techmanualqa.jsonl \
      --provider gemini --model gemini-2.0-flash --per-category 10

Metrics (per category and overall)
  correct / correct+partial  LLM-judge verdict vs the reference answer (answerable questions)
  token_recall               share of reference words present in the answer (cheap, judge-free sanity check)
  retrieval_hit              a gold page was among the retrieved passages
  cite_valid                 answer cites at least one page and every cited page was actually retrieved
  cited_gold                 answer cites a gold page
  false_abstain              said "not enough evidence" although the answer exists
  abstain_ok / hallucination (unanswerable) correctly abstained / answered anyway
Results are cached per (config, question id) in results/answers_<config>.jsonl, so reruns resume.
Hand-check ~30 rows of those files; LLM judges make mistakes.
"""
import argparse, csv, json, random, sys, time, zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever, manual_of
from manual_rag.answer import answer
from manual_rag.evaluate import load_questions
from manual_rag.answer_eval import evaluate_question, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--configs", default="hybrid,hybrid+rerank",
                help="comma list from: bm25, dense, hybrid, each optionally +rerank")
ap.add_argument("--provider", required=True, help="anthropic | gemini | openai | mock")
ap.add_argument("--model", default="")
ap.add_argument("--judge-provider", default=None, help="default: same as --provider")
ap.add_argument("--judge-model", default=None, help="default: same as --model")
ap.add_argument("--top-k", type=int, default=5)
ap.add_argument("--per-category", type=int, default=0, help="stratified sample size per category (0 = all)")
ap.add_argument("--limit", type=int, default=0, help="hard cap on number of questions (0 = all)")
ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", default="", help="K/N: run only questions whose id hashes to shard K of N, so several models/keys "
                "can share the work (use the SAME shard-to-model assignment for every config)")
ap.add_argument("--prompt", default="default", choices=["default", "strict"],
                help="answer prompt variant (strict = stricter abstention rule)")
ap.add_argument("--tag", default="", help="suffix for the cache file so variants do not overwrite each other")
ap.add_argument("--only-unanswerable", action="store_true",
                help="evaluate only the unanswerable questions (cheap: no judge calls)")
ap.add_argument("--scope", default="all", choices=["all", "router", "oracle"],
                help="all: search every manual | router: LLM-free router picks the manual | oracle: true manual")
a = ap.parse_args()

cfg = load_config()
llm_cfg = {"provider": a.provider, "model": a.model, "prompt": a.prompt}
judge_cfg = {"provider": a.judge_provider or a.provider, "model": a.judge_model or a.model}
questions = load_questions(ROOT / a.questions)
if a.only_unanswerable:
    questions = [q for q in questions if not q.get("gold")]

if a.per_category:
    rng, by_cat = random.Random(a.seed), defaultdict(list)
    for q in questions:
        by_cat[q.get("category", q.get("type"))].append(q)
    questions = [q for qs in by_cat.values() for q in rng.sample(qs, min(a.per_category, len(qs)))]
if a.shard:
    k_, n_ = (int(x) for x in a.shard.split("/"))
    questions = [q for q in questions if zlib.crc32(q["id"].encode()) % n_ == k_ - 1]
if a.limit:
    questions = questions[:a.limit]
print(f"{len(questions)} questions | answer model: {llm_cfg['provider']}:{llm_cfg['model']} | "
      f"judge: {judge_cfg['provider']}:{judge_cfg['model']}")

retriever = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)
out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
summary_rows = []

for conf in [c.strip() for c in a.configs.split(",") if c.strip()]:
    mode, rerank = conf.replace("+rerank", ""), conf.endswith("+rerank")
    suffix = ("" if a.scope == "all" else f"__{a.scope}") + (f"__{a.tag}" if a.tag else "")
    cache = out_dir / f"answers_{conf.replace('+', '_')}{suffix}.jsonl"
    done = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line); done[r["id"]] = r
    t0 = time.time()
    with open(cache, "a", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            if q["id"] in done:
                continue
            manual = (manual_of(q) if a.scope == "oracle"
                      else retriever.route(q["question"]) if a.scope == "router" else None)
            hits = retriever.search(q["question"], mode=mode, rerank=rerank, k=a.top_k, manual=manual)
            try:
                text = answer(q["question"], hits, llm_cfg)
            except Exception as e:
                print(f"  [{q['id']}] LLM failed: {e}"); continue
            row = evaluate_question(q, hits, text, judge_cfg)
            row["id"] = q["id"]
            row["answer_model"] = f"{a.provider}:{a.model}"
            row["prompt"] = a.prompt
            done[q["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {conf}: {i}/{len(questions)}", end="\r")
            if a.sleep:
                time.sleep(a.sleep)
    rows = [done[q["id"]] for q in questions if q["id"] in done]
    print(f"\n== {conf}  ({len(rows)} answered, {time.time() - t0:.0f}s) ==")
    summ = summarize(rows)
    cols = ["group", "n", "correct", "correct+partial", "token_recall", "retrieval_hit",
            "cite_valid", "cited_gold", "false_abstain", "abstain_ok", "hallucination"]
    W = lambda i: 27 if i == 0 else 15
    print("".join(f"{c[:W(i)-1]:<{W(i)}}" for i, c in enumerate(cols)))
    for s in summ:
        print("".join(f"{str(s.get(c, '-')):<{W(i)}}" for i, c in enumerate(cols)))
        summary_rows.append({"config": conf, **{c: s.get(c, "") for c in cols}})

with open(out_dir / "answers_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["config", "group", "n", "correct", "correct+partial", "token_recall",
                                       "retrieval_hit", "cite_valid", "cited_gold", "false_abstain",
                                       "abstain_ok", "hallucination"])
    w.writeheader(); w.writerows(summary_rows)
print(f"\nSaved results/answers_summary.csv and per-question files in results/ (hand-check some rows).")
