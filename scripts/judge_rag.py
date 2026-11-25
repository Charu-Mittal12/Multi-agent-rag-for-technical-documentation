"""Faithfulness and answer-relevance judging for cached answers (RAGAS-style, judged by an LLM).

For each non-abstaining answer the judge sees the question, the retrieved excerpts that the generator saw
(re-created by re-running the deterministic retrieval) and the answer, and returns
  faithfulness      supported | partial | unsupported   - are the answer's claims stated in the excerpts?
  answer_relevance  relevant  | partial | irrelevant   - does it address what the question asked (right or wrong)?

  python scripts/judge_rag.py --src results/answers_hybrid.jsonl results/answers_hybrid_rerank.jsonl \
      --provider openai --model qwen/qwen3.8-27b --sleep 3

Results are cached in results/ragjudge_<file>.jsonl; reruns resume. Use a judge from a different model family
than the generator. Cost: roughly 1.5-2k tokens per answer.
"""
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever, manual_of
from manual_rag.evaluate import load_questions
from manual_rag.answer import complete, format_context

PROMPT = """You audit answers produced by a retrieval-augmented assistant for technical manuals.

Question: {question}

Retrieved excerpts (the only information the assistant was given):
{context}

Assistant's answer: {answer}

Rate two things. Ignore citation markers such as [manual p.3].
1. faithfulness - is every factual claim in the answer stated in, or directly entailed by, the excerpts?
   supported = all claims are; partial = some are not; unsupported = most are not.
2. answer_relevance - does the answer address what the question asks (whatever its correctness)?
   relevant = yes; partial = only partly or with much off-topic content; irrelevant = no.

Reply with JSON only: {{"faithfulness": "supported" | "partial" | "unsupported", "answer_relevance": "relevant" | "partial" | "irrelevant", "reason": "<one short sentence>"}}"""

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--provider", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--top-k", type=int, default=5)
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=1500, help="room for reasoning models before the JSON verdict")
a = ap.parse_args()

cfg = load_config()
jcfg = {"provider": a.provider, "model": a.model}
qs = {q["id"]: q for q in load_questions(ROOT / a.questions)}
retriever = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)


def parse(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if d.get("faithfulness") in ("supported", "partial", "unsupported") and \
               d.get("answer_relevance") in ("relevant", "partial", "irrelevant"):
                return d
        except Exception:
            pass
    return None


for src in a.src:
    src = ROOT / src
    name, *extra = src.stem.removeprefix("answers_").split("__")
    scope = next((x for x in extra if x in ("router", "oracle")), "all")
    rerank, mode = name.endswith("_rerank"), name.removesuffix("_rerank")
    out = src.with_name(f"ragjudge_{src.stem.removeprefix('answers_')}.jsonl")
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("faithfulness"):
                    done[r["id"]] = r
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not r.get("abstained") and r["id"] in qs and r["id"] not in done]
    print(f"{src.name}: {len(rows)} answers, {sum(1 for r in rows if r.get('abstained'))} abstained, {len(todo)} to judge")
    with open(out, "a", encoding="utf-8") as f:
        for r in todo:
            q = qs[r["id"]]
            manual = (manual_of(q) if scope == "oracle"
                      else retriever.route(q["question"]) if scope == "router" else None)
            hits = retriever.search(q["question"], mode=mode, rerank=rerank, k=a.top_k, manual=manual)
            try:
                raw = complete(PROMPT.format(question=q["question"], context=format_context(hits), answer=r["answer"]),
                               jcfg, max_tokens=a.max_tokens)
            except Exception as e:
                print(f"  [{r['id']}] judge failed: {e}"); continue
            d = parse(raw)
            if d is None:
                print(f"  [{r['id']}] unparsable judge output: {raw[:80]!r}"); continue
            row = {"id": r["id"], "answerable": r.get("answerable"), "faithfulness": d["faithfulness"],
                   "answer_relevance": d["answer_relevance"], "reason": d.get("reason", "")}
            done[r["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if a.sleep:
                time.sleep(a.sleep)
    for label, sel in (("answerable", True), ("unanswerable (answered anyway)", False)):
        rs = [d for d in done.values() if bool(d.get("answerable")) == sel]
        if not rs:
            continue
        fc, rc = Counter(d["faithfulness"] for d in rs), Counter(d["answer_relevance"] for d in rs)
        n = len(rs)
        print(f"  {label}: n={n}  faithful(supported) {fc['supported'] / n:.2f}  "
              f"relevant {rc['relevant'] / n:.2f}   {dict(fc)} / {dict(rc)}")
print("\nNote: faithfulness is judged against the retrieved excerpts, not against the truth. A wrong but "
      "well-grounded answer is faithful; an answer that is right but not in the excerpts is not.")
