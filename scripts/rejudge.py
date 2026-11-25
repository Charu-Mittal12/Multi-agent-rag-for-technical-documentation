"""Re-judge cached answers with a DIFFERENT judge model (no new answers are generated).

Why: when the same model writes and grades answers it can favour its own style and share its blind spots.
Re-grading with another model family, then reporting agreement between the two judges, makes the
correctness numbers far more credible.

  python scripts/rejudge.py --src results/answers_hybrid.jsonl results/answers_hybrid_rerank.jsonl \
      --provider openai --model qwen/qwen3.8-27b --tag qwen --sleep 3

Writes results/answers_<config>__<tag>.jsonl (same rows, new verdict/judge_reason, old one kept as
verdict_orig). Compare configs with:
  python scripts/stats.py --a results/answers_hybrid__qwen.jsonl --b results/answers_hybrid_rerank__qwen.jsonl
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import ROOT
from manual_rag.evaluate import load_questions
from manual_rag.answer_eval import judge

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="cached answer files to re-judge")
ap.add_argument("--questions", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--provider", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tag", default="x", help="short name for the new judge, used in the output file name")
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--in-place", action="store_true",
                help="only re-judge rows whose verdict is 'unparsed'/missing and write the fix back into the SAME file "
                     "(backup .bak). --tag is ignored. Use this to repair judge-output truncation.")
a = ap.parse_args()

ref = {q["id"]: q.get("answer", "") for q in load_questions(ROOT / a.questions)}
cfg = {"provider": a.provider, "model": a.model}
LABELS = ["correct", "partial", "incorrect"]


def kappa(pairs, labels):
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[l] * cy[l] for l in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


for src in a.src:
    src = ROOT / src
    if a.in_place:
        import shutil
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        bad = [r for r in rows if r.get("answerable") and r.get("verdict") not in LABELS and ref.get(r["id"])]
        print(f"{src.name}: {len(bad)} rows with an unparsed/missing verdict")
        fixed = 0
        for r in bad:
            try:
                j = judge(r["question"], ref[r["id"]], r["answer"], cfg)
            except Exception as e:
                print(f"  [{r['id']}] judge failed: {e}"); continue
            if j["verdict"] in LABELS:
                r["verdict_before_fix"], r["verdict"], r["judge_reason"] = r.get("verdict"), j["verdict"], j["reason"]
                fixed += 1
            if a.sleep:
                time.sleep(a.sleep)
        if fixed:
            shutil.copy(src, src.with_suffix(".jsonl.bak2"))
            with open(src, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  repaired {fixed} of {len(bad)}")
        continue
    out = src.with_name(f"{src.stem}__{a.tag}.jsonl")
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("judge") and r.get("verdict") in LABELS:   # only rows with a real new verdict count as done
                    done[r["id"]] = r
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            if not r.get("answerable") or r["id"] in done or not ref.get(r["id"]):
                continue
            try:
                j = judge(r["question"], ref[r["id"]], r["answer"], cfg)
            except Exception as e:
                print(f"  [{r['id']}] judge failed: {e}"); continue
            new = dict(r, verdict_orig=r.get("verdict"), judge_reason_orig=r.get("judge_reason"),
                       verdict=j["verdict"], judge_reason=j["reason"], judge=f"{a.provider}:{a.model}")
            done[r["id"]] = new
            f.write(json.dumps(new, ensure_ascii=False) + "\n"); f.flush()
            if a.sleep:
                time.sleep(a.sleep)
    # refresh the stored copy of the original judge's verdict (the source file may have been repaired since)
    for r in rows:
        if r["id"] in done:
            done[r["id"]]["verdict_orig"] = r.get("verdict")
            done[r["id"]]["judge_reason_orig"] = r.get("judge_reason")
    # keep unanswerable rows too, so stats.py sees the same question set
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(done.get(r["id"], r), ensure_ascii=False) + "\n")
    pairs = [(r["verdict_orig"], r["verdict"]) for r in done.values()
             if r.get("verdict_orig") in LABELS and r.get("verdict") in LABELS]
    print(f"\n{src.name} -> {out.name}: {len(done)} re-judged, {len(pairs)} comparable")
    if pairs:
        agree = sum(x == y for x, y in pairs) / len(pairs)
        both = [(x == "correct", y == "correct") for x, y in pairs]
        print(f"  agreement (3 classes): {agree:.2f}   Cohen's kappa: {kappa(pairs, LABELS):.2f}")
        print(f"  'correct' rate: original judge {sum(x for x, _ in both) / len(both):.2f}"
              f"  vs new judge {sum(y for _, y in both) / len(both):.2f}")
        print("  confusion (original -> new):", dict(Counter(pairs)))
