"""Final numbers with 95% bootstrap CIs + rows to hand-check.   python scripts/final_report.py [--tag smoke20b]
Writes results/final_report.md and results/check_rows.csv."""
import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke20b"); a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / "results" / f"pipeline_{a.tag}.jsonl", encoding="utf-8") if l.strip()]
ans = [r for r in rows if r["answerable"]]; una = [r for r in rows if not r["answerable"]]
nonab = [r for r in rows if not r["abstained"]]
rng = random.Random(0)

def ci(vals):
    vals = [float(v) for v in vals]
    if not vals: return "n/a"
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
    return f"{m:.3f}  [{bs[50]:.3f}, {bs[1949]:.3f}]  (n={len(vals)})"

L = ["# Final results (full pipeline)", f"questions: {len(rows)} ({len(ans)} answerable, {len(una)} unanswerable)", "",
     "| Metric | value [95% CI] |", "|---|---|"]
M = [("Correct (strict)", [r.get("verdict") == "correct" for r in ans]),
     ("Correct or partial", [r.get("verdict") in ("correct", "partial") for r in ans]),
     ("Retrieval hit", [r["retrieval_hit"] for r in ans]),
     ("Valid citations", [r["cite_valid"] for r in ans]),
     ("Cites a gold page", [r["cited_gold"] for r in ans]),
     ("False abstain", [r["abstained"] for r in ans]),
     ("Hallucination (unanswerable)", [not r["abstained"] for r in una]),
     ("Faithfulness supported", [r.get("faithfulness") == "supported" for r in nonab if r.get("faithfulness") in ("supported", "partial", "unsupported")]),
     ("Answer relevant", [r.get("answer_relevance") == "relevant" for r in nonab if r.get("answer_relevance") in ("relevant", "partial", "irrelevant")])]
for name, v in M: L.append(f"| {name} | {ci(v)} |")
L += ["", "## Per category (answerable)", "| Category | n | correct | correct+partial | false abstain |", "|---|---|---|---|---|"]
g = defaultdict(list)
for r in ans: g[r["category"]].append(r)
for c, rs in sorted(g.items()):
    f = lambda k: round(sum(k(r) for r in rs) / len(rs), 3)
    L.append(f"| {c} | {len(rs)} | {f(lambda r: r.get('verdict')=='correct')} | {f(lambda r: r.get('verdict') in ('correct','partial'))} | {f(lambda r: r['abstained'])} |")
vet = sum(r.get("verifier_veto", False) for r in rows)
L += ["", f"Verifier vetoes: {vet}. Correct before/after verifier: "
      f"{round(sum(r.get('draft_verdict')=='correct' for r in ans)/len(ans),3)} -> {round(sum(r.get('verdict')=='correct' for r in ans)/len(ans),3)}"]
# possible truncation: non-abstained answers whose last char is not terminal punctuation / citation bracket
trunc = [r for r in nonab if r["answer"].strip() and r["answer"].strip()[-1] not in ".!?]】)*"]
L.append(f"Answers that may be cut off (no terminal punctuation): {len(trunc)} of {len(nonab)}")
(ROOT / "results" / "final_report.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))

pick = []
pick += [r for r in una if not r["abstained"]]                                   # all hallucinations
pick += [r for r in ans if r.get("verdict") == "partial" and r["category"].startswith("Procedural")][:8]
pick += [r for r in ans if r.get("verdict") == "incorrect" and not r["abstained"]][:8]
pick += [r for r in ans if r["abstained"]][:4]
pick += rng.sample([r for r in ans if r.get("verdict") == "correct"], 6)
seen = set(); pick = [r for r in pick if not (r["id"] in seen or seen.add(r["id"]))]
with open(ROOT / "results" / "check_rows.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["id", "category", "verdict", "question", "answer", "judge_reason", "my_check"])
    for r in pick:
        w.writerow([r["id"], r["category"], r.get("verdict", "unanswerable"), r["question"], r["answer"], r.get("judge_reason", ""), ""])
print(f"\nWrote results/final_report.md and results/check_rows.csv ({len(pick)} rows to hand-check)")
