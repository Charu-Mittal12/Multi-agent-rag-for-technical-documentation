"""Paired statistics for two answer-evaluation runs (e.g. hybrid vs hybrid+rerank).

Reads the per-question caches written by eval_answers.py, keeps questions present in BOTH,
and reports for each metric: mean per config with 95% bootstrap CI, the paired difference with
95% CI, how many questions flipped each way, and an exact McNemar-style sign-test p-value.
No API calls are made.

Usage:  python scripts/stats.py [--a results/answers_hybrid.jsonl] [--b results/answers_hybrid_rerank.jsonl]
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--a", default="results/answers_hybrid.jsonl")
ap.add_argument("--b", default="results/answers_hybrid_rerank.jsonl")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)


def load(p):
    rows = {}
    for line in open(ROOT / p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); rows[r["id"]] = r
    return rows


A, B = load(a.a), load(a.b)
ids = sorted(set(A) & set(B))
print(f"A = {a.a} ({len(A)} rows)\nB = {a.b} ({len(B)} rows)\npaired questions: {len(ids)}\n")


def sign_test_p(b, c):
    """Exact two-sided sign test on discordant pairs (b = A-only wins, c = B-only wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return lo, hi


def metric_rows(ids_, getter):
    pairs = []
    for i in ids_:
        va, vb = getter(A[i]), getter(B[i])
        if va is not None and vb is not None:
            pairs.append((float(va), float(vb)))
    return np.array(pairs)


def report(name, ids_, getter, binary=True):
    p = metric_rows(ids_, getter)
    if len(p) == 0:
        return None
    n = len(p)
    idx = rng.integers(0, n, size=(a.boot, n))
    ma, mb = p[idx, 0].mean(1), p[idx, 1].mean(1)
    d = mb - ma
    out = {"metric": name, "n": n, "A": p[:, 0].mean(), "A_lo": ci(ma)[0], "A_hi": ci(ma)[1],
           "B": p[:, 1].mean(), "B_lo": ci(mb)[0], "B_hi": ci(mb)[1],
           "diff": p[:, 1].mean() - p[:, 0].mean(), "d_lo": ci(d)[0], "d_hi": ci(d)[1]}
    if binary:
        wins_b = int(((p[:, 1] == 1) & (p[:, 0] == 0)).sum())   # B right, A wrong
        wins_a = int(((p[:, 0] == 1) & (p[:, 1] == 0)).sum())
        out.update(B_only=wins_b, A_only=wins_a, p=sign_test_p(wins_a, wins_b))
    return out


ans = [i for i in ids if A[i]["answerable"] and B[i]["answerable"]]
una = [i for i in ids if not A[i]["answerable"] and not B[i]["answerable"]]

results = []
specs = [("correct", lambda r: r.get("verdict") == "correct" if "verdict" in r else None, True),
         ("correct+partial", lambda r: r.get("verdict") in ("correct", "partial") if "verdict" in r else None, True),
         ("retrieval_hit", lambda r: r.get("retrieval_hit"), True),
         ("cite_valid", lambda r: r.get("cite_valid"), True),
         ("cited_gold", lambda r: r.get("cited_gold"), True),
         ("false_abstain", lambda r: r.get("abstained"), True),
         ("token_recall", lambda r: r.get("token_recall"), False)]
for name, g, binary in specs:
    r = report(name, ans, g, binary)
    if r:
        results.append(r)
if una:
    r = report("hallucination (unanswerable)", una, lambda r: not r["abstained"], True)
    if r:
        results.append(r)


def f(x): return f"{x:.3f}"


hdr = f"{'metric':<30}{'n':>4}  {'A [95% CI]':<22}{'B [95% CI]':<22}{'B-A [95% CI]':<24}{'A-only':>7}{'B-only':>7}{'p':>7}"
print(hdr); print("-" * len(hdr))
for r in results:
    print(f"{r['metric']:<30}{r['n']:>4}  "
          f"{f(r['A'])} [{f(r['A_lo'])},{f(r['A_hi'])}]  "
          f"{f(r['B'])} [{f(r['B_lo'])},{f(r['B_hi'])}]  "
          f"{r['diff']:+.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}]  "
          f"{r.get('A_only', ''):>5}  {r.get('B_only', ''):>5}  {f(r['p']) if 'p' in r else '':>6}")
print("\nA-only = questions only A got right/true; B-only = only B. p = exact sign test on those flips.")
print("A difference is 'real' only if its CI excludes 0 (and p is small). With few questions it usually will not.")

# per-category correctness, paired
print("\nPer category (verdict == correct):")
cats = sorted({A[i]["category"] for i in ans})
for c in cats:
    ci_ = [i for i in ans if A[i]["category"] == c]
    pa = np.mean([A[i].get("verdict") == "correct" for i in ci_])
    pb = np.mean([B[i].get("verdict") == "correct" for i in ci_])
    print(f"  {c:<38} n={len(ci_):<3} A={pa:.2f}  B={pb:.2f}  diff={pb - pa:+.2f}")

# questions that flipped: for hand-checking
flips = []
for i in ans:
    va, vb = A[i].get("verdict"), B[i].get("verdict")
    if va != vb:
        flips.append({"id": i, "category": A[i]["category"], "A_verdict": va, "B_verdict": vb,
                      "question": A[i]["question"], "A_answer": A[i]["answer"], "B_answer": B[i]["answer"],
                      "A_judge_reason": A[i].get("judge_reason", ""), "B_judge_reason": B[i].get("judge_reason", "")})
out = ROOT / "results"
with open(out / "paired_stats.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(results)
with open(out / "flipped_questions.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(flips[0].keys()) if flips else ["id"]); w.writeheader(); w.writerows(flips)
print(f"\nSaved results/paired_stats.csv and results/flipped_questions.csv ({len(flips)} questions where the verdict differs - read these).")
