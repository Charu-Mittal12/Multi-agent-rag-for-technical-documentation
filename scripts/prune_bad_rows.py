"""Remove empty / cut-off answers from a pipeline cache so the next run redoes only those questions.
   python scripts/prune_bad_rows.py --tag smoke20b      (backup: .bak3)"""
import argparse, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke20b"); a = ap.parse_args()
f = ROOT / "results" / f"pipeline_{a.tag}.jsonl"
rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
bad = lambda r: (not r["answer"].strip()) or (not r["abstained"] and r["answer"].strip()[-1] not in ".!?]】)*")
keep = [r for r in rows if not bad(r)]
shutil.copy(f, f.with_suffix(".jsonl.bak3"))
with open(f, "w", encoding="utf-8") as fh:
    for r in keep: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"removed {len(rows) - len(keep)} empty/cut-off rows, kept {len(keep)}. Now rerun run_pipeline.py with the same tag.")
