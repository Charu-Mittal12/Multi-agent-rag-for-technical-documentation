"""Print retrieval failures compactly.  python scripts/show_failures.py [prefix ...]
Examples: python scripts/show_failures.py            (all)
          python scripts/show_failures.py dewalt electrolux
"""
import csv, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
prefixes = tuple(sys.argv[1:])
path = Path(__file__).resolve().parents[1] / "results" / "retrieval_failures.csv"
for r in csv.DictReader(open(path, encoding="utf-8")):
    if prefixes and not r["id"].startswith(prefixes):
        continue
    print(r["id"], "|", r["category"], "| gold rank:", r["gold_rank_in_top30"] or "none")
    print("  Q:", r["question"][:140])
    print("  gold:", r["gold"], "| got:", r["top_retrieved"])
    print("  ref:", r["reference_answer"][:160].replace("\n", " "))
    print()
