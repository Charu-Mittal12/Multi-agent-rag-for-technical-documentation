"""Usage: python scripts/ask.py "status LED blinks red three times" [--mode hybrid] [--no-rerank]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever
from manual_rag.answer import answer

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
ap.add_argument("--no-rerank", action="store_true")
a = ap.parse_args()
cfg = load_config()
r = Retriever(HybridIndex.load(ROOT / cfg["paths"]["index_dir"]), cfg)
hits = r.search(a.question, a.mode, not a.no_rerank)
print(answer(a.question, hits, cfg["llm"]))
print("\nSources:")
for c in hits:
    print(f"  {c.manual} p.{c.page} ({c.kind})")
