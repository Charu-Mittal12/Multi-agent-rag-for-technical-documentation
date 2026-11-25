"""Show what the index holds for one manual page.   python scripts/show_chunks.py dewalt_saw 3 [8 ...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from manual_rag.config import load_config, ROOT
from manual_rag.index import HybridIndex

if len(sys.argv) < 3:
    sys.exit("usage: python scripts/show_chunks.py <manual> <page> [page ...]")
manual, pages = sys.argv[1], {int(p) for p in sys.argv[2:]}
idx = HybridIndex.load(ROOT / load_config()["paths"]["index_dir"])
found = False
for c in idx.chunks:
    if c.manual == manual and c.page in pages:
        found = True
        print(f"--- {c.manual} p.{c.page} [{c.kind}] ({len(c.text)} chars) ---")
        print(c.text[:900]); print()
if not found:
    print("No indexed chunks for that manual/page (figure placeholders are not indexed).")
