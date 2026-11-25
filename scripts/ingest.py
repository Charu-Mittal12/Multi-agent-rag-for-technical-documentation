"""Parse all PDFs in data/manuals and build the text index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config
from manual_rag.pipeline import build_index

stats = build_index(load_config(), progress=print)
for name, c in stats["manuals"].items():
    print(f"{name}: {c['text']} text, {c['table']} table, {c['figure']} figure chunks")
print(f"Indexed {stats['chunks']} chunks in {stats['seconds']}s")
