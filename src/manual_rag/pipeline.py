"""Shared helpers used by both the CLI scripts and the Streamlit app."""
from __future__ import annotations
import time
from pathlib import Path
import pymupdf
from .config import ROOT
from .parse import parse_pdf, Chunk
from .embed import make_embedder
from .index import HybridIndex


def list_manuals(cfg) -> list[Path]:
    return sorted((ROOT / cfg["paths"]["manuals_dir"]).glob("*.pdf"))


def index_worthy(c: Chunk) -> bool:
    """Figure placeholders carry no content until a VLM caption replaces them; indexing them only adds noise."""
    return not (c.kind == "figure" and "caption pending" in c.text)


def build_index(cfg, progress=None) -> dict:
    """Parse every PDF in the manuals dir, embed, and save the index. Returns summary stats."""
    say = progress or (lambda msg: None)
    pdfs = list_manuals(cfg)
    if not pdfs:
        raise FileNotFoundError("No PDFs in data/manuals/. Add manuals first.")
    t0, chunks, per_manual = time.time(), [], {}
    for p in pdfs:
        say(f"Parsing {p.name} ...")
        cs = parse_pdf(p, cfg["chunking"]["max_chars"], cfg["chunking"]["overlap_chars"])
        per_manual[p.stem] = {k: sum(c.kind == k for c in cs) for k in ("text", "table", "figure")}
        chunks += cs
    n_all = len(chunks)
    chunks = [c for c in chunks if index_worthy(c)]
    say(f"Skipping {n_all - len(chunks)} figure placeholders (no caption yet); keeping {len(chunks)} chunks")
    say(f"Embedding {len(chunks)} chunks (first run downloads the model) ...")
    vecs = make_embedder(cfg["embeddings"]).embed_documents([c.text for c in chunks])
    HybridIndex(chunks, vecs).save(ROOT / cfg["paths"]["index_dir"])
    return {"chunks": len(chunks), "manuals": per_manual, "seconds": round(time.time() - t0, 1)}


def index_path(cfg) -> Path:
    return ROOT / cfg["paths"]["index_dir"]


def index_exists(cfg) -> bool:
    return (index_path(cfg) / "chunks.jsonl").exists()


def render_page(cfg, chunk: Chunk, zoom: float = 1.6) -> bytes:
    """Render the chunk's PDF page as PNG, with the chunk's region highlighted when it can be found."""
    pdf = ROOT / cfg["paths"]["manuals_dir"] / f"{chunk.manual}.pdf"
    doc = pymupdf.open(pdf)
    page = doc[chunk.page - 1]
    rects = []
    if chunk.bbox:
        rects = [pymupdf.Rect(chunk.bbox)]
    elif chunk.kind == "text":
        # locate the chunk's opening words on the page
        first = " ".join(chunk.text.split())[:60]
        rects = page.search_for(first) or page.search_for(first[:30])
    for r in rects:
        page.draw_rect(r, color=(1, 0.55, 0), fill=(1, 0.85, 0.2), fill_opacity=0.35, width=1.2)
    png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    doc.close()
    return png
