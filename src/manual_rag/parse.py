"""PDF -> chunks. Each chunk keeps manual/page/kind metadata so answers can cite pages.

Layer 1 uses PyMuPDF only. Tables are extracted as markdown; figures are recorded as
placeholders (kind="figure") that a later layer fills with VLM captions.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import re
from pathlib import Path
import pymupdf as fitz  # PyMuPDF


@dataclass
class Chunk:
    id: str
    manual: str
    page: int          # 1-based
    kind: str          # text | table | figure
    text: str
    bbox: list | None = None   # [x0, y0, x1, y1] in PDF points, when known (used for highlighting)

    def to_dict(self):
        return asdict(self)


def _table_to_markdown(rows: list[list]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _extract_table(page, table) -> list[list[str]]:
    """Read each cell by clipping page text to the cell box. table.extract() can drop spaces and
    split tokens like 'E-203', which would break exact-match (BM25) search on error codes."""
    rows = []
    for row in table.rows:
        rows.append([page.get_text("text", clip=fitz.Rect(c)).strip() if c else "" for c in row.cells])
    return rows


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            while len(p) > max_chars:  # hard-split very long paragraphs
                chunks.append(p[:max_chars])
                p = p[max_chars - overlap:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


_ITEM_RE = re.compile(r"^(\d{1,2})\s+(?![\d.])([A-Za-z][A-Za-z0-9 \-/()'\",]{2,60})$")
_FIG_RE = re.compile(r"Fig(?:ure)?\.?\s*([A-Z0-9]{1,3})\b")


def legend_chunk_text(body: str) -> str | None:
    """Numbered diagram legends ("1 Trigger switch / 2 Main handle ...") are easy to miss: the numbers
    are weak search terms and the list often shares a chunk with unrelated text. When a page has a
    figure label plus a run of numbered items, return one compact, self-describing legend chunk."""
    lines = [ln.strip() for ln in body.splitlines()]
    items = [(int(m.group(1)), m.group(2).strip()) for ln in lines if (m := _ITEM_RE.match(ln))]
    if len({n for n, _ in items}) < 4:
        return None
    nums = [n for n, _ in items]
    if max(nums) - min(nums) >= 2 * len(items):      # not a dense 1..N style list
        return None
    first = next(ln for ln in lines if (m := _ITEM_RE.match(ln)))
    before = body[: body.find(first)]
    figs = _FIG_RE.findall(before) or _FIG_RE.findall(body)
    if not figs:
        return None
    label = f"Fig. {figs[-1]}"
    listing = "; ".join(f"{n} = {name}" for n, name in items)
    return (f"Legend of {label} (diagram / figure labels: part and component numbers and their names): "
            f"{listing}")


def parse_pdf(path: str | Path, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    path = Path(path)
    manual = path.stem
    chunks: list[Chunk] = []
    doc = fitz.open(path)
    for pno, page in enumerate(doc, start=1):
        # tables first, so we can exclude their area from running text
        table_rects = []
        try:
            for t in page.find_tables().tables:
                md = _table_to_markdown(_extract_table(page, t))
                if md:
                    table_rects.append(fitz.Rect(t.bbox))
                    chunks.append(Chunk(f"{manual}:p{pno}:t{len(table_rects)}", manual, pno, "table", md,
                                    [float(v) for v in t.bbox]))
        except Exception:
            pass

        # running text, skipping blocks that fall inside a table
        parts = []
        for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
            r = fitz.Rect(x0, y0, x1, y1)
            if any(r.intersects(tr) for tr in table_rects):
                continue
            parts.append(txt.strip())
        body = "\n\n".join(p for p in parts if p)
        for i, piece in enumerate(_split_text(body, max_chars, overlap)):
            chunks.append(Chunk(f"{manual}:p{pno}:x{i}", manual, pno, "text", piece))

        legend = legend_chunk_text(body)
        if legend:
            chunks.append(Chunk(f"{manual}:p{pno}:lg", manual, pno, "legend", legend))

        # figure placeholders (captions added by the VLM layer later)
        for i, _img in enumerate(page.get_images(full=True)):
            chunks.append(Chunk(f"{manual}:p{pno}:f{i}", manual, pno, "figure",
                                f"[figure on page {pno} of {manual}; caption pending]"))
    doc.close()
    return chunks
