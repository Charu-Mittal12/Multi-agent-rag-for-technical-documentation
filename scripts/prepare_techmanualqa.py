"""Convert TechManualQA to this project's question format and verify your PDFs match the dataset.

  python scripts/prepare_techmanualqa.py --src TechManualQA_350/TechManualQA_350/TechManualQA_350.jsonl

Put the dataset's manuals in data/manuals/ named exactly like `doc_id` (e.g. bosch_oven.pdf).
For every manual present, the gold answer text is searched for in the PDF and the manual is put in one of three modes:
  offset    gold page + a constant offset matches for >= --min-aligned of questions  -> same edition; use offset
  relocate  the text is found (>= --min-anywhere) but pages don't line up (different pagination/layout)
            -> each question's gold page is re-derived as the page where its answer text is actually found
            (nearest to the dataset's page); questions whose text is not found are dropped
  exclude   the text is mostly not found -> a different edition/manual; its questions are NOT written
Report which mode each manual used: 'relocate' changes the gold labels (the evidence text stays the same).
Output: data/eval/techmanualqa.jsonl
"""
import argparse, ast, json, re, sys
from collections import defaultdict
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="path to TechManualQA_350.jsonl")
ap.add_argument("--out", default="data/eval/techmanualqa.jsonl")
ap.add_argument("--all", action="store_true", help="also write questions whose PDF is missing")
ap.add_argument("--max-offset", type=int, default=6)
ap.add_argument("--min-aligned", type=float, default=0.70)
ap.add_argument("--min-anywhere", type=float, default=0.80)
a = ap.parse_args()

cfg = load_config()
mdir = ROOT / cfg["paths"]["manuals_dir"]
squash = lambda s: re.sub(r"[^a-z0-9]", "", s.replace("\\n", " ").lower())

rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["doc_id"]].append(r)


def key_text(r) -> str:
    """Text we expect to appear verbatim on the gold page."""
    steps = r.get("parsed_steps")
    if isinstance(steps, str) and steps.startswith("["):
        try:
            steps = ast.literal_eval(steps)
        except Exception:
            steps = None
    if isinstance(steps, list) and steps:
        return str(steps[0])
    return str(r["gt_answer_snippet"])


def is_unans(r) -> bool:
    return r["category"] == "Unanswerable" or not str(r["gt_page_number"]).isdigit()


mode, offsets, page_texts, report = {}, {}, {}, []
for doc, rs in sorted(by_doc.items()):
    pdf = mdir / doc
    if not pdf.exists():
        report.append((doc, "MISSING", "-", "-", "-", "-"))
        continue
    with pymupdf.open(pdf) as d:
        pages = [squash(p.get_text()) for p in d]
    page_texts[doc] = pages
    ans = [r for r in rs if not is_unans(r) and squash(key_text(r))]
    n = max(len(ans), 1)
    best = (-1, 0)
    for off in range(-a.max_offset, a.max_offset + 1):
        hit = sum(1 <= int(r["gt_page_number"]) + off <= len(pages)
                  and squash(key_text(r)) in pages[int(r["gt_page_number"]) + off - 1] for r in ans)
        best = max(best, (hit, off))
    anywhere = sum(any(squash(key_text(r)) in pg for pg in pages) for r in ans)
    offsets[doc] = best[1]
    if best[0] / n >= a.min_aligned:
        mode[doc] = "offset"
    elif anywhere / n >= a.min_anywhere:
        mode[doc] = "relocate"
    else:
        mode[doc] = "exclude"
    report.append((doc, f"{len(pages)} pages", f"{best[1]:+d}", f"{best[0]/n:.0%}", f"{anywhere/n:.0%}", mode[doc].upper()))

print(f"{'manual':<24}{'pdf':<12}{'offset':<8}{'on gold page':<14}{'anywhere':<10}{'mode'}")
for r in report:
    print(f"{r[0]:<24}{r[1]:<12}{r[2]:<8}{r[3]:<14}{r[4]:<10}{r[5]}")
print("\nOFFSET = right edition. RELOCATE = right text, different pagination (gold pages re-derived from the text).\n"
      "EXCLUDE = text mostly absent: wrong edition; get the edition the authors used, or leave the manual out.")

out = ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
kept = dropped = relocated = 0
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        doc = r["doc_id"]
        if doc not in mode and not (a.all and doc not in page_texts):
            continue
        if mode.get(doc) == "exclude":
            continue
        stem = doc.removesuffix(".pdf")
        unans = is_unans(r)
        gold = []
        if not unans:
            g = int(r["gt_page_number"])
            if mode.get(doc) == "relocate":
                k = squash(key_text(r))
                hits = [i + 1 for i, p in enumerate(page_texts[doc]) if k and k in p]
                if not hits:
                    dropped += 1
                    continue
                new = min(hits, key=lambda p: abs(p - (g + offsets[doc])))
                relocated += new != g
                g = new
            else:
                g += offsets.get(doc, 0)
            gold = [{"manual": stem, "page": g}]
        q = {"id": r["question_id"], "question": r["question_text"], "category": r["category"],
             "type": "unanswerable" if unans else r["category"].lower().replace("/", "_").replace(" ", "_"),
             "persona": r["persona"], "gold": gold}
        if not unans:
            q["answer"] = str(r["gt_answer_snippet"]).replace("\\n", "\n")
            q["evidence"] = key_text(r).replace("\\n", "\n")   # text expected on the gold page (chunk-level labels)
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
        kept += 1
print(f"\nWrote {kept} questions -> {out}  (relocated gold pages: {relocated}, dropped because text not found: {dropped})")
for doc, m in mode.items():
    if m == "exclude":
        print(f"EXCLUDED {doc}: wrong edition - its questions are not in the output.")
missing = [r[0] for r in report if r[1] == "MISSING"]
if missing:
    print("Missing PDFs (add to data/manuals/):", ", ".join(missing))
if kept == 0:
    sys.exit("No usable manuals found.")
