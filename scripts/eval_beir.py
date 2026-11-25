"""Run the retrieval stack (BM25 / dense / hybrid, +/- reranker) on a BEIR-format benchmark.

Examples
  python scripts/eval_beir.py --dataset scifact                       # downloads ~3 MB
  python scripts/eval_beir.py --dataset nfcorpus --max-queries 100
  python scripts/eval_beir.py --path data/benchmarks/mydata           # any BEIR-format folder

BEIR format: corpus.jsonl {_id,title,text} | queries.jsonl {_id,text} | qrels/<split>.tsv (query-id, corpus-id, score)
"""
import argparse, csv, io, json, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.config import load_config, ROOT
from manual_rag.parse import Chunk
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever
from manual_rag.embed import make_embedder
from manual_rag.evaluate import evaluate

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
BENCH = ROOT / "data" / "benchmarks"


def fetch(name: str) -> Path:
    target = BENCH / name
    if (target / "corpus.jsonl").exists():
        return target
    BENCH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    data = urllib.request.urlopen(URL.format(name), timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(BENCH)
    return target


def load_beir(path: Path, split: str):
    corpus = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = ((d.get("title") or "") + ". " + (d.get("text") or "")).strip(". ")
    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); queries[d["_id"]] = d["text"]
    qrels: dict[str, list[str]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
    return corpus, queries, qrels


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="BEIR dataset name, e.g. scifact, nfcorpus, fiqa, trec-covid")
    g.add_argument("--path", help="folder in BEIR format")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--no-rerank", action="store_true", help="skip the (slower) reranked runs")
    a = ap.parse_args()

    cfg = load_config()
    path = Path(a.path) if a.path else fetch(a.dataset)
    name = path.name
    corpus, queries, qrels = load_beir(path, a.split)
    ids = list(corpus)
    print(f"{name}: {len(ids)} docs, {len(qrels)} judged queries")

    chunks = [Chunk(id=i, manual=i, page=1, kind="text", text=corpus[i]) for i in ids]
    cache = path / f"emb_{cfg['embeddings']['backend']}_{cfg['embeddings']['model'].replace('/', '_')}.npy"
    emb = make_embedder(cfg["embeddings"])
    if cache.exists():
        vecs = np.load(cache)
    else:
        t0, parts = time.time(), []
        texts = [c.text for c in chunks]
        for s in range(0, len(texts), 256):
            parts.append(emb.embed_documents(texts[s:s + 256]))
            print(f"  embedded {min(s + 256, len(texts))}/{len(texts)}", end="\r")
        vecs = np.concatenate(parts); np.save(cache, vecs)
        print(f"\n  embedding took {time.time() - t0:.0f}s (cached for next time)")

    retriever = Retriever(HybridIndex(chunks, vecs), cfg)
    qids = [q for q in qrels if q in queries]
    if a.max_queries:
        qids = qids[:a.max_queries]
    questions = [{"question": queries[q], "gold": [{"manual": d, "page": 1} for d in qrels[q]]} for q in qids]

    ks = ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@10")
    print(f"\n{'config':<18}" + "".join(f"{m:>11}" for m in ks) + f"{'sec/query':>11}")
    rows = []
    for mode in ["bm25", "dense", "hybrid"]:
        for rr in ([False] if a.no_rerank else [False, True]):
            t0 = time.time()
            s = evaluate(retriever, questions, mode, rr)
            per_q = (time.time() - t0) / max(len(questions), 1)
            label = f"{mode}{'+rerank' if rr else ''}"
            print(f"{label:<18}" + "".join(f"{s[m]:>11}" for m in ks) + f"{per_q:>11.3f}")
            rows.append({"dataset": name, "config": label, **{m: s[m] for m in ks}, "sec_per_query": round(per_q, 3)})
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    f = out / f"{name}_{a.split}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nn={len(questions)} queries · saved {f}")


if __name__ == "__main__":
    main()
