# Multimodal Multi-Agent RAG for Technical Manuals

Layered build (each layer is evaluated before the next is added):

| Layer | Status | What |
|---|---|---|
| 1. Text pipeline | **done** | PyMuPDF parsing (text + tables), BM25 + dense, RRF, cross-encoder rerank, cited answer, eval harness |
| 2. Table/figure captions | todo | VLM API (e.g. Gemini Flash) fills `kind="figure"` placeholders and structured tables |
| 3. Visual page retrieval | todo | ColPali/ColQwen (index once on free Colab GPU) vs caption-based retrieval |
| 4. Agents (LangGraph) | todo | router, verifier (+ single-agent baseline for comparison) |
| 5. UI | **done** | Streamlit app (`streamlit run app.py`) with cited-page viewer; failure analysis still todo |

## Run the app
```bash
streamlit run app.py
```
Pages: **Ask** (chat + highlighted source page), **Library** (upload PDFs, build index), **Evaluate** (ablation table).
Optional: pick an LLM provider and paste an API key in the sidebar for written step-by-step answers.

Benchmarks and answer evaluation (TechManualQA, BEIR): see `EVALUATION.md` (`python scripts/eval_beir.py --dataset scifact`).

## Quick start
```bash
pip install -r requirements.txt
# put manuals in data/manuals/*.pdf  (or generate a toy one:)
python scripts/make_sample_manual.py
python scripts/ingest.py                       # builds indexes/text
python scripts/ask.py "status LED blinks red three times"
python scripts/evaluate.py                     # ablation table
```
`MANUAL_RAG_OFFLINE=1` skips model downloads (hash embeddings, no reranker) for smoke tests only;
results from it are NOT meaningful. On a normal machine the first run downloads
`bge-small-en-v1.5` and a MiniLM cross-encoder (CPU-friendly).

## Evaluation set
`data/eval/questions.jsonl`, one JSON per line:
`{"question": "...", "type": "table|text|diagram|photo|unanswerable", "gold": [{"manual": "<pdf stem>", "page": 3}]}`
Use `"gold": []` for questions the manuals cannot answer (tests the verifier's refusal later).
Aim for 50-100 questions on 3-5 real manuals.

## LLM
Set `llm.provider` in `config.yaml` (`anthropic` | `gemini` | `openai`) and the matching API key env var.
`none` returns an extractive cited answer.
