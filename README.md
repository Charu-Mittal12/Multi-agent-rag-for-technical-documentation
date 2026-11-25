# Multi-Agent RAG for Technical Manuals

A Retrieval-Augmented Generation (RAG) system for answering questions from technical PDF manuals using hybrid retrieval, cross-encoder reranking, evidence-grounded answer generation, and an optional verifier agent. The system provides page-level citations and abstains when sufficient evidence is unavailable.

## Architecture

```text
PDF Manuals → Parsing → Structure-aware Chunking → Indexing
                                                      ↓
User Query → BM25 + Dense Retrieval → RRF → Cross-Encoder Reranking
                                                      ↓
                                  Top 5 Chunks → Answer Agent
                                                      ↓
                                      Optional Verifier → Cited Answer
```

## Key Features

- **Document parsing:** PyMuPDF extracts text, tables (Markdown), and numbered diagram legends.
- **Chunking:** Page-aware, paragraph-based chunking with a target of 900 characters and 120-character overlap for long paragraphs.
- **Hybrid retrieval:** BM25 (rank-bm25) and BGE-small-en-v1.5 embeddings with cosine similarity, combined using Reciprocal Rank Fusion (RRF).
- **Reranking:** MS MARCO MiniLM cross-encoder reranks the top 30 candidates to return five chunks.
- **Answer generation:** LLM-generated answers with source citations and evidence-based abstention; optional verifier checks faithfulness and relevance.
- **Interface:** Streamlit application with PDF upload, indexing, question answering, and source-page viewing.

The pipeline is implemented in Python using PyMuPDF, FastEmbed, NumPy, rank-bm25, and Streamlit, without LangChain or an external vector database.

## Installation and Usage

Run the following commands from the project root:

```bash
pip install -r requirements.txt
```

Place PDF manuals in `data/manuals/`. Alternatively, generate a sample manual and evaluation questions:

```bash
pip install reportlab
python scripts/make_sample_manual.py
```

Build the index and query the system:

```bash
python scripts/ingest.py
python scripts/ask.py "What does error E-203 indicate?"
```

Launch the web interface:

```bash
streamlit run app.py
```

The first indexing run downloads the embedding and reranking models. By default, the system returns extractive answers without an LLM API key.

### Enable LLM Answer Generation

For example, install the OpenAI SDK and set `OPENAI_API_KEY` in your environment:

```bash
pip install openai
```

Update the LLM section of `config.yaml`:

```yaml
llm:
  provider: openai
  model: "<model-available-to-your-provider>"
```

Anthropic and Gemini are also supported with their respective SDKs and API keys. Retrieval and chunking parameters can be modified in `config.yaml`.

## Evaluation

Evaluated on 276 usable questions from **TechManualQA-350** (236 answerable and 40 unanswerable). Retrieval ground truth is derived from gold manual-page pairs and evidence snippets. Answer evaluation uses an LLM judge, deterministic citation checks, and abstention metrics.

| Metric | Reported result |
|---|---:|
| Page-level Hit@5 (reported as Recall@5) | 95.3% |
| Strict chunk-level Hit@5 | 92.8% |
| MRR | 0.828 |
| nDCG@10 | 0.866 |
| Strict answer correctness | 80.1% |
| Correct or partially correct | 90.3% |
| Faithfulness (verifier-rated) | 98.2% |
| Hallucination on unanswerable questions | 7.5% |

To reproduce retrieval experiments, obtain the benchmark JSONL and corresponding PDFs (named by dataset `doc_id`):

```bash
python scripts/prepare_techmanualqa.py --src /path/to/TechManualQA_350.jsonl
python scripts/ingest.py
python scripts/evaluate.py --questions data/eval/techmanualqa.jsonl
```

Run end-to-end evaluation with API-accessible model IDs (replace the placeholders):

```bash
python scripts/run_pipeline.py --provider openai --model ANSWER_MODEL \
  --verifier-provider openai --verifier-model VERIFIER_MODEL \
  --judge-provider openai --judge-model JUDGE_MODEL --tag full
python scripts/final_report.py --tag full
```

See [EVALUATION.md](EVALUATION.md) for detailed results, ablation studies, confidence intervals, and limitations.

## Project Structure

```text
src/manual_rag/    # Parsing, embedding, retrieval, reranking, agents, evaluation
scripts/           # Indexing, querying, benchmark preparation and evaluation
app.py             # Streamlit interface
config.yaml        # Pipeline configuration
requirements.txt   # Dependencies
EVALUATION.md      # Detailed experimental results
```

## Limitations and Future Work

- Procedural questions may require evidence across multiple pages.
- Manual routing and verifier ablations did not improve the reported final performance.
- Figure captioning and visual page retrieval are planned; the current system retrieves textual content, tables, and diagram legends.
